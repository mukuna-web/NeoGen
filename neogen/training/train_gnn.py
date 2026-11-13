"""
GNN Training Loop
==================
Trains the NeoGen GNN on our PK dataset.

Key features:
- Multi-task training: all 5 PK params trained simultaneously
- Physics-informed loss: enforces PK relationships
- MPS acceleration: uses Apple Silicon GPU if available
- Early stopping: prevents overfitting
- Checkpointing: saves best model
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_absolute_error
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from neogen.models.gnn import NeoGenGNN, PhysicsInformedLoss, smiles_to_graph, N_ATOM_FEATURES

logger = logging.getLogger(__name__)

PK_PARAMS = ["cmax", "auc", "half_life", "clearance", "vd"]


class PKDataset:
    """
    Dataset that converts our parquet splits into PyG graph batches.

    Each sample is a (graph, target_dict, mask_dict) tuple where:
    - graph: molecular graph from SMILES
    - target_dict: {pk_param: log10_value} for available params
    - mask_dict: {pk_param: True/False} indicating which params have labels
    """

    def __init__(self, split_path: str):
        self.df = pd.read_parquet(split_path)

        # Group by molecule — each molecule may have multiple PK params
        self.molecules = []
        grouped = self.df.groupby("canonical_smiles")

        for smiles, group in grouped:
            graph = smiles_to_graph(smiles)
            if graph is None:
                continue

            targets = {}
            masks = {}
            for param in PK_PARAMS:
                param_rows = group[group["pk_param"] == param]
                if len(param_rows) > 0:
                    # Use median if multiple measurements
                    targets[param] = param_rows["log_value"].median()
                    masks[param] = True
                else:
                    targets[param] = 0.0
                    masks[param] = False

            self.molecules.append({
                "graph": graph,
                "targets": targets,
                "masks": masks,
                "smiles": smiles,
            })

        logger.info("Dataset: %d molecules from %d records", len(self.molecules), len(self.df))

    def __len__(self):
        return len(self.molecules)

    def __getitem__(self, idx):
        mol = self.molecules[idx]
        graph = mol["graph"].clone()

        # Attach targets and masks to the graph
        for param in PK_PARAMS:
            graph[f"y_{param}"] = torch.tensor([mol["targets"][param]], dtype=torch.float)
            graph[f"mask_{param}"] = torch.tensor([mol["masks"][param]], dtype=torch.bool)

        return graph


def collate_fn(batch):
    """Custom collate to handle multi-target graphs."""
    return Batch.from_data_list(batch)


def train_epoch(model, loader, optimizer, loss_fn, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        predictions = model(batch)

        # Extract targets and masks from batch
        targets = {p: batch[f"y_{p}"] for p in PK_PARAMS}
        masks = {p: batch[f"mask_{p}"] for p in PK_PARAMS}

        loss, details = loss_fn(predictions, targets, masks)
        loss.backward()

        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Evaluate on validation/test set."""
    model.eval()
    all_preds = {p: [] for p in PK_PARAMS}
    all_targets = {p: [] for p in PK_PARAMS}
    total_loss = 0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        predictions = model(batch)

        targets = {p: batch[f"y_{p}"] for p in PK_PARAMS}
        masks = {p: batch[f"mask_{p}"] for p in PK_PARAMS}

        loss, _ = loss_fn(predictions, targets, masks)
        total_loss += loss.item()
        n_batches += 1

        for p in PK_PARAMS:
            mask = masks[p]
            if mask.any():
                all_preds[p].append(predictions[p][mask].cpu().numpy())
                all_targets[p].append(targets[p][mask].cpu().numpy())

    # Compute metrics
    metrics = {"loss": total_loss / max(n_batches, 1)}
    for p in PK_PARAMS:
        if all_preds[p]:
            preds = np.concatenate(all_preds[p])
            trues = np.concatenate(all_targets[p])
            metrics[f"{p}_r2"] = r2_score(trues, preds)
            metrics[f"{p}_mae"] = mean_absolute_error(trues, preds)
        else:
            metrics[f"{p}_r2"] = 0.0
            metrics[f"{p}_mae"] = 0.0

    return metrics


def train_model(
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    physics_weight: float = 0.1,
    patience: int = 15,
    device: str | None = None,
):
    """Main training function."""
    # Select device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    logger.info("Using device: %s", device)

    # Load datasets
    print("Loading datasets...")
    train_ds = PKDataset("data/splits/train.parquet")
    val_ds = PKDataset("data/splits/val.parquet")
    test_ds = PKDataset("data/splits/test.parquet")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_ds)} molecules, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Initialize model
    model = NeoGenGNN(n_atom_features=N_ATOM_FEATURES).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    loss_fn = PhysicsInformedLoss(physics_weight=physics_weight)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Training loop
    best_val_loss = float("inf")
    best_metrics = None
    no_improve = 0
    history = []

    print(f"\nTraining for {epochs} epochs (patience={patience})...")
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        val_loss = val_metrics["loss"]

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Print progress
        r2_str = " | ".join(
            f"{p[:4]}={val_metrics.get(f'{p}_r2', 0):.3f}"
            for p in PK_PARAMS
        )
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"R²: {r2_str} | "
            f"lr={current_lr:.1e} | {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics.copy()
            no_improve = 0

            # Save best model
            Path("models/checkpoints").mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_metrics": val_metrics,
            }, "models/checkpoints/gnn_best.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Load best model and evaluate on test set
    print("\n" + "=" * 80)
    print("FINAL EVALUATION (Test Set)")
    print("=" * 80)

    checkpoint = torch.load("models/checkpoints/gnn_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, loss_fn, device)

    print(f"\nBest model from epoch {checkpoint['epoch']}:")
    for p in PK_PARAMS:
        r2 = test_metrics.get(f"{p}_r2", 0)
        mae = test_metrics.get(f"{p}_mae", 0)
        print(f"  {p:12s}: R²={r2:.3f}, MAE={mae:.3f}")

    # Save results
    results = {
        "test_metrics": test_metrics,
        "best_val_metrics": best_metrics,
        "training_history": history,
        "model_params": total_params,
        "device": device,
        "epochs_trained": len(history),
    }
    with open("reports/benchmarks/gnn_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return model, test_metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    print("=" * 60)
    print("NeoGen — GNN Training")
    print("=" * 60)

    model, metrics = train_model(
        epochs=50,
        batch_size=128,
        lr=1e-3,
        physics_weight=0.1,
        patience=12,
        device="cpu",  # MPS has issues with PyG DataLoader
    )

    # Compare with baselines
    print("\n" + "=" * 60)
    print("COMPARISON: GNN vs XGBoost")
    print("=" * 60)

    baseline_path = Path("reports/benchmarks/baseline_benchmarks.json")
    if baseline_path.exists():
        with open(baseline_path) as f:
            baselines = json.load(f)

        for p in PK_PARAMS:
            gnn_r2 = metrics.get(f"{p}_r2", 0)
            xgb_r2 = baselines.get(p, {}).get("xgboost", {}).get("test_r2", 0)
            diff = (gnn_r2 - xgb_r2) / max(abs(xgb_r2), 0.001) * 100
            winner = "GNN" if gnn_r2 > xgb_r2 else "XGB"
            print(f"  {p:12s}: GNN R²={gnn_r2:.3f} vs XGB R²={xgb_r2:.3f}  ({diff:+.1f}%)  [{winner}]")
