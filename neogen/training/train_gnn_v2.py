"""
GNN v2 Training Loop
=====================
Trains the improved NeoGen GNN with:
- Cosine annealing with warm restarts
- Gradient accumulation for effective larger batches
- Better logging and checkpointing
- Automatic comparison with baselines
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_absolute_error
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from neogen.models.gnn_v2 import (
    NeoGenGNNv2,
    ImprovedPhysicsLoss,
    smiles_to_graph_v2,
    N_ATOM_FEATURES_V2,
)

logger = logging.getLogger(__name__)

PK_PARAMS = ["cmax", "auc", "half_life", "clearance", "vd"]


class PKDatasetV2:
    """
    Enhanced dataset using v2 graph features.
    Groups by molecule, aggregates multiple PK measurements.
    """

    def __init__(self, split_path: str, max_molecules: int | None = None):
        self.df = pd.read_parquet(split_path)
        self.molecules = []

        grouped = self.df.groupby("canonical_smiles")
        count = 0

        for smiles, group in grouped:
            if max_molecules and count >= max_molecules:
                break

            graph = smiles_to_graph_v2(smiles)
            if graph is None:
                continue

            targets = {}
            masks = {}
            for param in PK_PARAMS:
                param_rows = group[group["pk_param"] == param]
                if len(param_rows) > 0:
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
            count += 1

        logger.info("Dataset: %d molecules from %d records", len(self.molecules), len(self.df))

    def __len__(self):
        return len(self.molecules)

    def __getitem__(self, idx):
        mol = self.molecules[idx]
        graph = mol["graph"].clone()
        for param in PK_PARAMS:
            graph[f"y_{param}"] = torch.tensor([mol["targets"][param]], dtype=torch.float)
            graph[f"mask_{param}"] = torch.tensor([mol["masks"][param]], dtype=torch.bool)
        return graph


def train_epoch(model, loader, optimizer, loss_fn, device, accum_steps=1):
    """Train one epoch with optional gradient accumulation."""
    model.train()
    total_loss = 0
    n_batches = 0

    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        predictions = model(batch)
        targets = {p: batch[f"y_{p}"] for p in PK_PARAMS}
        masks = {p: batch[f"mask_{p}"] for p in PK_PARAMS}

        loss, _ = loss_fn(predictions, targets, masks)
        loss = loss / accum_steps
        loss.backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Evaluate on val/test set."""
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
            if masks[p].any():
                all_preds[p].append(predictions[p][masks[p]].cpu().numpy())
                all_targets[p].append(targets[p][masks[p]].cpu().numpy())

    metrics = {"loss": total_loss / max(n_batches, 1)}
    for p in PK_PARAMS:
        if all_preds[p]:
            preds = np.concatenate(all_preds[p])
            trues = np.concatenate(all_targets[p])
            metrics[f"{p}_r2"] = float(r2_score(trues, preds))
            metrics[f"{p}_mae"] = float(mean_absolute_error(trues, preds))
        else:
            metrics[f"{p}_r2"] = 0.0
            metrics[f"{p}_mae"] = 0.0

    return metrics


def train_model_v2(
    epochs: int = 150,
    batch_size: int = 32,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    physics_weight: float = 0.15,
    patience: int = 25,
    device: str | None = None,
    max_molecules: int | None = None,
    accum_steps: int = 2,
):
    """Main training function for GNN v2."""
    if device is None:
        device = "cpu"  # MPS has issues with PyG
    logger.info("Using device: %s", device)

    # Load datasets
    print("Loading datasets (v2 features — this takes a minute)...")
    t0 = time.time()
    train_ds = PKDatasetV2("data/splits/train.parquet", max_molecules=max_molecules)
    val_ds = PKDatasetV2("data/splits/val.parquet")
    test_ds = PKDatasetV2("data/splits/test.parquet")
    print(f"Datasets loaded in {time.time()-t0:.0f}s")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} molecules, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"Effective batch size: {batch_size * accum_steps}")

    # Initialize model
    model = NeoGenGNNv2(
        n_atom_features=N_ATOM_FEATURES_V2,
        hidden_dim=256,
        n_layers=5,
        dropout=0.15,
    ).to(device)

    # AdamW with weight decay (better regularization than Adam)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine annealing with warm restarts every 30 epochs
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)

    loss_fn = ImprovedPhysicsLoss(physics_weight=physics_weight)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (all trainable)")

    # Training loop
    best_val_loss = float("inf")
    best_val_r2_avg = -float("inf")
    best_metrics = None
    no_improve = 0
    history = []

    print(f"\nTraining for up to {epochs} epochs (patience={patience})...")
    print(f"LR schedule: Cosine annealing (T0=30, min_lr=1e-6)")
    print("-" * 90)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, accum_steps)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        val_loss = val_metrics["loss"]

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Average R² across all parameters (our primary metric)
        avg_r2 = np.mean([val_metrics.get(f"{p}_r2", 0) for p in PK_PARAMS])

        r2_str = " | ".join(
            f"{p[:4]}={val_metrics.get(f'{p}_r2', 0):.3f}"
            for p in PK_PARAMS
        )
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"avg_R²={avg_r2:.3f} | {r2_str} | "
            f"lr={current_lr:.1e} | {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "avg_r2": avg_r2,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        # Track improvement based on average R² (more stable than val_loss)
        improved = False
        if avg_r2 > best_val_r2_avg:
            best_val_r2_avg = avg_r2
            improved = True
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improved = True

        if improved:
            best_metrics = val_metrics.copy()
            no_improve = 0

            Path("models/checkpoints").mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "avg_r2": avg_r2,
            }, "models/checkpoints/gnn_v2_best.pt")

            if epoch > 5:
                print(f"  >>> New best! avg_R²={avg_r2:.3f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Final evaluation on test set
    print("\n" + "=" * 90)
    print("FINAL EVALUATION — GNN v2 (Test Set)")
    print("=" * 90)

    checkpoint = torch.load("models/checkpoints/gnn_v2_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, loss_fn, device)

    print(f"\nBest model from epoch {checkpoint['epoch']}:")
    for p in PK_PARAMS:
        r2 = test_metrics.get(f"{p}_r2", 0)
        mae = test_metrics.get(f"{p}_mae", 0)
        print(f"  {p:12s}: R²={r2:.3f}, MAE={mae:.3f}")

    avg_test_r2 = np.mean([test_metrics.get(f"{p}_r2", 0) for p in PK_PARAMS])
    print(f"\n  Average R²: {avg_test_r2:.3f}")

    # Compare with baselines
    print("\n" + "=" * 90)
    print("COMPARISON: GNN v2 vs GNN v1 vs XGBoost")
    print("=" * 90)

    baseline_path = Path("reports/benchmarks/baseline_benchmarks.json")
    gnn_v1_path = Path("reports/benchmarks/gnn_results.json")

    xgb_metrics = {}
    gnn_v1_metrics = {}

    if baseline_path.exists():
        with open(baseline_path) as f:
            baselines = json.load(f)
        for p in PK_PARAMS:
            xgb_metrics[p] = baselines.get(p, {}).get("xgboost", {}).get("test_r2", 0)

    if gnn_v1_path.exists():
        with open(gnn_v1_path) as f:
            v1_data = json.load(f)
        for p in PK_PARAMS:
            gnn_v1_metrics[p] = v1_data.get("test_metrics", {}).get(f"{p}_r2", 0)

    print(f"\n  {'Parameter':<14s} {'GNN v2':>8s} {'GNN v1':>8s} {'XGBoost':>8s}  {'v2 vs XGB':>10s}  Winner")
    print("  " + "-" * 70)

    wins = {"gnn_v2": 0, "gnn_v1": 0, "xgboost": 0}
    for p in PK_PARAMS:
        v2_r2 = test_metrics.get(f"{p}_r2", 0)
        v1_r2 = gnn_v1_metrics.get(p, 0)
        xgb_r2 = xgb_metrics.get(p, 0)

        best = max(v2_r2, v1_r2, xgb_r2)
        if best == v2_r2:
            winner = "GNN v2"
            wins["gnn_v2"] += 1
        elif best == xgb_r2:
            winner = "XGBoost"
            wins["xgboost"] += 1
        else:
            winner = "GNN v1"
            wins["gnn_v1"] += 1

        diff = ((v2_r2 - xgb_r2) / max(abs(xgb_r2), 0.001)) * 100
        sign = "+" if diff > 0 else ""
        print(f"  {p:<14s} {v2_r2:>7.3f}  {v1_r2:>7.3f}  {xgb_r2:>7.3f}   {sign}{diff:>6.1f}%    {winner}")

    print(f"\n  Wins: GNN v2={wins['gnn_v2']}, XGBoost={wins['xgboost']}, GNN v1={wins['gnn_v1']}")

    # Save results
    results = {
        "test_metrics": test_metrics,
        "best_val_metrics": best_metrics,
        "avg_test_r2": avg_test_r2,
        "training_history": history,
        "model_params": total_params,
        "device": device,
        "epochs_trained": len(history),
        "best_epoch": checkpoint["epoch"],
        "improvements": {
            "architecture": "GINConv with Jumping Knowledge",
            "pooling": "Attention + Mean + Max triple pooling",
            "features": f"{N_ATOM_FEATURES_V2} atom features + {10} molecular descriptors",
            "loss": "Weighted Huber + physics constraints",
            "optimizer": f"AdamW (lr={lr}, wd={weight_decay})",
            "scheduler": "CosineAnnealingWarmRestarts (T0=30)",
        },
    }
    Path("reports/benchmarks").mkdir(parents=True, exist_ok=True)
    with open("reports/benchmarks/gnn_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to reports/benchmarks/gnn_v2_results.json")
    return model, test_metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    print("=" * 60)
    print("NeoGen — GNN v2 Training")
    print("=" * 60)

    model, metrics = train_model_v2(
        epochs=150,
        batch_size=32,
        lr=5e-4,
        weight_decay=1e-4,
        physics_weight=0.15,
        patience=25,
        device="cpu",
        accum_steps=2,  # Effective batch size = 64
    )
