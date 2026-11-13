"""
Graph Neural Network for PK Prediction
========================================
This is the CORE AI model of NeoGen.

How it works:
1. SMILES string -> molecular graph (atoms=nodes, bonds=edges)
2. GNN encoder learns atom-level features (3-4 graph conv layers)
3. Global pooling aggregates atom features into one molecule vector
4. 5 prediction heads output one PK parameter each

Architecture:
    SMILES -> Graph -> [GCN -> GCN -> GCN -> GCN] -> Pool -> 256-dim
                                                          ├── Head_cmax -> 1
                                                          ├── Head_auc -> 1
                                                          ├── Head_halflife -> 1
                                                          ├── Head_cl -> 1
                                                          └── Head_vd -> 1

Why this beats fingerprints:
- Fingerprints are fixed templates. GNN LEARNS what patterns matter.
- GNN preserves spatial relationships between atoms.
- Multi-task learning: predicting 5 params together helps each prediction
  (if the model learns "this group increases clearance", it helps predict Vd too).
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# SMILES -> Graph Conversion
# ══════════════════════════════════════════════════════════════

# Atom features we extract from each atom in the molecule
ATOM_FEATURES = {
    "element": [6, 7, 8, 9, 15, 16, 17, 35, 53],  # C, N, O, F, P, S, Cl, Br, I
    "degree": [0, 1, 2, 3, 4, 5],
    "charge": [-2, -1, 0, 1, 2],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
    ],
}

# Bond features
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def one_hot(value, options):
    """Create a one-hot vector. Returns [0,0,...,1,...,0] where 1 is at the value's position."""
    vec = [0] * (len(options) + 1)  # +1 for "other"
    if value in options:
        vec[options.index(value)] = 1
    else:
        vec[-1] = 1  # "other" category
    return vec


def atom_features(atom):
    """
    Extract features from a single atom.

    Features include:
    - Element type (one-hot): which element is this? C, N, O, etc.
    - Degree (one-hot): how many bonds does it have?
    - Formal charge (one-hot): is it charged?
    - Hybridization (one-hot): sp, sp2, sp3?
    - Is aromatic? (binary)
    - Is in ring? (binary)
    - Number of hydrogens (numeric)

    Total: 9+1 + 6+1 + 5+1 + 3+1 + 2 + 1 = 30 features per atom
    """
    features = []
    features += one_hot(atom.GetAtomicNum(), ATOM_FEATURES["element"])
    features += one_hot(atom.GetTotalDegree(), ATOM_FEATURES["degree"])
    features += one_hot(atom.GetFormalCharge(), ATOM_FEATURES["charge"])
    features += one_hot(atom.GetHybridization(), ATOM_FEATURES["hybridization"])
    features.append(1 if atom.GetIsAromatic() else 0)
    features.append(atom.IsInRing())
    features.append(atom.GetTotalNumHs() / 4.0)  # Normalize
    return features


def smiles_to_graph(smiles: str) -> Data | None:
    """
    Convert a SMILES string into a PyTorch Geometric graph.

    The graph has:
    - x: Node features (atom features, shape: [n_atoms, n_features])
    - edge_index: Edge connectivity (bonds, shape: [2, n_bonds*2])
    - edge_attr: Edge features (bond types, shape: [n_bonds*2, n_bond_features])

    Why *2 for bonds? Because each bond is represented twice
    (atom A -> atom B AND atom B -> atom A) for undirected graph.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features (atoms)
    atom_feats = []
    for atom in mol.GetAtoms():
        atom_feats.append(atom_features(atom))
    x = torch.tensor(atom_feats, dtype=torch.float)

    # Edge index and features (bonds)
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_feat = one_hot(bond.GetBondType(), BOND_TYPES)
        bond_feat.append(1 if bond.IsInRing() else 0)

        # Add both directions (undirected graph)
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_attrs.append(bond_feat)
        edge_attrs.append(bond_feat)

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # GCNConv doesn't use edge_attr, so we omit it to avoid batching size mismatches
    return Data(x=x, edge_index=edge_index)


# Number of features per atom (computed from the feature encoding)
N_ATOM_FEATURES = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))


# ══════════════════════════════════════════════════════════════
# GNN MODEL
# ══════════════════════════════════════════════════════════════

class PKPredictionHead(nn.Module):
    """
    A single prediction head for one PK parameter.

    Architecture: 256 -> 128 -> 64 -> 1
    Each head is specialized for its PK parameter.
    """

    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class NeoGenGNN(nn.Module):
    """
    NeoGen's Graph Neural Network for multi-task PK prediction.

    Architecture:
    1. Input: Atom features (30-dim per atom)
    2. 4 GCN layers with batch norm and residual connections
       30 -> 128 -> 128 -> 256 -> 256
    3. Global pooling: mean + max concatenated -> 512-dim
    4. Shared MLP: 512 -> 256
    5. 5 specialized prediction heads, one per PK param

    Key design choices:
    - Residual connections: prevent gradient vanishing in deeper layers
    - BatchNorm: stabilize training
    - Mean+Max pooling: captures both average and extreme atom features
    - Dropout: prevent overfitting (MC Dropout for uncertainty later)
    """

    PK_PARAMS = ["cmax", "auc", "half_life", "clearance", "vd"]

    def __init__(
        self,
        n_atom_features: int = N_ATOM_FEATURES,
        hidden_dim: int = 128,
        embedding_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Graph convolution layers
        self.conv1 = GCNConv(n_atom_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.conv3 = GCNConv(hidden_dim, embedding_dim)
        self.bn3 = nn.BatchNorm1d(embedding_dim)

        self.conv4 = GCNConv(embedding_dim, embedding_dim)
        self.bn4 = nn.BatchNorm1d(embedding_dim)

        # Shared projection after pooling (mean + max = 2 * embedding_dim)
        self.shared_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 5 separate prediction heads
        self.heads = nn.ModuleDict({
            param: PKPredictionHead(embedding_dim)
            for param in self.PK_PARAMS
        })

        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        """
        Forward pass: graph -> PK predictions.

        Args:
            data: PyTorch Geometric Batch with x, edge_index, batch

        Returns:
            Dict mapping PK param name -> predicted values
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2 (with residual connection)
        residual = x
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = x + residual  # Residual connection
        x = self.dropout(x)

        # Layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 4 (with residual)
        residual = x
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x)
        x = x + residual
        x = self.dropout(x)

        # Global pooling: combine all atom features into one molecule vector
        # Mean captures average properties, Max captures extreme features
        x_mean = global_mean_pool(x, batch)  # [batch_size, embedding_dim]
        x_max = global_max_pool(x, batch)    # [batch_size, embedding_dim]
        x = torch.cat([x_mean, x_max], dim=1)  # [batch_size, embedding_dim * 2]

        # Shared projection
        x = self.shared_mlp(x)  # [batch_size, embedding_dim]

        # Each head predicts its PK parameter
        predictions = {
            param: head(x) for param, head in self.heads.items()
        }

        return predictions

    def predict_single(self, smiles: str, device: str = "cpu") -> dict:
        """
        Predict PK parameters for a single SMILES string.
        Convenience method for inference.
        """
        self.eval()
        graph = smiles_to_graph(smiles)
        if graph is None:
            return None

        batch = Batch.from_data_list([graph]).to(device)
        with torch.no_grad():
            preds = self(batch)

        # Convert log predictions back to real values
        return {
            param: 10 ** pred.item()
            for param, pred in preds.items()
        }


# ══════════════════════════════════════════════════════════════
# PHYSICS-INFORMED LOSS
# ══════════════════════════════════════════════════════════════

class PhysicsInformedLoss(nn.Module):
    """
    Custom loss function that enforces pharmacokinetic relationships.

    Standard MSE loss says: "make each prediction close to the true value."
    Physics loss ALSO says: "your predictions must obey PK equations."

    Key PK relationships:
    1. CL = Dose / AUC           (clearance = dose divided by total exposure)
    2. Vd = CL * t½ / ln(2)     (volume of distribution relates to CL and half-life)
    3. Mass balance: AUC * CL should be consistent

    These constraints act as a regularizer — even when individual labels are noisy,
    the relationships between parameters provide additional learning signal.

    THIS IS THE KEY DIFFERENTIATOR of NeoGen.
    """

    def __init__(self, mse_weight: float = 1.0, physics_weight: float = 0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.physics_weight = physics_weight
        self.mse = nn.MSELoss()

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        mask: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute combined MSE + physics loss.

        Args:
            predictions: Dict of PK param -> predicted log10 values
            targets: Dict of PK param -> true log10 values
            mask: Dict of PK param -> boolean mask (True if label available)

        Returns:
            total_loss, loss_details dict
        """
        losses = {}
        total_mse = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        n_terms = 0

        # Standard MSE loss for each parameter
        for param in predictions:
            if param in targets and mask[param].any():
                pred = predictions[param][mask[param]]
                true = targets[param][mask[param]]
                loss = self.mse(pred, true)
                losses[f"mse_{param}"] = loss.item()
                total_mse = total_mse + loss
                n_terms += 1

        if n_terms > 0:
            total_mse = total_mse / n_terms

        # Physics constraint: Vd ≈ CL * t½ / ln(2)
        # In log space: log(Vd) ≈ log(CL) + log(t½) - log(ln(2))
        physics_loss = torch.tensor(0.0, device=total_mse.device)
        n_physics = 0

        if all(p in predictions for p in ["vd", "clearance", "half_life"]):
            # Find molecules that have all 3 predictions
            batch_size = predictions["vd"].shape[0]
            if batch_size > 0:
                log_vd_pred = predictions["vd"]
                log_cl_pred = predictions["clearance"]
                log_t12_pred = predictions["half_life"]

                # Physics: log(Vd) should ≈ log(CL) + log(t½) - log(ln2)
                log_ln2 = np.log10(np.log(2))  # ≈ -0.164
                vd_from_physics = log_cl_pred + log_t12_pred - log_ln2

                constraint = self.mse(log_vd_pred, vd_from_physics)
                physics_loss = physics_loss + constraint
                losses["physics_vd_consistency"] = constraint.item()
                n_physics += 1

        if n_physics > 0:
            physics_loss = physics_loss / n_physics

        total = self.mse_weight * total_mse + self.physics_weight * physics_loss
        losses["total"] = total.item()
        losses["mse_total"] = total_mse.item()
        losses["physics_total"] = physics_loss.item()

        return total, losses


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("NeoGen GNN — Architecture Test")
    print("=" * 60)

    # Test SMILES -> Graph
    test_smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"]
    test_names = ["Aspirin", "Caffeine", "Ibuprofen"]

    graphs = []
    for smiles, name in zip(test_smiles, test_names):
        g = smiles_to_graph(smiles)
        print(f"\n{name} ({smiles[:30]}...):")
        print(f"  Atoms: {g.x.shape[0]}, Features per atom: {g.x.shape[1]}")
        print(f"  Bonds: {g.edge_index.shape[1] // 2}")
        graphs.append(g)

    # Test model
    print(f"\nAtom feature dimension: {N_ATOM_FEATURES}")

    model = NeoGenGNN(n_atom_features=N_ATOM_FEATURES)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Forward pass
    batch = Batch.from_data_list(graphs)
    predictions = model(batch)

    print(f"\nPredictions for batch of {len(graphs)} molecules:")
    for param, values in predictions.items():
        real_values = [10 ** v.item() for v in values]
        print(f"  {param}: {[f'{v:.1f}' for v in real_values]}")

    # Test single prediction
    print("\n--- Single Molecule Prediction ---")
    result = model.predict_single("CC(=O)OC1=CC=CC=C1C(=O)O")
    print("Aspirin predictions (untrained model):")
    for param, value in result.items():
        print(f"  {param}: {value:.2f}")

    print("\nGNN architecture test passed!")
