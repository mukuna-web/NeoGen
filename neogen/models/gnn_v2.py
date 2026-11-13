"""
NeoGen GNN v2 — Improved Graph Neural Network
================================================
Key improvements over v1:
1. GIN (Graph Isomorphism Network) instead of GCN — proven better for molecules
2. Edge features (bond type, aromaticity) via message passing
3. Global molecular descriptors concatenated to graph embedding
4. Attention-based graph readout (learns WHICH atoms matter most)
5. Per-parameter loss weighting (balances dataset size differences)
6. Deeper prediction heads with skip connections

Architecture:
    SMILES -> Graph + Descriptors
              |
              v
    [GIN_1 -> GIN_2 -> GIN_3 -> GIN_4 -> GIN_5]  (5 layers with JK connections)
              |
              v
    Attention Pool -> 512-dim graph embedding
              |
              v
    Concat with 10 molecular descriptors -> 522-dim
              |
              v
    Shared MLP -> 256-dim
              |
              +--> Head_cmax -> 1
              +--> Head_auc -> 1
              +--> Head_half_life -> 1
              +--> Head_clearance -> 1
              +--> Head_vd -> 1
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from torch_geometric.data import Data, Batch
from torch_geometric.nn import (
    GINConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)

logger = logging.getLogger(__name__)

# ============================================================
# SMILES -> Graph Conversion (Enhanced)
# ============================================================

ATOM_FEATURES = {
    "element": [6, 7, 8, 9, 15, 16, 17, 35, 53],
    "degree": [0, 1, 2, 3, 4, 5],
    "charge": [-2, -1, 0, 1, 2],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    "num_hs": [0, 1, 2, 3, 4],
}

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def one_hot(value, options):
    vec = [0] * (len(options) + 1)
    if value in options:
        vec[options.index(value)] = 1
    else:
        vec[-1] = 1
    return vec


def atom_features_v2(atom):
    """
    Enhanced atom features (39 dimensions):
    - Element type (10): one-hot + other
    - Degree (7): one-hot + other
    - Formal charge (6): one-hot + other
    - Hybridization (6): one-hot + other
    - Num Hs (6): one-hot + other
    - Is aromatic (1): binary
    - Is in ring (1): binary
    - Ring size (1): normalized
    - Mass (1): normalized
    """
    features = []
    features += one_hot(atom.GetAtomicNum(), ATOM_FEATURES["element"])
    features += one_hot(atom.GetTotalDegree(), ATOM_FEATURES["degree"])
    features += one_hot(atom.GetFormalCharge(), ATOM_FEATURES["charge"])
    features += one_hot(atom.GetHybridization(), ATOM_FEATURES["hybridization"])
    features += one_hot(atom.GetTotalNumHs(), ATOM_FEATURES["num_hs"])
    features.append(1.0 if atom.GetIsAromatic() else 0.0)
    features.append(1.0 if atom.IsInRing() else 0.0)

    # Ring size (smallest ring the atom is in)
    ri = atom.GetOwningMol().GetRingInfo()
    ring_sizes = [len(r) for r in ri.AtomRings() if atom.GetIdx() in r]
    features.append(min(ring_sizes) / 10.0 if ring_sizes else 0.0)

    # Normalized atomic mass
    features.append(atom.GetMass() / 100.0)

    return features


def bond_features(bond):
    """Bond features (6 dimensions)."""
    feats = one_hot(bond.GetBondType(), BOND_TYPES)  # 5 dim
    feats.append(1.0 if bond.IsInRing() else 0.0)
    return feats


def compute_descriptors(smiles: str):
    """
    Compute 10 global molecular descriptors from RDKit.
    These capture whole-molecule properties that atoms alone can't express.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0.0] * 10

    try:
        desc = [
            Descriptors.MolLogP(mol) / 5.0,                    # Lipophilicity
            Descriptors.MolWt(mol) / 500.0,                    # Molecular weight
            Descriptors.TPSA(mol) / 150.0,                     # Polar surface area
            rdMolDescriptors.CalcNumRotatableBonds(mol) / 15.0, # Flexibility
            rdMolDescriptors.CalcNumHBA(mol) / 10.0,           # H-bond acceptors
            rdMolDescriptors.CalcNumHBD(mol) / 5.0,            # H-bond donors
            Descriptors.NumAromaticRings(mol) / 5.0,           # Aromaticity
            rdMolDescriptors.CalcFractionCSP3(mol),            # Fraction sp3
            Descriptors.NumHeteroatoms(mol) / 15.0,            # Heteroatom count
            Descriptors.RingCount(mol) / 6.0,                  # Ring count
        ]
    except Exception:
        desc = [0.0] * 10

    return desc


def smiles_to_graph_v2(smiles: str) -> Data | None:
    """
    Enhanced SMILES -> Graph conversion.
    Now includes bond features and molecular descriptors.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    atom_feats = [atom_features_v2(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float)

    # Edge index and features
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_attrs.append(bf)
        edge_attrs.append(bf)

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 6), dtype=torch.float)

    # Global descriptors
    desc = compute_descriptors(smiles)
    descriptors = torch.tensor(desc, dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        descriptors=descriptors,
    )


# Compute feature dimensions from a sample molecule
_sample_mol = Chem.MolFromSmiles("C")
N_ATOM_FEATURES_V2 = len(atom_features_v2(_sample_mol.GetAtomWithIdx(0)))
N_BOND_FEATURES = 6
N_DESCRIPTORS = 10


# ============================================================
# GNN v2 MODEL
# ============================================================

class GINBlock(nn.Module):
    """
    A single GIN (Graph Isomorphism Network) layer.

    GIN is provably more powerful than GCN for distinguishing
    different graph structures. It uses a learnable MLP instead
    of simple averaging for message passing.
    """

    def __init__(self, in_dim, out_dim, edge_dim=0):
        super().__init__()
        # GIN's MLP applied after aggregation
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.conv = GINConv(mlp, train_eps=True)
        self.bn = nn.BatchNorm1d(out_dim)

        # Edge feature projection (optional)
        self.edge_proj = None
        if edge_dim > 0:
            self.edge_proj = nn.Linear(edge_dim, in_dim)

    def forward(self, x, edge_index, edge_attr=None):
        # If we have edge features, add them to source node features
        if self.edge_proj is not None and edge_attr is not None:
            edge_emb = self.edge_proj(edge_attr)
            # Add edge features to source nodes (simple but effective)
            row = edge_index[0]
            x_aug = x.clone()
            x_aug.index_add_(0, row, edge_emb)
            h = self.conv(x_aug, edge_index)
        else:
            h = self.conv(x, edge_index)

        h = self.bn(h)
        h = F.relu(h)
        return h


class AttentionReadout(nn.Module):
    """
    Attention-based graph readout.

    Instead of naive mean/max pooling (treating all atoms equally),
    this learns which atoms are most important for the prediction.
    A pharmacophore-like attention — atoms near reactive sites get
    higher weights than carbons in the middle of a chain.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, batch):
        # Compute attention weights per atom
        weights = self.gate(x)  # [n_atoms, 1]
        # Softmax within each graph (molecule)
        weights = self._softmax_per_graph(weights, batch)
        # Weighted sum
        weighted = x * weights
        return global_add_pool(weighted, batch)

    def _softmax_per_graph(self, weights, batch):
        """Softmax within each graph in the batch."""
        # Subtract max for numerical stability
        max_vals = global_max_pool(weights, batch)
        weights = weights - max_vals[batch]
        exp_weights = torch.exp(weights)
        sum_exp = global_add_pool(exp_weights, batch)
        return exp_weights / (sum_exp[batch] + 1e-8)


class PKHeadV2(nn.Module):
    """
    Improved prediction head with skip connection.
    256 -> 128 -> 64 -> 1, with a shortcut from 256 -> 64.
    """

    def __init__(self, input_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)
        self.skip = nn.Linear(input_dim, 64)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        skip = self.skip(x)
        h = F.relu(self.bn1(self.fc1(x)))
        h = self.dropout(h)
        h = F.relu(self.bn2(self.fc2(h)) + skip)
        h = self.dropout(h)
        return self.fc3(h).squeeze(-1)


class NeoGenGNNv2(nn.Module):
    """
    NeoGen GNN v2 — significantly improved architecture.

    Key differences from v1:
    1. GIN layers (more expressive than GCN)
    2. 5 layers with Jumping Knowledge (concatenate all layer outputs)
    3. Edge features (bond type) incorporated
    4. Attention readout (learns which atoms matter)
    5. Molecular descriptors concatenated to embedding
    6. Per-parameter heads with skip connections
    """

    PK_PARAMS = ["cmax", "auc", "half_life", "clearance", "vd"]

    def __init__(
        self,
        n_atom_features: int = N_ATOM_FEATURES_V2,
        n_bond_features: int = N_BOND_FEATURES,
        n_descriptors: int = N_DESCRIPTORS,
        hidden_dim: int = 256,
        n_layers: int = 5,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.n_layers = n_layers

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_atom_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # GIN layers
        self.gin_layers = nn.ModuleList()
        for i in range(n_layers):
            edge_dim = n_bond_features if i == 0 else 0  # Only first layer uses edge features
            self.gin_layers.append(GINBlock(hidden_dim, hidden_dim, edge_dim))

        # Jumping Knowledge: combine all layer representations
        self.jk_proj = nn.Linear(hidden_dim * n_layers, hidden_dim * 2)

        # Attention readout
        self.attention = AttentionReadout(hidden_dim * 2)

        # Also do mean + max pooling (combine with attention)
        # Final graph embedding = attention(512) + mean(512) + max(512) = 1536 -> project to 512
        self.pool_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 3, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Descriptor integration
        self.desc_proj = nn.Sequential(
            nn.Linear(n_descriptors, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Shared projection: 512 + 64 = 576 -> 256
        shared_input_dim = hidden_dim * 2 + 64
        self.shared_mlp = nn.Sequential(
            nn.Linear(shared_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Per-parameter prediction heads
        self.heads = nn.ModuleDict({
            param: PKHeadV2(256) for param in self.PK_PARAMS
        })

        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        batch = data.batch

        # Input projection
        x = self.input_proj(x)

        # GIN layers with Jumping Knowledge
        layer_outputs = []
        for i, gin in enumerate(self.gin_layers):
            ea = edge_attr if i == 0 else None
            x_new = gin(x, edge_index, ea)
            x = x + x_new  # Residual
            x = self.dropout(x)
            layer_outputs.append(x)

        # Jumping Knowledge: concatenate all layer outputs
        jk = torch.cat(layer_outputs, dim=-1)
        jk = self.jk_proj(jk)

        # Triple pooling: attention + mean + max
        x_att = self.attention(jk, batch)
        x_mean = global_mean_pool(jk, batch)
        x_max = global_max_pool(jk, batch)
        pooled = torch.cat([x_att, x_mean, x_max], dim=-1)
        pooled = self.pool_proj(pooled)

        # Add molecular descriptors
        # PyG batching flattens descriptors — reshape back to [batch_size, n_desc]
        desc = data.descriptors  # flattened: [batch_size * n_descriptors]
        n_graphs = batch.max().item() + 1
        desc = desc.view(n_graphs, -1)  # [batch_size, n_descriptors]
        desc_emb = self.desc_proj(desc)

        # Combine
        combined = torch.cat([pooled, desc_emb], dim=-1)
        shared = self.shared_mlp(combined)

        # Per-parameter predictions
        predictions = {
            param: head(shared) for param, head in self.heads.items()
        }

        return predictions

    def predict_single(self, smiles: str, device: str = "cpu") -> dict | None:
        """Predict PK parameters for a single SMILES string."""
        self.eval()
        graph = smiles_to_graph_v2(smiles)
        if graph is None:
            return None

        batch = Batch.from_data_list([graph]).to(device)
        with torch.no_grad():
            preds = self(batch)

        return {
            param: 10 ** pred.item()
            for param, pred in preds.items()
        }


# ============================================================
# IMPROVED LOSS FUNCTION
# ============================================================

class ImprovedPhysicsLoss(nn.Module):
    """
    Enhanced loss with:
    1. Per-parameter weighting (inverse of dataset size)
    2. Huber loss instead of MSE (robust to outliers)
    3. Stronger physics constraints
    4. Consistency regularization
    """

    def __init__(
        self,
        param_weights: dict | None = None,
        physics_weight: float = 0.15,
    ):
        super().__init__()
        self.physics_weight = physics_weight
        self.huber = nn.HuberLoss(delta=1.0)

        # Default weights inversely proportional to dataset size
        # This balances the loss so small datasets (Vd: 7K) aren't overwhelmed
        # by large ones (half_life: 39K)
        if param_weights is None:
            param_weights = {
                "cmax": 2.0,       # 8.7K samples — upweight
                "auc": 1.2,        # 13.5K samples
                "half_life": 0.6,  # 38.6K samples — downweight
                "clearance": 1.0,  # 18.2K samples
                "vd": 2.5,         # 7K samples — most upweight
            }
        self.param_weights = param_weights

    def forward(self, predictions, targets, masks):
        losses = {}
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        n_terms = 0

        # Weighted Huber loss per parameter
        for param in predictions:
            if param in targets and masks[param].any():
                pred = predictions[param][masks[param]]
                true = targets[param][masks[param]]
                loss = self.huber(pred, true)
                weight = self.param_weights.get(param, 1.0)
                weighted_loss = loss * weight
                losses[f"mse_{param}"] = loss.item()
                total_loss = total_loss + weighted_loss
                n_terms += 1

        if n_terms > 0:
            total_loss = total_loss / n_terms

        # Physics: Vd = CL * t½ / ln(2) => log(Vd) = log(CL) + log(t½) - log(ln2)
        physics_loss = torch.tensor(0.0, device=total_loss.device)
        n_physics = 0

        if all(p in predictions for p in ["vd", "clearance", "half_life"]):
            # Find samples where all 3 params have labels
            joint_mask = masks["vd"] & masks["clearance"] & masks["half_life"]
            if joint_mask.any():
                log_vd = predictions["vd"][joint_mask]
                log_cl = predictions["clearance"][joint_mask]
                log_t12 = predictions["half_life"][joint_mask]
                log_ln2 = np.log10(np.log(2))
                vd_physics = log_cl + log_t12 - log_ln2
                constraint = self.huber(log_vd, vd_physics)
                physics_loss = physics_loss + constraint
                losses["physics_vd"] = constraint.item()
                n_physics += 1

        # Physics: CL = Dose / AUC (in log space, relative constraint)
        if all(p in predictions for p in ["clearance", "auc"]):
            joint_mask = masks["clearance"] & masks["auc"]
            if joint_mask.any():
                log_cl = predictions["clearance"][joint_mask]
                log_auc = predictions["auc"][joint_mask]
                # CL and AUC should be inversely related
                # Penalize if both go up or both go down together
                corr = torch.mean(log_cl * log_auc) - torch.mean(log_cl) * torch.mean(log_auc)
                # We want negative correlation (higher AUC = lower CL)
                if corr > 0:
                    physics_loss = physics_loss + corr * 0.5
                    losses["physics_cl_auc"] = corr.item()
                    n_physics += 1

        if n_physics > 0:
            physics_loss = physics_loss / n_physics

        total = total_loss + self.physics_weight * physics_loss
        losses["total"] = total.item()
        losses["mse_total"] = total_loss.item()
        losses["physics_total"] = physics_loss.item()

        return total, losses


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("NeoGen GNN v2 — Architecture Test")
    print("=" * 60)

    test_smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    ]
    test_names = ["Aspirin", "Caffeine", "Ibuprofen"]

    graphs = []
    for smiles, name in zip(test_smiles, test_names):
        g = smiles_to_graph_v2(smiles)
        print(f"\n{name}:")
        print(f"  Atoms: {g.x.shape[0]}, Features: {g.x.shape[1]}")
        print(f"  Bonds: {g.edge_index.shape[1] // 2}, Edge features: {g.edge_attr.shape[1]}")
        print(f"  Descriptors: {g.descriptors.shape}")
        graphs.append(g)

    print(f"\nAtom features: {N_ATOM_FEATURES_V2}")
    print(f"Bond features: {N_BOND_FEATURES}")
    print(f"Descriptors: {N_DESCRIPTORS}")

    model = NeoGenGNNv2()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    batch = Batch.from_data_list(graphs)
    predictions = model(batch)

    print(f"\nPredictions for batch of {len(graphs)} molecules:")
    for param, values in predictions.items():
        real_values = [10 ** v.item() for v in values]
        print(f"  {param}: {[f'{v:.2f}' for v in real_values]}")

    # Test single prediction
    result = model.predict_single("CC(=O)OC1=CC=CC=C1C(=O)O")
    print("\nAspirin single prediction (untrained):")
    for param, value in result.items():
        print(f"  {param}: {value:.2f}")

    print("\nGNN v2 architecture test PASSED!")
