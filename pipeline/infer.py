from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from pipeline.AttentionalGINEModel import AttentionalGINEModel
from pipeline.Config import Config, PROJECT_ROOT
from pipeline.graph_encoding import smiles_to_graph


def _find_latest_saved_model(runs_root: Optional[Path] = None) -> Optional[Path]:
    """Utility to locate the most recently saved PyTorch model."""
    runs_root = runs_root or (PROJECT_ROOT / "plots" / "runs")
    if not runs_root.exists():
        return None
    candidates = []
    for child in runs_root.iterdir():
        if child.is_dir():
            candidate = child / "GNN" / "best_gine_model.pt"
            if candidate.exists():
                candidates.append((child.stat().st_mtime, candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def load_model(in_channels: int = 34) -> nn.Module:
    """Load model weights either from production directory or recent runtime."""
    model = AttentionalGINEModel(in_channels=in_channels, cfg=Config)

    prod_weights = Config.PRODUCTION_DIR / "best_gine_model.pt"
    if prod_weights.exists():
        model.load_state_dict(torch.load(prod_weights, map_location=Config.device))
        model = model.to(Config.device)
        model.eval()
        return model

    latest_runtime = _find_latest_saved_model()
    if latest_runtime and latest_runtime.exists():
        model.load_state_dict(torch.load(latest_runtime, map_location=Config.device))
        model = model.to(Config.device)
        model.eval()
        return model

    print("Warning: Model weight loading checkpoint bypassed. Uninitialized states present.")
    return model


def _load_y_scaler() -> Optional[StandardScaler]:
    """Attempt to load target Y-scaler."""
    prod_scaler = Config.PRODUCTION_DIR / "fitted_scaler.pkl"
    if prod_scaler.exists():
        return joblib.load(prod_scaler)

    latest_runtime = _find_latest_saved_model()
    if latest_runtime is not None:
        runtime_scaler = latest_runtime.parent / "fitted_scaler.pkl"
        if runtime_scaler.exists():
            return joblib.load(runtime_scaler)

    print("Warning: Target StandardScaler lookup failed. Outputs will remain unscaled.")
    return None


def _load_global_scaler() -> Optional[StandardScaler]:
    """Attempt to load global descriptor scaler."""
    prod_scaler = Config.PRODUCTION_DIR / "global_descriptor_scaler.pkl"
    if prod_scaler.exists():
        return joblib.load(prod_scaler)

    latest_runtime = _find_latest_saved_model()
    if latest_runtime is not None:
        runtime_scaler = latest_runtime.parent / "global_descriptor_scaler.pkl"
        if runtime_scaler.exists():
            return joblib.load(runtime_scaler)
    return None


def predict_pic50_from_smiles(smiles: str, model: Optional[nn.Module] = None, scaler: Optional[StandardScaler] = None) -> float:
    """Predict pIC50 value for a single SMILES string using trained model components."""
    if not smiles or not isinstance(smiles, str):
        raise ValueError("smiles must be a non-empty string")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES: cannot parse")

    try: heavy_atoms = float(mol.GetNumHeavyAtoms())
    except Exception: heavy_atoms = 0.0

    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        lip_viol = float(int((mw > 500)) + int((logp > 5)) + int((hbd > 5)) + int((hba > 10)))
    except Exception:
        lip_viol = 0.0

    scaler_g = _load_global_scaler()
    if scaler_g is not None:
        scaled_features = scaler_g.transform([[heavy_atoms, lip_viol]])[0]
        g_desc = [float(scaled_features[0]), float(scaled_features[1])]
    else:
        print("Warning: global descriptor scaler missing. Using unscaled macro features.")
        g_desc = [heavy_atoms, lip_viol]

    data = smiles_to_graph(smiles, 0.0, g_desc)
    if data is None:
        raise RuntimeError("Failed to build graph from SMILES")

    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)

    if model is None: model = load_model(in_channels=data.x.size(1))
    scaler_y = scaler if scaler is not None else _load_y_scaler()

    model.eval()
    with torch.no_grad():
        out = model(data.to(next(model.parameters()).device))
        pred = float(out.detach().cpu().item())

    if scaler_y is not None:
        try:
            return float(scaler_y.inverse_transform(np.array([[pred]])).flatten()[0])
        except Exception:
            pass

    return float(pred)


def compute_atom_explainability(model: torch.nn.Module, data: Data) -> np.ndarray:
    """Compute attribution scores indicating which atoms contributed most to the prediction."""
    model.eval()
    x_input = data.x.clone().detach().requires_grad_(True)

    data_tracked = Data(
        x=x_input,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        batch=torch.zeros(x_input.size(0), dtype=torch.long, device=x_input.device),
        g_desc=data.g_desc
    ).to(next(model.parameters()).device)

    prediction = model(data_tracked)
    prediction.backward()

    atom_gradients = x_input.grad.cpu().numpy()
    atom_importance = np.linalg.norm(atom_gradients, axis=1)

    denom = (atom_importance.max() - atom_importance.min())
    if denom > 0:
        atom_importance = (atom_importance - atom_importance.min()) / denom

    return atom_importance