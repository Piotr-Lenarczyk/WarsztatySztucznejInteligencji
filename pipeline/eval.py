import random
from typing import List, Optional, Dict

import random
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from pipeline.Config import Config


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: str, y_scaler: Optional[StandardScaler] = None) -> Dict[str, float]:
    """Calculate standard regression metrics (MSE, RMSE, MAE, R2) for a given DataLoader."""
    model.eval()
    preds, trues = [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        preds.append(out.detach().cpu().numpy())
        trues.append(batch.y.view(-1).detach().cpu().numpy())

    p = np.concatenate(preds).reshape(-1, 1)
    y = np.concatenate(trues).reshape(-1, 1)

    if y_scaler is not None:
        p = y_scaler.inverse_transform(p)
        y = y_scaler.inverse_transform(y)

    p, y = p.reshape(-1), y.reshape(-1)
    mse = float(np.mean((p - y) ** 2))
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum((p - y) ** 2) / denom) if denom > 0 else float('nan')

    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(p - y))), "r2": r2}


def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Filter to standard valid pChEMBL range."""
    df = df_raw.copy()
    return df[(df['pchembl_value'] >= 3.0) & (df['pchembl_value'] <= 12.0)].copy()


def aggregate_compounds(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate measurements per unique SMILES into a single row."""
    return df.groupby('smiles').agg({
        'pchembl_value': 'median',
        'heavy_atoms': 'first',
        'lipinski_violations': 'first'
    }).reset_index()


def scale_global_descriptors(df: pd.DataFrame):
    """Fit scaler for global descriptors."""
    scaler_g = StandardScaler()
    cols = ['heavy_atoms', 'lipinski_violations']
    df[cols] = scaler_g.fit_transform(df[cols])
    return scaler_g, df


def scaffold_split_indices(dataset: List[Data], df_reference: pd.DataFrame):
    """Perform a Murcko Scaffold split (80/10/10) to separate structurally similar compounds."""
    scaffs = {}
    for i, g in enumerate(dataset):
        try:
            s = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(df_reference.iloc[i]['smiles']))
        except Exception:
            s = None
        scaffs.setdefault(s, []).append(i)

    keys = list(scaffs.keys())
    random.shuffle(keys)
    train_idx, val_idx, test_idx = [], [], []
    for k in keys:
        ids = scaffs[k]
        if len(train_idx) < 0.8 * len(dataset):
            train_idx.extend(ids)
        elif len(val_idx) < 0.1 * len(dataset):
            val_idx.extend(ids)
        else:
            test_idx.extend(ids)
    return train_idx, val_idx, test_idx


def fit_y_scaler(train_idx: List[int], dataset: List[Data]):
    """Standardize the target variables."""
    scaler_y = StandardScaler()
    train_y = np.array([dataset[i].y.item() for i in train_idx]).reshape(-1, 1)
    scaler_y.fit(train_y)
    return scaler_y


def make_dataloader_from_indices(idxs: List[int], dataset: List[Data], scaler_y: StandardScaler, shuffle: bool = False):
    """Create a DataLoader corresponding to specific indices (train/val/test)."""
    data_list = []
    for i in idxs:
        g = dataset[i].clone()
        g.y = torch.tensor(scaler_y.transform([[g.y.item()]])[0], dtype=torch.float)
        data_list.append(g)
    return DataLoader(data_list, batch_size=Config.batch_size, shuffle=shuffle)