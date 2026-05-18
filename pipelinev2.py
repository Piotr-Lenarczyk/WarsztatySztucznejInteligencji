import os
import json
import time
import torch
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool, JumpingKnowledge

# ==========================================
# CONFIGURATION & DIRECTORIES
# ==========================================
class Config:
    seed = 42
    target_id = "CHEMBL2147"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model Hyperparameters
    hidden_dim = 128
    num_layers = 4
    dropout = 0.3
    edge_dim = 4

    # Training Hyperparameters
    batch_size = 128
    epochs = 50
    lr = 5e-4
    weight_decay = 1e-3
    early_stopping_patience = 4

    # Paths - Use absolute paths to be safe
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Using .resolve() ensures the path is absolute and correctly formatted for your OS
    plots_dir = Path("plots").resolve() / "runs" / timestamp

def setup_dirs():
    # Ensure every parent in the chain is created
    Config.plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"Directory created at: {Config.plots_dir}")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==========================================
# EDA FUNCTIONS
# ==========================================
def run_full_eda(df: pd.DataFrame):
    setup_dirs() # Ensure dir exists before saving
    print("--- Starting Full EDA ---")

    # 1. Correlation Analysis
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr().abs()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, square=True)
        plt.title('Correlation Heatmap')
        # Explicitly cast Path to string for savefig
        plt.savefig(str(Config.plots_dir / "eda_correlation.png"))
        plt.close()

    # 2. Null Value Check
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        plt.figure(figsize=(12, 6))
        null_counts[null_counts > 0].sort_values(ascending=False).plot(kind='bar')
        plt.title('Null Value Counts')
        plt.savefig(str(Config.plots_dir / "eda_nulls.png"))
        plt.close()

    # 3. Distribution of pIC50 and IC50
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(df["standard_value"], bins=50, ax=axes[0], kde=True, log_scale=True)
    axes[0].set_title("IC50 Distribution (nM)")
    sns.histplot(df["pchembl_value"], bins=50, ax=axes[1], kde=True)
    axes[1].set_title("pIC50 Distribution")
    plt.savefig(str(Config.plots_dir / "eda_distributions.png"))
    plt.close()

    # 4. Lipinski Analysis
    def calc_lipinski(smi):
        mol = Chem.MolFromSmiles(smi)
        if not mol: return [None]*4
        return [Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)]

    lip_cols = ["MW", "LogP", "HBD", "HBA"]
    df[lip_cols] = pd.DataFrame(df["smiles"].apply(calc_lipinski).tolist(), index=df.index)

    violations = (df["MW"] > 500).astype(int) + (df["LogP"] > 5).astype(int) + \
                 (df["HBD"] > 5).astype(int) + (df["HBA"] > 10).astype(int)
    df["lipinski_violations"] = violations

    plt.figure(figsize=(8, 5))
    sns.countplot(x="lipinski_violations", data=df, palette="viridis")
    plt.title("Lipinski's Rule Violations")
    plt.savefig(str(Config.plots_dir / "eda_lipinski_violations.png"))
    plt.close()

    # 5. Scaffold Analysis
    def get_scaffold(smi):
        mol = Chem.MolFromSmiles(smi)
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else None

    df["scaffold"] = df["smiles"].apply(get_scaffold)
    scaff_counts = df["scaffold"].value_counts().head(10)
    print(f"Unique Scaffolds Found: {df['scaffold'].nunique()}")

    # Save a visual grid of top scaffolds
    mols = [Chem.MolFromSmiles(s) for s in scaff_counts.index]
    img = Draw.MolsToGridImage(mols, molsPerRow=5, legends=[f"Count: {c}" for c in scaff_counts.values])
    img.save(Config.plots_dir / "eda_top_scaffolds.png")

# ==========================================
# FEATURIZATION & MODEL
# ==========================================
def atom_features(atom):
    return [
        *([1.0 if i == atom.GetAtomicNum() else 0.0 for i in [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53]] + [0.0]),
        *([1.0 if i == atom.GetDegree() else 0.0 for i in range(7)] + [0.0]),
        1.0 if atom.GetIsAromatic() else 0.0,
        float(atom.GetTotalNumHs())
    ]

def bond_features(bond):
    bt = bond.GetBondType()
    return [float(bt == t) for t in [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                                     Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]]

def smiles_to_graph(smiles, y_val):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edges, attrs = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_features(b)
        edges.extend([[i, j], [j, i]]); attrs.extend([f, f])
    return Data(x=x, edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(attrs, dtype=torch.float), y=torch.tensor([y_val], dtype=torch.float))

class GINEModel(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(cfg.num_layers):
            mlp = nn.Sequential(nn.Linear(in_channels if i==0 else cfg.hidden_dim, cfg.hidden_dim),
                                nn.ReLU(), nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
            self.convs.append(GINEConv(mlp, edge_dim=cfg.edge_dim))
        self.jk = JumpingKnowledge(mode='cat')
        self.head = nn.Sequential(nn.Linear(cfg.hidden_dim*cfg.num_layers, 256), nn.ReLU(),
                                  nn.Dropout(cfg.dropout), nn.Linear(256, 1))

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        xs = []
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr))
            xs.append(x)
        x = global_add_pool(self.jk(xs), batch)
        return self.head(x).view(-1)

# New: plotting/metrics dirs and helper
RUN_TIMESTAMP = Config.timestamp
PLOTS_DIR = Config.plots_dir
PLOTS_DIR_MLP = PLOTS_DIR / "MLP"
PLOTS_DIR_GNN = PLOTS_DIR / "GNN"
PLOTS_DIR_MLP.mkdir(parents=True, exist_ok=True)
PLOTS_DIR_GNN.mkdir(parents=True, exist_ok=True)

def save_and_close_plot(fig, filepath: Path):
    fig.savefig(str(filepath), bbox_inches='tight')
    plt.close(fig)

# Copy of evaluate_gnn from pipeline.py adapted to local names
@torch.no_grad()
def evaluate_gnn_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    y_scaler: Optional[StandardScaler] = None,
) -> Dict[str, float]:
    model.eval()
    preds = []
    trues = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        y = batch.y.view(-1)
        preds.append(out.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())

    if len(preds) == 0:
        return {"mse": float('nan'), "rmse": float('nan'), "mae": float('nan'), "r2": float('nan'), "accuracy": float('nan')}

    p = np.concatenate(preds).reshape(-1, 1)
    y = np.concatenate(trues).reshape(-1, 1)

    if y_scaler is not None:
        try:
            p = y_scaler.inverse_transform(p)
            y = y_scaler.inverse_transform(y)
        except Exception:
            pass

    p = p.reshape(-1)
    y = y.reshape(-1)

    mse = float(np.mean((p - y) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(p - y)))
    acc = float(np.mean(np.abs(p - y) <= 1000.0))
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum((p - y) ** 2) / denom) if denom > 0 else float('nan')
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "accuracy": acc}

# ==========================================
# PIPELINE EXECUTION
# ==========================================
def run_training_pipeline(df_raw: pd.DataFrame):
    setup_dirs()
    seed_everything(Config.seed)

    # 1. PRE-CLEANING DATA (Median Aggregation)
    df = df_raw.groupby('smiles')['pchembl_value'].median().reset_index()
    print(f"Dataset unique compounds: {len(df)}")

    # 2. FEATURIZATION
    dataset = []
    for _, row in df.iterrows():
        g = smiles_to_graph(row['smiles'], row['pchembl_value'])
        if g: dataset.append(g)

    # 3. SCAFFOLD SPLIT
    scaffs = {}
    for i, g in enumerate(dataset):
        s = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(df.iloc[i]['smiles']))
        scaffs.setdefault(s, []).append(i)

    keys = list(scaffs.keys()); random.shuffle(keys)
    train_idx, val_idx, test_idx = [], [], []
    for k in keys:
        ids = scaffs[k]
        if len(train_idx) < 0.8*len(dataset): train_idx.extend(ids)
        elif len(val_idx) < 0.1*len(dataset): val_idx.extend(ids)
        else: test_idx.extend(ids)

    # 4. SCALING
    scaler = StandardScaler()
    train_y = np.array([dataset[i].y.item() for i in train_idx]).reshape(-1, 1)
    scaler.fit(train_y)

    def get_loader(idxs, shuffle=False):
        data_list = []
        for i in idxs:
            g = dataset[i].clone()
            g.y = torch.tensor(scaler.transform([[g.y.item()]])[0], dtype=torch.float)
            data_list.append(g)
        return DataLoader(data_list, batch_size=Config.batch_size, shuffle=shuffle)

    train_loader = get_loader(train_idx, True)
    val_loader = get_loader(val_idx)
    test_loader = get_loader(test_idx)

    # 5. TRAINING
    model = GINEModel(dataset[0].x.size(1), Config).to(Config.device)
    opt = torch.optim.Adam(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    crit = nn.MSELoss()

    best_r2 = -float('inf')
    patience = 0

    # NEW: history collection (re-added to save plots like previous runs)
    history = {
        "epoch": [],
        "train_mse": [], "val_mse": [], "test_mse": [],
        "train_rmse": [], "val_rmse": [], "test_rmse": [],
        "train_mae": [], "val_mae": [], "test_mae": [],
        "train_r2": [], "val_r2": [], "test_r2": [],
        "train_accuracy": [], "val_accuracy": [], "test_accuracy": [],
        "lr": [],
    }

    for epoch in range(Config.epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(Config.device)
            opt.zero_grad()
            crit(model(batch), batch.y).backward()
            opt.step()

        # Evaluate on train/val/test (no_grad inside evaluate_gnn_metrics)
        train_metrics = evaluate_gnn_metrics(model, train_loader, Config.device, y_scaler=scaler)
        val_metrics = evaluate_gnn_metrics(model, val_loader, Config.device, y_scaler=scaler)
        test_metrics = evaluate_gnn_metrics(model, test_loader, Config.device, y_scaler=scaler)

        # Append to history
        history["epoch"].append(epoch + 1)
        for k in ["mse", "rmse", "mae", "r2", "accuracy"]:
            history[f"train_{k}"].append(train_metrics.get(k, float('nan')))
            history[f"val_{k}"].append(val_metrics.get(k, float('nan')))
            history[f"test_{k}"].append(test_metrics.get(k, float('nan')))
        history["lr"].append(float(opt.param_groups[0]["lr"]))

        # Logging using val metrics
        print(f"Epoch {epoch+1} | Val R2: {val_metrics['r2']:.4f} | Val RMSE: {val_metrics['rmse']:.4f}")

        if val_metrics['r2'] > best_r2:
            best_r2 = val_metrics['r2']
            patience = 0
            torch.save(model.state_dict(), PLOTS_DIR_GNN / "best_gine_model.pt")
        else:
            patience += 1
            if patience >= Config.early_stopping_patience:
                print("Early stopping triggered")
                break

    # Save history JSON (so plots from last run exist)
    try:
        hist_path = PLOTS_DIR_GNN / f"gine_history_{RUN_TIMESTAMP}.json"
        with open(hist_path, "w") as fh:
            json.dump(history, fh, indent=4)

        # Create and save plots (R2, RMSE, MSE, LR)
        # R2 plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history["epoch"], history["train_r2"], label="Train R2", marker='o')
        ax.plot(history["epoch"], history["val_r2"], label="Val R2", marker='o')
        ax.plot(history["epoch"], history["test_r2"], label="Test R2", marker='o')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("R2")
        ax.set_title("GINE - R2 vs Epoch")
        ax.legend(); ax.grid()
        save_and_close_plot(fig, PLOTS_DIR_GNN / f"gine_r2_{RUN_TIMESTAMP}.png")

        # RMSE plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history["epoch"], history["train_rmse"], label="Train RMSE", marker='o')
        ax.plot(history["epoch"], history["val_rmse"], label="Val RMSE", marker='o')
        ax.plot(history["epoch"], history["test_rmse"], label="Test RMSE", marker='o')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("RMSE")
        ax.set_title("GINE - RMSE vs Epoch")
        ax.legend(); ax.grid()
        save_and_close_plot(fig, PLOTS_DIR_GNN / f"gine_rmse_{RUN_TIMESTAMP}.png")

        # Loss (MSE) plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history["epoch"], history["train_mse"], label="Train MSE")
        ax.plot(history["epoch"], history["val_mse"], label="Val MSE")
        ax.plot(history["epoch"], history["test_mse"], label="Test MSE")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_title("GINE - MSE vs Epoch")
        ax.legend(); ax.grid()
        save_and_close_plot(fig, PLOTS_DIR_GNN / f"gine_mse_{RUN_TIMESTAMP}.png")

        # LR plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history["epoch"], history["lr"], marker='o', color='orange')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.set_title("GINE - Learning Rate vs Epoch")
        ax.grid(); save_and_close_plot(fig, PLOTS_DIR_GNN / f"gine_lr_{RUN_TIMESTAMP}.png")

    except Exception as e:
        print("Error saving history/plots:", e)

    # After training, run a final train/val/test evaluation (non-invasive, does not change training)
    try:
        # final eval and print
        model.eval()
        def final_eval(loader):
            preds, trues = [], []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(Config.device)
                    preds.append(model(batch).cpu().numpy())
                    trues.append(batch.y.cpu().numpy())
            if not preds:
                return None
            p = scaler.inverse_transform(np.concatenate(preds).reshape(-1,1))
            t = scaler.inverse_transform(np.concatenate(trues).reshape(-1,1))
            p = p.reshape(-1); t = t.reshape(-1)
            mse = float(np.mean((p-t)**2)); rmse = float(np.sqrt(mse)); mae = float(np.mean(np.abs(p-t)))
            denom = float(np.sum((t - np.mean(t))**2))
            r2 = float(1.0 - np.sum((p-t)**2)/denom) if denom>0 else float('nan')
            return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

        train_res = final_eval(train_loader)
        val_res = final_eval(val_loader)
        test_res = final_eval(test_loader)

        print("\nFinal evaluation:")
        if train_res: print(f"Train: RMSE={train_res['rmse']:.4f} R2={train_res['r2']:.4f}")
        if val_res: print(f"Val:   RMSE={val_res['rmse']:.4f} R2={val_res['r2']:.4f}")
        if test_res: print(f"Test:  RMSE={test_res['rmse']:.4f} R2={test_res['r2']:.4f}")
    except Exception:
        pass

    # Remove history/json/plots saving to preserve original training behavior
    # ...existing code...

if __name__ == "__main__":
    raw_df = pd.read_parquet("data/eda_ready.parquet")

    run_full_eda(raw_df)      # Visualizes the noisy 4,800+ dataset
    run_training_pipeline(raw_df) # Trains on the clean 3,100+ unique dataset
    pass