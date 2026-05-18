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
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool, JumpingKnowledge

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
    epochs = 60
    lr = 5e-4
    weight_decay = 1e-3
    early_stopping_patience = 8  # Increased to give Scheduler room to work

    # Paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = Path("plots").resolve() / "runs" / timestamp

def setup_dirs():
    Config.plots_dir.mkdir(parents=True, exist_ok=True)
    (Config.plots_dir / "GNN").mkdir(parents=True, exist_ok=True)
    print(f"Directory verified at: {Config.plots_dir}")

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
    setup_dirs()
    print("--- Starting Full EDA ---")

    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr().abs()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, square=True)
        plt.title('Correlation Heatmap')
        plt.savefig(str(Config.plots_dir / "eda_correlation.png"))
        plt.close()

    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        plt.figure(figsize=(12, 6))
        null_counts[null_counts > 0].sort_values(ascending=False).plot(kind='bar')
        plt.title('Null Value Counts')
        plt.savefig(str(Config.plots_dir / "eda_nulls.png"))
        plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(df["standard_value"], bins=50, ax=axes[0], kde=True, log_scale=True)
    axes[0].set_title("IC50 Distribution (nM)")
    sns.histplot(df["pchembl_value"], bins=50, ax=axes[1], kde=True)
    axes[1].set_title("pIC50 Distribution")
    plt.savefig(str(Config.plots_dir / "eda_distributions.png"))
    plt.close()

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

    def get_scaffold(smi):
        mol = Chem.MolFromSmiles(smi)
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else None

    df["scaffold"] = df["smiles"].apply(get_scaffold)
    scaff_counts = df["scaffold"].value_counts().head(10)
    print(f"Unique Scaffolds Found in Raw Set: {df['scaffold'].nunique()}")

    mols = [Chem.MolFromSmiles(s) for s in scaff_counts.index]
    img = Draw.MolsToGridImage(mols, molsPerRow=5, legends=[f"Count: {c}" for c in scaff_counts.values])
    img.save(str(Config.plots_dir / "eda_top_scaffolds.png"))

# ==========================================
# FEATURIZATION & SPECIFIC GRAPH CONVERSION
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

# ==========================================
# HYBRID MODEL ARCHITECTURE
# ==========================================
class HybridGINEModel(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(cfg.num_layers):
            dim = in_channels if i == 0 else cfg.hidden_dim
            mlp = nn.Sequential(
                nn.Linear(dim, cfg.hidden_dim),
                nn.BatchNorm1d(cfg.hidden_dim),
                nn.ReLU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            )
            self.convs.append(GINEConv(mlp, edge_dim=cfg.edge_dim))
            self.bns.append(nn.BatchNorm1d(cfg.hidden_dim))

        self.jk = JumpingKnowledge(mode='cat')

        # Mean + Max dual pooling requires doubling input dimensions (hidden * layers * 2)
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_dim * cfg.num_layers * 2, 256),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        xs = []
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)
            xs.append(x)

        x_jk = self.jk(xs)

        # Hybrid Multi-resolution pooling formulation
        pool_mean = global_mean_pool(x_jk, batch)
        pool_max = global_max_pool(x_jk, batch)
        x_pooled = torch.cat([pool_mean, pool_max], dim=-1)

        return self.head(x_pooled).view(-1)

@torch.no_grad()
def evaluate_gnn_metrics(model: nn.Module, loader: DataLoader, device: str, y_scaler: Optional[StandardScaler] = None) -> Dict[str, float]:
    model.eval()
    preds, trues = [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        preds.append(out.detach().cpu().numpy())
        trues.append(batch.y.view(-1).detach().cpu().numpy())

    if len(preds) == 0:
        return {"mse": float('nan'), "rmse": float('nan'), "mae": float('nan'), "r2": float('nan')}

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

# ==========================================
# PIPELINE EXECUTION
# ==========================================
def run_training_pipeline(df_raw: pd.DataFrame):
    setup_dirs()
    seed_everything(Config.seed)

    # 1. FIXED: Trim unphysical outliers captured by EDA (pIC50 > 13.0)
    df_raw = df_raw[(df_raw['pchembl_value'] >= 2.0) & (df_raw['pchembl_value'] <= 13.0)].copy()

    # 2. PRE-CLEANING DATA (Median Consolidation)
    df = df_raw.groupby('smiles')['pchembl_value'].median().reset_index()
    print(f"Dataset clean unique compounds: {len(df)}")

    # 3. FEATURIZATION
    dataset = []
    for _, row in df.iterrows():
        g = smiles_to_graph(row['smiles'], row['pchembl_value'])
        if g: dataset.append(g)

    # 4. SCAFFOLD STRUCTURAL SPLIT
    scaffs = {}
    for i, g in enumerate(dataset):
        s = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(df.iloc[i]['smiles']))
        scaffs.setdefault(s, []).append(i)

    keys = list(scaffs.keys())
    random.shuffle(keys)
    train_idx, val_idx, test_idx = [], [], []
    for k in keys:
        ids = scaffs[k]
        if len(train_idx) < 0.8 * len(dataset): train_idx.extend(ids)
        elif len(val_idx) < 0.1 * len(dataset): val_idx.extend(ids)
        else: test_idx.extend(ids)

    # 5. TARGET VALUE NORMALIZATION
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

    # 6. OPTIMIZATION AND MODEL ARCHITECTURE INITIALIZATION
    model = HybridGINEModel(dataset[0].x.size(1), Config).to(Config.device)
    opt = torch.optim.Adam(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    crit = nn.MSELoss()

    # UPGRADE: Learning Rate plateau reduction technique to settle into steep minima
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=2
    )

    best_r2 = -float('inf')
    patience = 0
    history = {
        "epoch": [], "train_mse": [], "val_mse": [], "test_mse": [],
        "train_rmse": [], "val_rmse": [], "test_rmse": [],
        "train_mae": [], "val_mae": [], "test_mae": [],
        "train_r2": [], "val_r2": [], "test_r2": [], "lr": []
    }

    for epoch in range(Config.epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(Config.device)
            opt.zero_grad()
            crit(model(batch), batch.y).backward()
            opt.step()

        train_metrics = evaluate_gnn_metrics(model, train_loader, Config.device, y_scaler=scaler)
        val_metrics = evaluate_gnn_metrics(model, val_loader, Config.device, y_scaler=scaler)
        test_metrics = evaluate_gnn_metrics(model, test_loader, Config.device, y_scaler=scaler)

        current_lr = float(opt.param_groups[0]["lr"])
        history["epoch"].append(epoch + 1)
        history["lr"].append(current_lr)
        for k in ["mse", "rmse", "mae", "r2"]:
            history[f"train_{k}"].append(train_metrics[k])
            history[f"val_{k}"].append(val_metrics[k])
            history[f"test_{k}"].append(test_metrics[k])

        print(f"Epoch {epoch+1:02d} | Val R2: {val_metrics['r2']:.4f} | Test R2: {test_metrics['r2']:.4f} | LR: {current_lr}")

        # Update dynamic scheduler using validation R2 metric
        scheduler.step(val_metrics['r2'])

        if val_metrics['r2'] > best_r2:
            best_r2 = val_metrics['r2']
            patience = 0
            torch.save(model.state_dict(), Config.plots_dir / "GNN" / "best_gine_model.pt")
        else:
            patience += 1
            if patience >= Config.early_stopping_patience:
                print("Early stopping sequence terminated pipeline.")
                break

    # 7. GENERATE ANALYTICAL PLOTS
    try:
        with open(Config.plots_dir / "GNN" / f"gine_history_{Config.timestamp}.json", "w") as fh:
            json.dump(history, fh, indent=4)

        # Plot R2 Curves
        plt.figure(figsize=(10, 5))
        plt.plot(history["epoch"], history["train_r2"], label="Train R2", marker='o')
        plt.plot(history["epoch"], history["val_r2"], label="Val R2", marker='o')
        plt.plot(history["epoch"], history["test_r2"], label="Test R2", marker='o')
        plt.xlabel("Epochs"); plt.ylabel("R2 Value"); plt.title("GINE Hybrid - Structural R2 Profiles")
        plt.legend(); plt.grid(); plt.savefig(str(Config.plots_dir / "GNN" / f"gine_r2_{Config.timestamp}.png"))
        plt.close()

        # Plot RMSE Curves
        plt.figure(figsize=(10, 5))
        plt.plot(history["epoch"], history["train_rmse"], label="Train RMSE", marker='o')
        plt.plot(history["epoch"], history["val_rmse"], label="Val RMSE", marker='o')
        plt.plot(history["epoch"], history["test_rmse"], label="Test RMSE", marker='o')
        plt.xlabel("Epochs"); plt.ylabel("RMSE (pIC50 Units)"); plt.title("GINE Hybrid - Error Profiles")
        plt.legend(); plt.grid(); plt.savefig(str(Config.plots_dir / "GNN" / f"gine_rmse_{Config.timestamp}.png"))
        plt.close()
        print("\nPipeline execution sequence completed successfully. Output files stored.")
    except Exception as e:
        print("Visualization output error encounter:", e)

if __name__ == "__main__":
    # Point this path to your current database extracted Parquet file
    PARQUET_PATH = "data/eda_ready.parquet"
    if os.path.exists(PARQUET_PATH):
        raw_df = pd.read_parquet(PARQUET_PATH)
        run_full_eda(raw_df)
        run_training_pipeline(raw_df)
    else:
        print(f"Source file path error: {PARQUET_PATH} does not exist.")