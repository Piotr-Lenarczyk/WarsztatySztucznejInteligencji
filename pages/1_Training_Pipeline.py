import streamlit as st

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool, JumpingKnowledge


st.set_page_config(page_title="GINE Training Pipeline")
st.sidebar.header("GINE Training Pipeline")


def show_saved_plot(image_path: Path, title: str):
    if image_path.exists():
        st.subheader(title)
        st.image(str(image_path), use_container_width=True)

class Config:
    seed = 42
    target_id = "CHEMBL2147"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hidden_dim = 160
    num_layers = 5
    dropout = 0.25
    edge_dim = 4

    batch_size = 128
    epochs = 80
    lr = 4e-4
    weight_decay = 8e-4
    early_stopping_patience = 10

    # ADDED: Static fallback directory for the production deployment
    PRODUCTION_DIR = Path("plots/production").resolve()

    # Dynamic folder generator strictly for fresh training sessions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = Path("plots").resolve() / "runs" / timestamp

def setup_dirs():
    Config.plots_dir.mkdir(parents=True, exist_ok=True)
    (Config.plots_dir / "GNN").mkdir(parents=True, exist_ok=True)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def update_progress(progress_bar, status_placeholder, progress_fraction: float, message: str):
    if progress_bar is not None:
        progress_bar.progress(max(0, min(1, progress_fraction)))
    if status_placeholder is not None:
        status_placeholder.write(message)

def run_full_eda(df: pd.DataFrame, progress_bar=None, status_placeholder=None, start_fraction: float = 0.0, end_fraction: float = 0.3):
    setup_dirs()
    st.write("--- Running EDA ---")

    step_span = end_fraction - start_fraction
    step_count = 4
    step_idx = 0

    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        plt.figure(figsize=(12, 10))
        sns.heatmap(numeric_df.corr().abs(), cmap='coolwarm', center=0, square=True)
        plt.title('Correlation Analysis Matrix')
        plt.savefig(str(Config.plots_dir / "eda_correlation.png"))
        plt.close()
        show_saved_plot(Config.plots_dir / "eda_correlation.png", "EDA: Correlation analysis")
    step_idx += 1
    update_progress(progress_bar, status_placeholder, start_fraction + step_span * step_idx / step_count, "EDA: correlation analysis done")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(df["standard_value"], bins=50, ax=axes[0], kde=True, log_scale=True)
    axes[0].set_title("Raw IC50 (nM) Log-Scale Distribution")
    sns.histplot(df["pchembl_value"], bins=50, ax=axes[1], kde=True)
    axes[1].set_title("Standardized pIC50 Distribution Profile")
    plt.savefig(str(Config.plots_dir / "eda_distributions.png"))
    plt.close()
    show_saved_plot(Config.plots_dir / "eda_distributions.png", "EDA: Distributions")
    step_idx += 1
    update_progress(progress_bar, status_placeholder, start_fraction + step_span * step_idx / step_count, "EDA: distributions done")

    def calc_lipinski(smi):
        mol = Chem.MolFromSmiles(smi)
        if not mol: return [None]*4
        return [Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)]

    lip_cols = ["MW", "LogP", "HBD", "HBA"]
    df[lip_cols] = pd.DataFrame(df["smiles"].apply(calc_lipinski).tolist(), index=df.index)
    df["lipinski_violations"] = (df["MW"] > 500).astype(int) + (df["LogP"] > 5).astype(int) + \
                                (df["HBD"] > 5).astype(int) + (df["HBA"] > 10).astype(int)

    plt.figure(figsize=(8, 5))
    sns.countplot(x="lipinski_violations", data=df, palette="viridis")
    plt.title("Lipinski Rules Structural Violation Profile")
    plt.savefig(str(Config.plots_dir / "eda_lipinski_violations.png"))
    plt.close()
    show_saved_plot(Config.plots_dir / "eda_lipinski_violations.png", "EDA: Lipinski violations")
    step_idx += 1
    update_progress(progress_bar, status_placeholder, start_fraction + step_span * step_idx / step_count, "EDA: Lipinski analysis done")

    df["scaffold"] = df["smiles"].apply(lambda s: MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None)
    scaff_counts = df["scaffold"].value_counts().head(10)

    mols = [Chem.MolFromSmiles(s) for s in scaff_counts.index if s]
    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=5, legends=[f"Count: {c}" for c in scaff_counts.values])
        img.save(str(Config.plots_dir / "eda_top_scaffolds.png"))
        show_saved_plot(Config.plots_dir / "eda_top_scaffolds.png", "EDA: Top scaffolds")
    update_progress(progress_bar, status_placeholder, end_fraction, "EDA: finished")


def get_advanced_atom_features(atom: Chem.Atom) -> List[float]:
    features = []
    features.extend([float(atom.GetAtomicNum() == i) for i in [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53]])
    features.extend([float(atom.GetDegree() == i) for i in range(7)])
    features.extend([float(atom.GetFormalCharge() == i) for i in [-2, -1, 0, 1, 2]])
    features.extend([float(atom.GetHybridization() == h) for h in [
        Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2
    ]])
    features.append(float(atom.GetImplicitValence()))
    features.append(float(atom.GetNumRadicalElectrons()))
    features.append(1.0 if atom.GetIsAromatic() else 0.0)
    features.append(float(atom.GetTotalNumHs()))

    chiral = atom.GetChiralTag()
    features.extend([
        float(chiral == Chem.ChiralType.CHI_TETRAHEDRAL_CW),
        float(chiral == Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    ])
    return features

def get_bond_features(bond: Chem.Bond) -> List[float]:
    bt = bond.GetBondType()
    return [
        float(bt == Chem.BondType.SINGLE),
        float(bt == Chem.BondType.DOUBLE),
        float(bt == Chem.BondType.TRIPLE),
        float(bt == Chem.BondType.AROMATIC)
    ]

def smiles_to_advanced_graph(smiles: str, y_val: float, global_desc: List[float]) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None

    x = torch.tensor([get_advanced_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edges, attrs = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = get_bond_features(b)
        edges.extend([[i, j], [j, i]])
        attrs.extend([f, f])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(attrs, dtype=torch.float)
    y = torch.tensor([y_val], dtype=torch.float)
    g_desc = torch.tensor([global_desc], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, g_desc=g_desc)


class AttentionalGINEModel(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Deep contextual GINE blocks
        for i in range(cfg.num_layers):
            dim = in_channels if i == 0 else cfg.hidden_dim
            mlp = nn.Sequential(
                nn.Linear(dim, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                nn.ReLU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            )
            self.convs.append(GINEConv(mlp, edge_dim=cfg.edge_dim))
            self.bns.append(nn.BatchNorm1d(cfg.hidden_dim))

        # Upgrade: Adaptive Jumping Knowledge Fusion Architecture (LSTM/Max routing)
        self.jk = JumpingKnowledge(mode='lstm', channels=cfg.hidden_dim, num_layers=cfg.num_layers)

        self.head = nn.Sequential(
            nn.Linear((cfg.hidden_dim * 2) + 2, 256),
            nn.LayerNorm(256),
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
            x = F.dropout(x, p=0.2, training=self.training)
            xs.append(x)

        x_jk = self.jk(xs)

        pool_mean = global_mean_pool(x_jk, batch)
        pool_max = global_max_pool(x_jk, batch)

        # Inject contextually normalized raw global graph attributes (Heavy Atoms, Lipinski Violations)
        x_pooled = torch.cat([pool_mean, pool_max, data.g_desc], dim=-1)

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


def run_training_pipeline(df_raw: pd.DataFrame, progress_bar=None, status_placeholder=None, start_fraction: float = 0.3, end_fraction: float = 1.0):
    setup_dirs()
    seed_everything(Config.seed)

    df_raw = df_raw[(df_raw['pchembl_value'] >= 3.0) & (df_raw['pchembl_value'] <= 12.0)].copy()

    df = df_raw.groupby('smiles').agg({
        'pchembl_value': 'median',
        'heavy_atoms': 'first',
        'lipinski_violations': 'first'
    }).reset_index()
    st.write(f"Unique components remaining for training: {len(df)}")

    scaler_g = StandardScaler()
    df[['heavy_atoms', 'lipinski_violations']] = scaler_g.fit_transform(df[['heavy_atoms', 'lipinski_violations']])
    update_progress(progress_bar, status_placeholder, start_fraction + 0.05 * (end_fraction - start_fraction), "Training: feature normalization done")

    dataset = []
    for _, row in df.iterrows():
        g_desc = [float(row['heavy_atoms']), float(row['lipinski_violations'])]
        g = smiles_to_graph_data = smiles_to_advanced_graph(row['smiles'], row['pchembl_value'], g_desc)
        if g: dataset.append(g)
    update_progress(progress_bar, status_placeholder, start_fraction + 0.15 * (end_fraction - start_fraction), "Training: graph featurization done")

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
    update_progress(progress_bar, status_placeholder, start_fraction + 0.25 * (end_fraction - start_fraction), "Training: scaffold split done")

    scaler_y = StandardScaler()
    train_y = np.array([dataset[i].y.item() for i in train_idx]).reshape(-1, 1)
    scaler_y.fit(train_y)

    def get_loader(idxs, shuffle=False):
        data_list = []
        for i in idxs:
            g = dataset[i].clone()
            g.y = torch.tensor(scaler_y.transform([[g.y.item()]])[0], dtype=torch.float)
            data_list.append(g)
        return DataLoader(data_list, batch_size=Config.batch_size, shuffle=shuffle)

    train_loader = get_loader(train_idx, True)
    val_loader = get_loader(val_idx)
    test_loader = get_loader(test_idx)
    update_progress(progress_bar, status_placeholder, start_fraction + 0.3 * (end_fraction - start_fraction), "Training: data loaders ready")

    model = AttentionalGINEModel(dataset[0].x.size(1), Config).to(Config.device)
    opt = torch.optim.Adam(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    crit = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=2, min_lr=1e-5
    )

    best_r2 = -float('inf')
    patience = 0
    history = {
        "epoch": [], "train_mse": [], "val_mse": [], "test_mse": [],
        "train_rmse": [], "val_rmse": [], "test_rmse": [],
        "train_mae": [], "val_mae": [], "test_mae": [],
        "train_r2": [], "val_r2": [], "test_r2": [], "lr": []
    }

    st.write("\n--- Model Training... ---")
    num_epochs = Config.epochs
    for epoch in range(Config.epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(Config.device)
            opt.zero_grad()
            crit(model(batch), batch.y).backward()
            opt.step()

        train_metrics = evaluate_gnn_metrics(model, train_loader, Config.device, y_scaler=scaler_y)
        val_metrics = evaluate_gnn_metrics(model, val_loader, Config.device, y_scaler=scaler_y)
        test_metrics = evaluate_gnn_metrics(model, test_loader, Config.device, y_scaler=scaler_y)

        current_lr = float(opt.param_groups[0]["lr"])
        history["epoch"].append(epoch + 1)
        history["lr"].append(current_lr)
        for k in ["mse", "rmse", "mae", "r2"]:
            history[f"train_{k}"].append(train_metrics[k])
            history[f"val_{k}"].append(val_metrics[k])
            history[f"test_{k}"].append(test_metrics[k])

        st.write(f"Epoch {epoch+1:02d} | Train R2: {train_metrics['r2']:.3f} | Val R2: {val_metrics['r2']:.4f} | Test R2: {test_metrics['r2']:.4f} | LR: {current_lr}")

        scheduler.step(val_metrics['r2'])

        training_fraction = start_fraction + 0.35 * (end_fraction - start_fraction)
        remaining_fraction = end_fraction - training_fraction
        epoch_progress = training_fraction + remaining_fraction * ((epoch + 1) / max(1, num_epochs))
        update_progress(progress_bar, status_placeholder, epoch_progress, "Training in progress...")


        if val_metrics['r2'] > best_r2:
            best_r2 = val_metrics['r2']
            patience = 0
            torch.save(model.state_dict(), Config.plots_dir / "GNN" / "best_gine_model.pt")
        else:
            patience += 1
            if patience >= Config.early_stopping_patience:
                st.write("Early stopping sequence executed.")
                break

    update_progress(progress_bar, status_placeholder, start_fraction + 0.92 * (end_fraction - start_fraction), "Training: saving metrics and plots")

    try:
        with open(Config.plots_dir / "GNN" / f"gine_history_{Config.timestamp}.json", "w") as fh:
            json.dump(history, fh, indent=4)

        plt.figure(figsize=(10, 5))
        plt.plot(history["epoch"], history["train_r2"], label="Train R2", marker='o')
        plt.plot(history["epoch"], history["val_r2"], label="Val R2", marker='o')
        plt.plot(history["epoch"], history["test_r2"], label="Test R2", marker='o')
        plt.xlabel("Epochs"); plt.ylabel("R2 Range"); plt.title("Production GINE Attentional Profile - R2 Convergence")
        plt.legend(); plt.grid(); plt.savefig(str(Config.plots_dir / "GNN" / f"gine_r2_{Config.timestamp}.png"))

        plt.figure(figsize=(10, 5))
        plt.plot(history["epoch"], history["train_rmse"], label="Train RMSE", marker='o')
        plt.plot(history["epoch"], history["val_rmse"], label="Val RMSE", marker='o')
        plt.plot(history["epoch"], history["test_rmse"], label="Test RMSE", marker='o')
        plt.xlabel("Epochs"); plt.ylabel("RMSE Range"); plt.title("RMSE Convergence")
        plt.legend(); plt.grid(); plt.savefig(str(Config.plots_dir / "GNN" / f"gine_rmse_{Config.timestamp}.png"))

        plt.figure(figsize=(10, 5))
        plt.plot(history["epoch"], history["train_mse"], label="Train MSE", marker='o')
        plt.plot(history["epoch"], history["val_mse"], label="Val MSE", marker='o')
        plt.plot(history["epoch"], history["test_mse"], label="Test MSE", marker='o')
        plt.xlabel("Epochs"); plt.ylabel("MSE Range"); plt.title("MSE Convergence")
        plt.legend(); plt.grid(); plt.savefig(str(Config.plots_dir / "GNN" / f"gine_mse_{Config.timestamp}.png"))

        plt.figure(figsize=(10, 5))
        plt.plot(history["epoch"], history["train_mae"], label="Train MAE", marker='o')
        plt.plot(history["epoch"], history["val_mae"], label="Val MAE", marker='o')
        plt.plot(history["epoch"], history["test_mae"], label="Test MAE", marker='o')
        plt.xlabel("Epochs"); plt.ylabel("MAE Range"); plt.title("MAE Convergence")
        plt.legend(); plt.grid(); plt.savefig(str(Config.plots_dir / "GNN" / f"gine_mae_{Config.timestamp}.png"))

        plt.close()
        show_saved_plot(Config.plots_dir / "GNN" / f"gine_r2_{Config.timestamp}.png", "Training: R2 convergence")
        show_saved_plot(Config.plots_dir / "GNN" / f"gine_rmse_{Config.timestamp}.png", "Training: RMSE convergence")
        show_saved_plot(Config.plots_dir / "GNN" / f"gine_mse_{Config.timestamp}.png", "Training: MSE convergence")
        show_saved_plot(Config.plots_dir / "GNN" / f"gine_mae_{Config.timestamp}.png", "Training: MAE convergence")
        st.write("\nOptimization execution terminated  Training telemetry recorded.")
    except Exception as e:
        st.write("Telemetry graphing failure:", e)

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def calculate_virtual_auc(model, loader, device, scaler_y, activity_threshold=7.0):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds.append(out.cpu().numpy())
            trues.append(batch.y.view(-1).cpu().numpy())

    # De-normalize back to raw pIC50 units
    p = scaler_y.inverse_transform(np.concatenate(preds).reshape(-1, 1)).flatten()
    y = scaler_y.inverse_transform(np.concatenate(trues).reshape(-1, 1)).flatten()

    # Binarize true ground truth labels (1 = Active, 0 = Inactive)
    y_true_binary = (y >= activity_threshold).astype(int)

    # Safety check: ensure both classes exist in the test split
    if len(np.unique(y_true_binary)) < 2:
        st.write("AUC calculation skipped: Subset does not contain both active and inactive examples.")
        return None

    # 1. ROC-AUC Score (Probability of ranking a true active above a true inactive)
    roc_auc = roc_auc_score(y_true_binary, p)

    # 2. PR-AUC Score (Highly resilient metric for unbalanced hit datasets)
    precision, recall, _ = precision_recall_curve(y_true_binary, p)
    pr_auc = auc(recall, precision)

    st.write(f"--- Virtual Classification Metrics (Threshold pIC50 >= {activity_threshold}) ---")
    st.write(f"ROC-AUC: {roc_auc:.4f}")
    st.write(f"PR-AUC:  {pr_auc:.4f}")

    return roc_auc, pr_auc


PARQUET_PATH = "data/eda_ready.parquet"
if os.path.exists(PARQUET_PATH):
    raw_df = pd.read_parquet(PARQUET_PATH)
    pipeline_progress = st.progress(0)
    pipeline_status = st.empty()
    update_progress(pipeline_progress, pipeline_status, 0.0, "Pipeline started")
    run_full_eda(raw_df, progress_bar=pipeline_progress, status_placeholder=pipeline_status, start_fraction=0.0, end_fraction=0.3)
    run_training_pipeline(raw_df, progress_bar=pipeline_progress, status_placeholder=pipeline_status, start_fraction=0.3, end_fraction=1.0)
    update_progress(pipeline_progress, pipeline_status, 1.0, "Pipeline finished")
else:
    st.write(f"File verification checkpoint failed: {PARQUET_PATH} missing.")