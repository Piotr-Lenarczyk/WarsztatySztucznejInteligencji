"""
Graph Neural Network (GNN) Pipeline for Predicting pIC50 from SMILES.
This script contains the entire end-to-end pipeline including EDA, graph feature extraction, 
model architecture, training loops, and inference utilities.
"""

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, ValenceType
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool, JumpingKnowledge

from pipeline.Config import PROJECT_ROOT

# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================

class Config:
    """Central configuration for training, paths, and model hyperparameters."""
    seed = 42
    target_id = "CHEMBL2147"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model parameters
    hidden_dim = 160
    num_layers = 5
    dropout = 0.25
    edge_dim = 4

    # Training parameters
    batch_size = 128
    epochs = 80
    lr = 4e-4
    weight_decay = 8e-4
    early_stopping_patience = 10

    # Paths
    PRODUCTION_DIR = PROJECT_ROOT / "plots" / "production"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = PROJECT_ROOT / "plots" / "runs" / timestamp


def setup_dirs():
    """Create necessary directories for saving plots and models."""
    Config.plots_dir.mkdir(parents=True, exist_ok=True)
    (Config.plots_dir / "GNN").mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =============================================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

def run_full_eda(df: pd.DataFrame):
    """Run standard exploratory data analysis and generate plots."""
    setup_dirs()
    print("--- Running EDA ---")

    # Correlation Matrix
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        plt.figure(figsize=(12, 10))
        sns.heatmap(numeric_df.corr().abs(), cmap='coolwarm', center=0, square=True)
        plt.title('Correlation Analysis Matrix')
        plt.savefig(str(Config.plots_dir / "eda_correlation.png"))
        plt.close()

    # Distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(df["standard_value"], bins=50, ax=axes[0], kde=True, log_scale=True)
    axes[0].set_title("Raw IC50 (nM) Log-Scale Distribution")
    sns.histplot(df["pchembl_value"], bins=50, ax=axes[1], kde=True)
    axes[1].set_title("pIC50 Distribution")
    plt.savefig(str(Config.plots_dir / "eda_distributions.png"))
    plt.close()

    # Lipinski's Rule of 5 properties
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
    sns.countplot(x="lipinski_violations", data=df, palette="viridis", hue="lipinski_violations", legend=False)
    plt.title("Lipinski Rules Structural Violations")
    plt.savefig(str(Config.plots_dir / "eda_lipinski_violations.png"))
    plt.close()

    # Scaffold Analysis
    df["scaffold"] = df["smiles"].apply(lambda s: MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None)
    scaff_counts = df["scaffold"].value_counts().head(10)

    mols = [Chem.MolFromSmiles(s) for s in scaff_counts.index if s]
    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=5, legends=[f"Count: {c}" for c in scaff_counts.values])
        img.save(str(Config.plots_dir / "eda_top_scaffolds.png"))

# =============================================================================
# STRUCTURAL GRAPH ENCODING
# =============================================================================

def extract_atom_features(atom: Chem.Atom) -> List[float]:
    """Extract standard chemical features for a given RDKit atom."""
    features = []
    # One-hot encoded atomic number
    features.extend([float(atom.GetAtomicNum() == i) for i in [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53]])
    features.extend([float(atom.GetDegree() == i) for i in range(7)])
    features.extend([float(atom.GetFormalCharge() == i) for i in [-2, -1, 0, 1, 2]])
    features.extend([float(atom.GetHybridization() == h) for h in [
        Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2
    ]])
    features.append(float(atom.GetValence(which=ValenceType.EXPLICIT)))
    features.append(float(atom.GetNumRadicalElectrons()))
    features.append(1.0 if atom.GetIsAromatic() else 0.0)
    features.append(float(atom.GetTotalNumHs()))

    # Chirality
    chiral = atom.GetChiralTag()
    features.extend([
        float(chiral == Chem.ChiralType.CHI_TETRAHEDRAL_CW),
        float(chiral == Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    ])
    return features


def extract_bond_features(bond: Chem.Bond) -> List[float]:
    """Extract standard chemical features for a given RDKit bond."""
    bt = bond.GetBondType()
    return [
        float(bt == Chem.BondType.SINGLE),
        float(bt == Chem.BondType.DOUBLE),
        float(bt == Chem.BondType.TRIPLE),
        float(bt == Chem.BondType.AROMATIC)
    ]


def smiles_to_graph(smiles: str, y_val: float, global_desc: List[float]) -> Optional[Data]:
    """Convert a SMILES string into a PyTorch Geometric Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None

    x = torch.tensor([extract_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edges, attrs = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = extract_bond_features(b)
        edges.extend([[i, j], [j, i]])
        attrs.extend([f, f])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float) if attrs else torch.empty((0, len(extract_bond_features(mol.GetBonds()[0])) if mol.GetNumBonds() else 4), dtype=torch.float)
    y = torch.tensor([y_val], dtype=torch.float)
    g_desc = torch.tensor([global_desc], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, g_desc=g_desc)

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class AttentionalGINEModel(nn.Module):
    """
    Graph Isomorphism Network with Edge features (GINE).
    Incorporates Jumping Knowledge (JK) and global descriptors into its head.
    """
    def __init__(self, in_channels, cfg):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

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

        # Adaptive Jumping Knowledge Fusion Architecture
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

        # Extract hierarchical multi-scale neighborhood matrices
        x_jk = self.jk(xs)

        # Symmetric multi-resolution feature pooling
        pool_mean = global_mean_pool(x_jk, batch)
        pool_max = global_max_pool(x_jk, batch)

        # Inject contextually normalized raw global graph attributes
        x_pooled = torch.cat([pool_mean, pool_max, data.g_desc], dim=-1)

        return self.head(x_pooled).view(-1)

# =============================================================================
# EVALUATION & TRAINING UTILITIES
# =============================================================================

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

# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train_loop(model: nn.Module, opt: torch.optim.Optimizer, crit: nn.Module,
               train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
               scaler_y: StandardScaler):
    """Execute training iteration and capture evaluation telemetry."""
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

    for epoch in range(Config.epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(Config.device)
            opt.zero_grad()
            loss = crit(model(batch), batch.y)
            loss.backward()
            opt.step()

        train_metrics = evaluate_model(model, train_loader, Config.device, y_scaler=scaler_y)
        val_metrics = evaluate_model(model, val_loader, Config.device, y_scaler=scaler_y)
        test_metrics = evaluate_model(model, test_loader, Config.device, y_scaler=scaler_y)

        current_lr = float(opt.param_groups[0]["lr"])
        history["epoch"].append(epoch + 1)
        history["lr"].append(current_lr)

        for k in ["mse", "rmse", "mae", "r2"]:
            history[f"train_{k}"].append(train_metrics[k])
            history[f"val_{k}"].append(val_metrics[k])
            history[f"test_{k}"].append(test_metrics[k])

        print(f"Epoch {epoch+1:02d} | Train R2: {train_metrics['r2']:.3f} | Val R2: {val_metrics['r2']:.4f} | Test R2: {test_metrics['r2']:.4f} | LR: {current_lr}")

        scheduler.step(val_metrics['r2'])

        # Save best states
        if val_metrics['r2'] > best_r2:
            best_r2 = val_metrics['r2']
            patience = 0
            torch.save(model.state_dict(), Config.plots_dir / "GNN" / "best_gine_model.pt")
            try:
                joblib.dump(scaler_y, Config.plots_dir / "GNN" / "y_scaler.joblib")
            except Exception as e:
                print(f"Warning: failed to persist y-scaler: {e}")
        else:
            patience += 1
            if patience >= Config.early_stopping_patience:
                print("Early stopping sequence executed.")
                break

    return history


def save_history_and_plots(history: Dict[str, list]):
    """Output convergence plots based on the training dictionary."""
    try:
        with open(Config.plots_dir / "GNN" / f"gine_history_{Config.timestamp}.json", "w") as fh:
            json.dump(history, fh, indent=4)

        metrics = ["r2", "rmse", "mse", "mae"]
        for metric in metrics:
            plt.figure(figsize=(10, 5))
            plt.plot(history["epoch"], history[f"train_{metric}"], label=f"Train {metric.upper()}", marker='o')
            plt.plot(history["epoch"], history[f"val_{metric}"], label=f"Val {metric.upper()}", marker='o')
            plt.plot(history["epoch"], history[f"test_{metric}"], label=f"Test {metric.upper()}", marker='o')
            plt.xlabel("Epochs")
            plt.ylabel(f"{metric.upper()} Range")
            plt.title(f"{metric.upper()} Convergence")
            plt.legend()
            plt.grid()
            plt.savefig(str(Config.plots_dir / "GNN" / f"gine_{metric}_{Config.timestamp}.png"))
            plt.close()

        print("\nTraining telemetry recorded.")
    except Exception as e:
        print("Telemetry graphing failure:", e)


def train_model(df_raw: pd.DataFrame):
    """High-level training entry point: Orchestrates the entire processing and modeling loop."""
    setup_dirs()
    seed_everything(Config.seed)

    df_clean = clean_data(df_raw)
    df_agg = aggregate_compounds(df_clean)
    print(f"Unique compounds (after aggregation): {len(df_agg)}")

    scaler_g, df_scaled = scale_global_descriptors(df_agg)
    joblib.dump(scaler_g, Config.plots_dir / "GNN" / "global_descriptor_scaler.pkl")

    dataset, df_rows = [], []
    for idx, row in df_scaled.reset_index(drop=True).iterrows():
        g_desc = [float(row['heavy_atoms']), float(row['lipinski_violations'])]
        g = smiles_to_graph(row['smiles'], row['pchembl_value'], g_desc)
        if g is not None:
            dataset.append(g)
            df_rows.append(row)

    if len(dataset) == 0:
        raise RuntimeError("No valid graph data could be created from input SMILES. Aborting training.")

    df_filtered = pd.DataFrame(df_rows).reset_index(drop=True)
    print(f"Dataset size after graph conversion: {len(dataset)}")

    train_idx, val_idx, test_idx = scaffold_split_indices(dataset, df_filtered)
    scaler_y = fit_y_scaler(train_idx, dataset)
    joblib.dump(scaler_y, Config.plots_dir / "GNN" / "fitted_scaler.pkl")
    print(f"Target scaler saved to: {Config.plots_dir / 'GNN' / 'fitted_scaler.pkl'}")

    train_loader = make_dataloader_from_indices(train_idx, dataset, scaler_y, shuffle=True)
    val_loader = make_dataloader_from_indices(val_idx, dataset, scaler_y)
    test_loader = make_dataloader_from_indices(test_idx, dataset, scaler_y)

    device = Config.device if torch.cuda.is_available() and 'cuda' in str(Config.device).lower() else 'cpu'
    model = AttentionalGINEModel(in_channels=dataset[0].x.size(1), cfg=Config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    crit = nn.MSELoss()

    history = train_loop(model, opt, crit, train_loader, val_loader, test_loader, scaler_y)
    save_history_and_plots(history)
    print("Training finished.")

# =============================================================================
# INFERENCE & EXPLAINABILITY
# =============================================================================

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

    runtime_weights = Config.plots_dir / "GNN" / "best_gine_model.pt"
    if runtime_weights.exists():
        model.load_state_dict(torch.load(runtime_weights, map_location=Config.device))
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

    runtime_scaler = Config.plots_dir / "GNN" / "fitted_scaler.pkl"
    if runtime_scaler.exists():
        return joblib.load(runtime_scaler)

    print("Warning: Target StandardScaler lookup failed. Outputs will remain unscaled.")
    return None


def _load_global_scaler() -> Optional[StandardScaler]:
    """Attempt to load global descriptor scaler."""
    prod_scaler = Config.PRODUCTION_DIR / "global_descriptor_scaler.pkl"
    if prod_scaler.exists():
        return joblib.load(prod_scaler)

    runtime_scaler = Config.plots_dir / "GNN" / "global_descriptor_scaler.pkl"
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

# =============================================================================
# EXECUTION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    PARQUET_PATH = PROJECT_ROOT / "data" / "eda_ready.parquet"
    if PARQUET_PATH.exists():
        raw_df = pd.read_parquet(PARQUET_PATH)
        run_full_eda(raw_df)
        train_model(raw_df)
    else:
        print(f"File verification checkpoint failed: {PARQUET_PATH} missing.")