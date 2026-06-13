from datetime import datetime
from pathlib import Path
import random
import torch
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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