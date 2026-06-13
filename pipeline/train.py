import json
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

from pipeline.AttentionalGINEModel import AttentionalGINEModel
from pipeline.Config import Config, setup_dirs, seed_everything
from pipeline.eval import evaluate_model, clean_data, aggregate_compounds, scale_global_descriptors, \
    scaffold_split_indices, fit_y_scaler, make_dataloader_from_indices
from pipeline.graph_encoding import smiles_to_graph


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