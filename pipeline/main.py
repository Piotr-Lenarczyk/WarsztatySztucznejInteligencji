from pathlib import Path
import pandas as pd

from pipeline.eda import run_full_eda
from pipeline.train import train_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    PARQUET_PATH = PROJECT_ROOT / "data" / "eda_ready.parquet"

    print("Resolved:", PARQUET_PATH)

    if PARQUET_PATH.exists():
        raw_df = pd.read_parquet(PARQUET_PATH)
        run_full_eda(raw_df)
        train_model(raw_df)
    else:
        print(f"File verification checkpoint failed: {PARQUET_PATH} missing.")