import argparse
import json
from datetime import date
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import s3fs
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

# ======== EDIT THESE IF NEEDED ========
BUCKET        = "nyc-taxi-ml-sangeeth"
CHUNKS_PREFIX = "processed/chunks"     # folder that contains chunk_XXX.csv
ART_PREFIX    = "artifacts"            # metrics & models go under artifacts/<YYYY-MM-DD>/
PROFILE       = "mwaa"                 # local AWS CLI profile
CHUNK_EXT     = "csv"                  # "csv" or "parquet"
TARGET_COL    = "total_amount"         # <-- main target for this script
# ======================================

# When predicting total_amount, these columns are parts of it. Drop to avoid leakage.
LEAKY_FOR_TOTAL = {"fare_amount", "mta_tax", "tip_amount", "tolls_amount", "total_amount"}

def _chunk_uri(idx: int) -> str:
    return f"s3://{BUCKET}/{CHUNKS_PREFIX}/chunk_{idx:03d}.{CHUNK_EXT}"

def _read_chunk(fs: s3fs.S3FileSystem, idx: int) -> pd.DataFrame:
    uri = _chunk_uri(idx)
    if CHUNK_EXT == "parquet":
        with fs.open(uri, "rb") as f:
            return pd.read_parquet(f)
    with fs.open(uri, "r") as f:
        return pd.read_csv(f)

def _prepare_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in data. Columns: {list(df.columns)}")

    y = df[TARGET_COL].to_numpy()

    # drop target and (if target is total_amount) its additive components to prevent leakage
    drop_cols = {TARGET_COL}
    if TARGET_COL == "total_amount":
        drop_cols |= (LEAKY_FOR_TOTAL - {TARGET_COL})

    X_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # keep only numeric features
    X_df = X_df.select_dtypes(include=["number"])
    feature_names = X_df.columns.to_numpy()
    X = X_df.to_numpy(dtype=np.float64)

    # basic guardrails
    if X.shape[0] < 10 or X.shape[1] < 1:
        raise ValueError(f"Not enough data to train. X shape={X.shape}, y len={len(y)}")

    return X, y, feature_names

def _train_regression(X: np.ndarray, y: np.ndarray) -> Tuple[RandomForestRegressor, Dict[str, float]]:
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(
        n_estimators=300, n_jobs=-1, random_state=42,
        max_depth=None, min_samples_split=2, min_samples_leaf=1
    )
    model.fit(Xtr, ytr)
    pred = model.predict(Xva)

    rmse = float(mean_squared_error(yva, pred, squared=False))
    mae  = float(mean_absolute_error(yva, pred))
    r2   = float(r2_score(yva, pred))

    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    return model, metrics

def _write_metrics_and_model(fs: s3fs.S3FileSystem,
                             ds: str,
                             chunk_idx: int,
                             metrics: Dict[str, Any],
                             model_obj: Any,
                             feature_names: np.ndarray) -> None:
    out_dir = f"s3://{BUCKET}/{ART_PREFIX}/{ds}"
    fs.makedirs(out_dir, exist_ok=True)

    # Append to run_summary.csv
    csv_uri = f"{out_dir}/run_summary.csv"
    header = "run_date,chunk_id,target,rmse,mae,r2\n"
    row = f"{ds},{chunk_idx},{TARGET_COL},{metrics['rmse']},{metrics['mae']},{metrics['r2']}\n"

    if not fs.exists(csv_uri):
        with fs.open(csv_uri, "wb") as f:
            f.write(header.encode("utf-8"))
    with fs.open(csv_uri, "ab") as f:
        f.write(row.encode("utf-8"))

    # Save a compact JSON
    json_uri = f"{out_dir}/metrics_chunk_{chunk_idx:03d}.json"
    with fs.open(json_uri, "wb") as jf:
        jf.write(json.dumps(
            {"chunk_id": chunk_idx, "target": TARGET_COL, **metrics},
            indent=2
        ).encode("utf-8"))

    # Save the trained model + metadata
    model_dir = f"{out_dir}/models"
    fs.makedirs(model_dir, exist_ok=True)
    model_uri = f"{model_dir}/chunk_{chunk_idx:03d}.joblib"

    payload = {
        "target": TARGET_COL,
        "model_type": "RandomForestRegressor",
        "model": model_obj,
        "feature_names": feature_names.tolist(),
        "chunk_id": chunk_idx,
    }
    with fs.open(model_uri, "wb") as mf:
        joblib.dump(payload, mf)

def main():
    parser = argparse.ArgumentParser(description="Train ML on one S3 chunk and write metrics/model to S3.")
    parser.add_argument("--chunk", type=int, required=True, help="Chunk index (e.g., 0 -> chunk_000.csv)")
    args = parser.parse_args()

    fs = s3fs.S3FileSystem(profile=PROFILE)
    ds = date.today().isoformat()

    print(f"Reading chunk {args.chunk} from: {_chunk_uri(args.chunk)}", flush=True)
    df = _read_chunk(fs, args.chunk)

    print(f"Columns: {list(df.columns)}", flush=True)
    X, y, feature_names = _prepare_xy(df)

    print(f"Training regression for target='{TARGET_COL}' on X{X.shape}, y({len(y)})...", flush=True)
    model, metrics = _train_regression(X, y)

    print("Metrics:", metrics, flush=True)
    _write_metrics_and_model(fs, ds, args.chunk, metrics, model, feature_names)
    print("Done.", flush=True)

if __name__ == "__main__":
    main()
