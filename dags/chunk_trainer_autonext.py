from __future__ import annotations
from datetime import datetime, date
import json, re
from typing import List

import boto3
import numpy as np
import pandas as pd
import s3fs
from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
import joblib

BUCKET        = "nyc-taxi-ml-sangeeth"
CHUNKS_PREFIX = "processed/chunks/"
ART_PREFIX    = "artifacts"
TARGET_COL    = "total_amount"          # <— predict total_amount (regression). Change if you want.
ALLOWED_EXTS  = ("csv","parquet")

def _list_chunk_keys() -> List[str]:
    s3 = boto3.client("s3")
    rx = re.compile(rf"^{re.escape(CHUNKS_PREFIX)}chunk_(\d+)\.({'|'.join(ALLOWED_EXTS)})$")
    pairs = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=CHUNKS_PREFIX):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            m = rx.match(k)
            if m: pairs.append((int(m.group(1)), k))
    pairs.sort(key=lambda t: t[0])
    return [k for _, k in pairs]

def _next_key(keys: List[str], cur: str|None) -> str:
    if not keys: raise RuntimeError("No chunk files found.")
    if not cur or cur not in keys: return keys[0]
    i = keys.index(cur)
    return keys[(i+1) % len(keys)]

def _read(fs: s3fs.S3FileSystem, key: str) -> pd.DataFrame:
    uri = f"s3://{BUCKET}/{key}"
    if key.endswith(".parquet"):
        with fs.open(uri, "rb") as f: return pd.read_parquet(f)
    with fs.open(uri, "r") as f: return pd.read_csv(f)

def _prepare_Xy(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        raise RuntimeError(f"Target column '{TARGET_COL}' not found. Got: {list(df.columns)}")
    y = df[TARGET_COL].to_numpy()
    drop = {TARGET_COL}
    if TARGET_COL == "total_amount":
        drop |= {"fare_amount","mta_tax","tip_amount","tolls_amount"}  # avoid leakage
    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore").select_dtypes(include=["number"]).to_numpy(dtype=np.float64)
    return X, y

def _write_outputs(fs: s3fs.S3FileSystem, ds: str, chunk_key: str, metrics: dict, model_obj):
    out_dir = f"s3://{BUCKET}/{ART_PREFIX}/{ds}"
    fs.makedirs(out_dir, exist_ok=True)
    # CSV
    csv_uri = f"{out_dir}/run_summary.csv"
    if not fs.exists(csv_uri):
        with fs.open(csv_uri, "wb") as f:
            f.write(b"run_date,chunk_key,target,rmse,mae,r2,accuracy,f1_weighted\n")
    with fs.open(csv_uri, "ab") as f:
        f.write(f"{ds},{chunk_key},{TARGET_COL},{metrics.get('rmse','')},{metrics.get('mae','')},{metrics.get('r2','')},{metrics.get('accuracy','')},{metrics.get('f1_weighted','')}\n".encode())
    # JSON
    with fs.open(f"{out_dir}/metrics_{chunk_key.split('/')[-1].rsplit('.',1)[0]}.json","wb") as jf:
        jf.write(json.dumps({"chunk_key": chunk_key, "target": TARGET_COL, **metrics}, indent=2).encode())
    # Model
    fs.makedirs(f"{out_dir}/models", exist_ok=True)
    with fs.open(f"{out_dir}/models/{chunk_key.split('/')[-1].rsplit('.',1)[0]}.joblib","wb") as mf:
        joblib.dump(model_obj, mf)
    # Helpful marker
    with fs.open(f"s3://{BUCKET}/{ART_PREFIX}/_latest_run.txt","wb") as lf:
        lf.write(f"{ds},{chunk_key}\n".encode())

with DAG(
    dag_id="chunk_trainer_autonext",
    start_date=datetime(2025,1,1),
    schedule="@daily",      # change to "@hourly" to iterate faster
    catchup=False,
    max_active_runs=1,
    tags=["chunks","ml"],
) as dag:

    @task
    def list_chunks() -> list[str]:
        return _list_chunk_keys()

    @task
    def choose_next(keys: list[str]) -> str:
        cur = Variable.get("next_chunk_key", default_var="", deserialize_json=False)
        return _next_key(keys, cur if cur else None)

    @task
    def train_on_chunk(chunk_key: str, ds: str) -> dict:
        fs = s3fs.S3FileSystem()
        df = _read(fs, chunk_key)
        X, y = _prepare_Xy(df)

        # Decide problem by heuristics; here TARGET_COL is total_amount (regression)
        metrics = {}
        if TARGET_COL == "total_amount":  # regression
            Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42).fit(Xtr, ytr)
            pred  = model.predict(Xva)
            metrics["rmse"] = float(mean_squared_error(yva, pred, squared=False))
            metrics["mae"]  = float(mean_absolute_error(yva, pred))
            metrics["r2"]   = float(r2_score(yva, pred))
            model_obj = ("rf_reg", model)
        else:  # generic fallback: classification
            Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)
            try:
                scaler = StandardScaler()
                Xtr_s = scaler.fit_transform(Xtr); Xva_s = scaler.transform(Xva)
                clf = LogisticRegression(max_iter=1000).fit(Xtr_s, ytr)
                pred = clf.predict(Xva_s)
                metrics["accuracy"]     = float(accuracy_score(yva, pred))
                metrics["f1_weighted"]  = float(f1_score(yva, pred, average="weighted"))
                model_obj = ("logreg", clf, scaler)
            except Exception:
                clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42).fit(Xtr, ytr)
                pred = clf.predict(Xva)
                metrics["accuracy"]     = float(accuracy_score(yva, pred))
                metrics["f1_weighted"]  = float(f1_score(yva, pred, average="weighted"))
                model_obj = ("rf_clf", clf)

        _write_outputs(fs, ds, chunk_key, metrics, model_obj)
        return {"chunk_key": chunk_key, "metrics": metrics}

    @task
    def bump_pointer(keys: list[str], processed_key: str) -> str:
        nxt = _next_key(keys, processed_key)
        Variable.set("next_chunk_key", nxt)
        return nxt

    keys = list_chunks()
    key  = choose_next(keys)
    _m   = train_on_chunk(key, ds="{{ ds }}")
    _bp  = bump_pointer(keys, processed_key=key)
