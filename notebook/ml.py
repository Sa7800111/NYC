import io
from urllib.parse import urlparse
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

LEAKY_FOR_TOTAL = {"fare_amount", "mta_tax", "tip_amount", "tolls_amount", "total_amount"}

def load_df(path: str, profile: str | None) -> pd.DataFrame:
    if path.startswith("s3://"):
        # try s3fs (fast); fallback to boto3
        try:
            import s3fs  # type: ignore
            storage_opts = {"profile": profile} if profile else {}
            return pd.read_csv(path, storage_options=storage_opts)
        except ModuleNotFoundError:
            import boto3
            sess = boto3.Session(profile_name=profile) if profile else boto3.Session()
            s3 = sess.client("s3")
            u = urlparse(path)
            obj = s3.get_object(Bucket=u.netloc, Key=u.path.lstrip("/"))
            return pd.read_csv(io.BytesIO(obj["Body"].read()))
    return pd.read_csv(path)

def prepare_xy(df: pd.DataFrame, target: str):
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in columns: {list(df.columns)}")
    y = df[target].to_numpy()

    drop_cols = {target}
    if target == "total_amount":
        drop_cols |= (LEAKY_FOR_TOTAL - {target})  # avoid leakage

    Xdf = (df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
             .select_dtypes(include=["number"])
             .replace([np.inf, -np.inf], np.nan)
             .dropna())
    if len(Xdf) != len(df):
        y = y[Xdf.index]  # align

    if Xdf.shape[0] < 10 or Xdf.shape[1] < 1:
        raise ValueError(f"Not enough data to train. X{Xdf.shape}, ylen={len(y)}")

    return Xdf.to_numpy(dtype=np.float64), y, Xdf.columns.tolist()

def main():
    ap = argparse.ArgumentParser(description="Train RF on one chunk and print metrics.")
    ap.add_argument("--path", required=True, help="Local path or s3://bucket/key")
    ap.add_argument("--target", default="total_amount")
    ap.add_argument("--profile", default=None, help="AWS profile for S3 (e.g., mwaa)")
    ap.add_argument("--save", default=None, help="Optional: save model to this .joblib path")
    args = ap.parse_args()

    print("Loading:", args.path)
    df = load_df(args.path, args.profile)
    print("Shape:", df.shape)

    X, y, feats = prepare_xy(df, args.target)
    print(f"Training on X{X.shape}, y({len(y)}), target='{args.target}'")

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xva)

    rmse = mean_squared_error(yva, pred, squared=False)
    mae  = mean_absolute_error(yva, pred)
    r2   = r2_score(yva, pred)

    print("\n=== Metrics ===")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"R^2 : {r2:.6f}")

    # Top features (optional)
    try:
        order = np.argsort(rf.feature_importances_)[::-1][:10]
        print("\nTop features:")
        for i in order:
            print(f"  {feats[i]:<25} {rf.feature_importances_[i]:.4f}")
    except Exception:
        pass

    if args.save:
        payload = {
            "model_type": "RandomForestRegressor",
            "model": rf,
            "feature_names": feats,
            "target": args.target,
            "trained_on": args.path,
        }
        joblib.dump(payload, args.save)
        print("Saved model ->", args.save)

if __name__ == "__main__":
    main()
