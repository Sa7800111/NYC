
import re, sys, traceback
import boto3
import pandas as pd
import s3fs

BUCKET          = "nyc-taxi-ml-sangeeth"
SRC_KEY         = "processed/yellow_tripdata_2016_clean.csv"
DST_PREFIX      = "processed/chunks"      
ROWS_PER_CHUNK  = 100_000                 
EXT             = "csv"                  
PROFILE         = "mwaa"                 
RESUME          = True                    


def log(*a): print(*a, flush=True)

def ensure_identity_and_region():
    """Log identity and bucket region, return (session, region_str)."""
    try:
        sess = boto3.Session(profile_name=PROFILE)
        sts = sess.client("sts")
        who = sts.get_caller_identity()
        s3 = sess.client("s3")
        loc = s3.get_bucket_location(Bucket=BUCKET).get("LocationConstraint") or "us-east-1"
        log(f"AWS OK | Account={who['Account']} | UserId={who['UserId']} | BucketRegion={loc} | Profile={PROFILE}")
        return sess, loc
    except Exception as e:
        log("CREDENTIAL/REGION ERROR:", e)
        raise

def list_existing_indices(fs: s3fs.S3FileSystem, dst_uri: str) -> set[int]:
    """Return existing chunk indexes at dst_uri (chunk_###.<EXT>)."""
    try:
        paths = fs.glob(f"{dst_uri}/chunk_*.{EXT}")
    except FileNotFoundError:
        return set()
    got, pat = set(), re.compile(r"chunk_(\d+)\."+re.escape(EXT)+r"$")
    for p in paths:
        m = pat.search(p)
        if m:
            got.add(int(m.group(1)))
    return got

def write_one(fs: s3fs.S3FileSystem, df: pd.DataFrame, uri: str):
    if EXT == "csv":
        with fs.open(uri, "w") as f:
            df.to_csv(f, index=False)
    else:
        # requires: pip install pyarrow
        with fs.open(uri, "wb") as f:
            df.to_parquet(f, index=False)

def main():
    # Identity & region (for logging)
    sess, bucket_region = ensure_identity_and_region()

    # S3FS + pandas storage options (hardcoded profile)
    fs = s3fs.S3FileSystem(profile=PROFILE)
    storage_opts = {"profile": PROFILE}

    src_uri = f"s3://{BUCKET}/{SRC_KEY.lstrip('/')}"
    dst_uri = f"s3://{BUCKET}/{DST_PREFIX.strip('/')}"

    # Sanity: peek at header
    log("Reading head from:", src_uri)
    head = pd.read_csv(src_uri, nrows=5, storage_options=storage_opts)
    log("Columns:", list(head.columns))

    # Resume discovery
    existing = list_existing_indices(fs, dst_uri) if RESUME else set()
    if existing:
        log(f"Resume: found {len(existing)} chunks (max idx {max(existing)})")
    else:
        log("Resume: no existing chunks found")

    # Stream + write
    it = pd.read_csv(src_uri, chunksize=ROWS_PER_CHUNK, storage_options=storage_opts)
    written = 0
    for idx, chunk in enumerate(it):
        out_uri = f"{dst_uri}/chunk_{idx:03d}.{EXT}"
        if RESUME and idx in existing:
            log("skip (exists):", out_uri)
            continue

        log(f"Writing {out_uri} rows={len(chunk)}")
        write_one(fs, chunk, out_uri)

        # Verify
        if not fs.exists(out_uri):
            raise RuntimeError(f"Write verification failed for {out_uri}")
        written += 1

        # Uncomment to test with a few files only:
        # if written >= 3:
        #     log("Test stop after 3 new chunks"); break

    log(f"Done. Wrote {written} new chunk(s).")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR:", e)
        traceback.print_exc()
        sys.exit(1)
