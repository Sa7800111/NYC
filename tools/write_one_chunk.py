import pandas as pd, s3fs

BUCKET = "nyc-taxi-ml-sangeeth"
SRC = f"s3://{BUCKET}/processed/yellow_tripdata_2016_clean.csv"
DST = f"s3://{BUCKET}/processed/chunks/chunk_000.csv"

fs = s3fs.S3FileSystem(profile="mwaa")

print("Reading head...")
print(pd.read_csv(SRC, nrows=5, storage_options={"profile":"mwaa"}).head().to_string())

it = pd.read_csv(SRC, chunksize=100_000, storage_options={"profile":"mwaa"})
first = next(it)

print("Writing first chunk rows =", len(first))
with fs.open(DST, "w") as f:
    first.to_csv(f, index=False)
print("Wrote", DST)
