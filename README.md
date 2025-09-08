NYC Taxi (2019) — Chunked ML Pipeline with Airflow (MWAA)

This project processes ~10.9 million NYC taxi rows (2019), converts features to fully numeric, trains a machine-learning model on the first chunk locally, and uses Airflow (AWS MWAA) to automatically train on the remaining chunks. Outputs (metrics + models) land in Amazon S3.

📦 Tech Stack

Python: pandas, scikit-learn, numpy, joblib, boto3, s3fs, fsspec

Orchestration: Apache Airflow (on AWS MWAA)

Storage: Amazon S3

OS: Windows (local dev), compute in MWAA for Airflow
