"""
AWS Glue Python ETL Job: ChurnPredictionPreprocessor
Member 02: EDA & Data Preprocessing
Project: AWS-Based Customer Churn Prediction System

This AWS Glue job runs serverless ETL on Amazon S3:
- Reads 's3://churnpredictionaws-team-2026/raw/real/'
- Cleans, imputes missing values, and one-hot encodes features
- Writes clean CSV partitions directly to 's3://churnpredictionaws-team-2026/processed/real/'
"""

import sys
import boto3
import io
import pandas as pd
import numpy as np

# AWS Glue imports (available in AWS Glue Python-Shell / PySpark environment)
try:
    from awsglue.utils import getResolvedOptions
    args = getResolvedOptions(sys.argv, ['BUCKET_NAME'])
    BUCKET_NAME = args['BUCKET_NAME']
except Exception:
    BUCKET_NAME = 'churnpredictionaws-team-2026'

print(f"Starting AWS Glue Preprocessing Job for bucket: {BUCKET_NAME}")

s3 = boto3.client('s3')

# 1. Download raw data
print("Reading raw CSVs from S3...")
train_raw = s3.get_object(Bucket=BUCKET_NAME, Key='raw/real/cell2celltrain.csv')
train_df = pd.read_csv(io.BytesIO(train_raw['Body'].read()))

holdout_raw = s3.get_object(Bucket=BUCKET_NAME, Key='raw/real/cell2cellholdout.csv')
holdout_df = pd.read_csv(io.BytesIO(holdout_raw['Body'].read()))

# 2. Impute and encode (Core Logic)
from member02.preprocess import Cell2CellPreprocessor
preprocessor = Cell2CellPreprocessor()
clean_train = preprocessor.fit_transform(train_df)
clean_holdout = preprocessor.transform_holdout(holdout_df)

# 3. Write processed output to S3
print("Writing processed clean CSVs to S3...")
train_buf = io.StringIO()
clean_train.to_csv(train_buf, index=False)
s3.put_object(
    Bucket=BUCKET_NAME,
    Key='processed/real/cell2celltrain_clean.csv',
    Body=train_buf.getvalue().encode('utf-8'),
    ContentType='text/csv'
)

holdout_buf = io.StringIO()
clean_holdout.to_csv(holdout_buf, index=False)
s3.put_object(
    Bucket=BUCKET_NAME,
    Key='processed/real/cell2cellholdout_clean.csv',
    Body=holdout_buf.getvalue().encode('utf-8'),
    ContentType='text/csv'
)

print("AWS Glue ETL Preprocessing Job finished successfully!")
