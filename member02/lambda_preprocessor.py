"""
AWS Lambda Function: churnPredictionPreprocessor
Member 02: EDA & Data Preprocessing
Project: AWS-Based Customer Churn Prediction System

This Lambda function:
1. Receives an S3 event or Step Function trigger when Member 01 finishes validating raw data.
2. Reads 'raw/real/cell2celltrain.csv' and 'raw/real/cell2cellholdout.csv' from S3 bucket.
3. Cleans, imputes, and encodes the datasets.
4. Writes 'processed/real/cell2celltrain_clean.csv' and 'processed/real/cell2cellholdout_clean.csv' to S3.
5. Emits structured CloudWatch logs for operational monitoring.
"""

import os
import io
import json
import logging
import boto3
import pandas as pd
from member02.preprocess import Cell2CellPreprocessor

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')

BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'churnpredictionaws-team-2026')
RAW_PREFIX = 'raw/real/'
PROCESSED_PREFIX = 'processed/real/'


def lambda_handler(event, context):
    logger.info(f"Received event: {json.dumps(event)}")
    bucket = event.get('bucket', BUCKET_NAME)

    raw_train_key = f"{RAW_PREFIX}cell2celltrain.csv"
    raw_holdout_key = f"{RAW_PREFIX}cell2cellholdout.csv"
    clean_train_key = f"{PROCESSED_PREFIX}cell2celltrain_clean.csv"
    clean_holdout_key = f"{PROCESSED_PREFIX}cell2cellholdout_clean.csv"

    try:
        logger.info(f"Member 02: Downloading raw files from s3://{bucket}/{RAW_PREFIX}")
        
        # 1. Fetch train data
        train_obj = s3.get_object(Bucket=bucket, Key=raw_train_key)
        train_df = pd.read_csv(io.BytesIO(train_obj['Body'].read()))
        logger.info(f"Loaded train dataset: {train_df.shape[0]} rows, {train_df.shape[1]} columns")

        # 2. Fetch holdout data
        holdout_obj = s3.get_object(Bucket=bucket, Key=raw_holdout_key)
        holdout_df = pd.read_csv(io.BytesIO(holdout_obj['Body'].read()))
        logger.info(f"Loaded holdout dataset: {holdout_df.shape[0]} rows, {holdout_df.shape[1]} columns")

        # 3. Clean and Preprocess
        logger.info("Executing Cell2CellPreprocessor...")
        preprocessor = Cell2CellPreprocessor()
        clean_train = preprocessor.fit_transform(train_df)
        clean_holdout = preprocessor.transform_holdout(holdout_df)

        logger.info(f"Preprocessing completed. Clean train: {clean_train.shape}, Clean holdout: {clean_holdout.shape}")

        # 4. Upload clean train data
        logger.info(f"Writing clean train CSV to s3://{bucket}/{clean_train_key}")
        train_buffer = io.StringIO()
        clean_train.to_csv(train_buffer, index=False)
        s3.put_object(
            Bucket=bucket,
            Key=clean_train_key,
            Body=train_buffer.getvalue().encode('utf-8'),
            ContentType='text/csv'
        )

        # 5. Upload clean holdout data
        logger.info(f"Writing clean holdout CSV to s3://{bucket}/{clean_holdout_key}")
        holdout_buffer = io.StringIO()
        clean_holdout.to_csv(holdout_buffer, index=False)
        s3.put_object(
            Bucket=bucket,
            Key=clean_holdout_key,
            Body=holdout_buffer.getvalue().encode('utf-8'),
            ContentType='text/csv'
        )

        response_body = {
            'status': 'SUCCESS',
            'stage': 'Member 02 - Preprocessing',
            'bucket': bucket,
            'clean_train_key': clean_train_key,
            'clean_holdout_key': clean_holdout_key,
            'train_rows': len(clean_train),
            'holdout_rows': len(clean_holdout),
            'features_count': len(clean_train.columns),
            'message': 'Cleaned datasets successfully generated and uploaded to S3 processed/real/'
        }

        logger.info(f"Member 02 Lambda executed successfully: {json.dumps(response_body)}")

        return {
            'statusCode': 200,
            'body': json.dumps(response_body)
        }

    except Exception as e:
        logger.error(f"Error during Member 02 preprocessing: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'FAILURE',
                'error': str(e),
                'stage': 'Member 02 - Preprocessing'
            })
        }
