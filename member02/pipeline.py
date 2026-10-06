"""
Member 02: End-to-End Cloud & Local Data Preprocessing Pipeline
Project: AWS-Based Customer Churn Prediction System

Usage:
  # Run locally with local files
  python member02/pipeline.py --mode local

  # Run in AWS cloud mode with S3
  python member02/pipeline.py --mode cloud --bucket churnpredictionaws-team-2026 --region us-east-1
"""

import os
import sys
import io
import argparse
import json
import logging
import boto3
from botocore.exceptions import ClientError
import pandas as pd

# Import EDA and Preprocessing modules
from member02.eda import run_eda
from member02.preprocess import Cell2CellPreprocessor


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Member02Pipeline")


def run_local_pipeline(data_dir='datasets', output_dir='datasets/processed'):
    """Executes the complete Member 02 pipeline on local files."""
    train_path = os.path.join(data_dir, 'cell2celltrain.csv')
    holdout_path = os.path.join(data_dir, 'cell2cellholdout.csv')

    logger.info("Executing Member 02 pipeline in LOCAL mode")
    logger.info("Step 1: Running Exploratory Data Analysis (EDA)")
    eda_summary = run_eda(train_path=train_path)

    logger.info("Step 2: Running Data Cleaning & Preprocessing")
    from member02.preprocess import run_preprocessing
    clean_train, clean_holdout = run_preprocessing(
        train_path=train_path,
        holdout_path=holdout_path,
        output_dir=output_dir
    )

    logger.info("Local pipeline completed successfully!")
    return {
        'status': 'SUCCESS',
        'mode': 'local',
        'train_records': len(clean_train),
        'holdout_records': len(clean_holdout),
        'output_files': [
            os.path.join(output_dir, 'cell2celltrain_clean.csv'),
            os.path.join(output_dir, 'cell2cellholdout_clean.csv')
        ]
    }


def run_cloud_pipeline(bucket_name='churnpredictionaws-team-2026', region='us-east-1'):
    """
    Connects to Amazon S3, reads raw datasets from raw/real/, executes
    the cleaning & preprocessing pipeline, and uploads processed datasets
    to processed/real/ under least-privilege Member 02 permissions.
    """
    logger.info("Executing Member 02 pipeline in AWS CLOUD mode")
    logger.info(f"Target S3 Bucket: {bucket_name} (Region: {region})")

    s3 = boto3.client('s3', region_name=region)

    raw_train_key = 'raw/real/cell2celltrain.csv'
    raw_holdout_key = 'raw/real/cell2cellholdout.csv'
    processed_train_key = 'processed/real/cell2celltrain_clean.csv'
    processed_holdout_key = 'processed/real/cell2cellholdout_clean.csv'

    # 1. Download from S3 raw/real/
    logger.info(f"Downloading s3://{bucket_name}/{raw_train_key} ...")
    train_obj = s3.get_object(Bucket=bucket_name, Key=raw_train_key)
    train_df = pd.read_csv(io.BytesIO(train_obj['Body'].read()))
    logger.info(f"Train dataset downloaded successfully: {train_df.shape}")

    logger.info(f"Downloading s3://{bucket_name}/{raw_holdout_key} ...")
    holdout_obj = s3.get_object(Bucket=bucket_name, Key=raw_holdout_key)
    holdout_df = pd.read_csv(io.BytesIO(holdout_obj['Body'].read()))
    logger.info(f"Holdout dataset downloaded successfully: {holdout_df.shape}")

    # 2. Run Preprocessing
    logger.info("Cleaning and transforming datasets with zero data leakage...")
    preprocessor = Cell2CellPreprocessor()
    clean_train = preprocessor.fit_transform(train_df)
    clean_holdout = preprocessor.transform_holdout(holdout_df)

    # 3. Upload to S3 processed/real/
    logger.info(f"Uploading cleaned train data to s3://{bucket_name}/{processed_train_key} ...")
    train_csv_buffer = io.StringIO()
    clean_train.to_csv(train_csv_buffer, index=False)
    s3.put_object(
        Bucket=bucket_name,
        Key=processed_train_key,
        Body=train_csv_buffer.getvalue().encode('utf-8'),
        ContentType='text/csv'
    )
    logger.info(f"Cleaned train dataset successfully uploaded to S3!")

    logger.info(f"Uploading cleaned holdout data to s3://{bucket_name}/{processed_holdout_key} ...")
    holdout_csv_buffer = io.StringIO()
    clean_holdout.to_csv(holdout_csv_buffer, index=False)
    s3.put_object(
        Bucket=bucket_name,
        Key=processed_holdout_key,
        Body=holdout_csv_buffer.getvalue().encode('utf-8'),
        ContentType='text/csv'
    )
    logger.info(f"Cleaned holdout dataset successfully uploaded to S3!")

    # 4. Upload Preprocessing Metadata to S3
    metadata_key = 'processed/real/preprocessing_metadata.json'
    metadata = {
        'status': 'SUCCESS',
        'bucket': bucket_name,
        'train_shape': list(clean_train.shape),
        'holdout_shape': list(clean_holdout.shape),
        'features_count': len(clean_train.columns),
        'columns': clean_train.columns.tolist()
    }
    s3.put_object(
        Bucket=bucket_name,
        Key=metadata_key,
        Body=json.dumps(metadata, indent=2).encode('utf-8'),
        ContentType='application/json'
    )
    logger.info(f"Pipeline metadata uploaded to s3://{bucket_name}/{metadata_key}")

    logger.info("=" * 60)
    logger.info("MEMBER 02 CLOUD PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("Cleaned datasets are now available for Member 03 ML and Member 04 Athena!")
    logger.info("=" * 60)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Member 02 Pipeline: EDA & Preprocessing")
    parser.add_argument('--mode', choices=['local', 'cloud'], default='local',
                        help="Execution mode: 'local' (filesystem) or 'cloud' (Amazon S3)")
    parser.add_argument('--bucket', default=os.getenv('CHURN_S3_BUCKET', 'churnpredictionaws-team-2026'),
                        help="AWS S3 Bucket name")
    parser.add_argument('--region', default=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'),
                        help="AWS Region")

    args = parser.parse_args()

    if args.mode == 'local':
        run_local_pipeline()
    elif args.mode == 'cloud':
        try:
            run_cloud_pipeline(bucket_name=args.bucket, region=args.region)
        except ClientError as e:
            logger.error(f"AWS S3 Client Error: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Cloud execution error: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()
