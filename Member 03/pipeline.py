"""
Member 03: Machine Learning & Cloud Inference Pipeline
Project: AWS-Based Customer Churn Prediction System

Usage:
  # Run locally with local files
  python "Member 03/pipeline.py" --mode local

  # Run in AWS cloud mode with S3 (credentials auto-detected or provided)
  python "Member 03/pipeline.py" --mode cloud --bucket churnpredictionaws-team-2026 --region us-east-1
"""

import os
import sys
import io
import argparse
import logging
import joblib
import pandas as pd
import numpy as np
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Member03Pipeline")


def resolve_aws_credentials(cli_access_key=None, cli_secret_key=None):
    """
    Locates AWS credentials from:
    1. CLI arguments
    2. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    3. admin_accessKeys.csv (searches current and parent directories)
    """
    if cli_access_key and cli_secret_key:
        return cli_access_key, cli_secret_key

    env_access = os.environ.get('AWS_ACCESS_KEY_ID')
    env_secret = os.environ.get('AWS_SECRET_ACCESS_KEY')
    if env_access and env_secret:
        return env_access, env_secret

    # Search for admin_accessKeys.csv in current and parent directories
    candidate_paths = [
        'admin_accessKeys.csv',
        '../admin_accessKeys.csv',
        '../../admin_accessKeys.csv',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'admin_accessKeys.csv'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'admin_accessKeys.csv')
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                col_key = [c for c in df.columns if 'access key id' in c.lower()][0]
                col_sec = [c for c in df.columns if 'secret access key' in c.lower()][0]
                access_key = df.iloc[0][col_key].strip()
                secret_key = df.iloc[0][col_sec].strip()
                logger.info(f"Loaded AWS credentials automatically from: {os.path.abspath(path)}")
                return access_key, secret_key
            except Exception as e:
                logger.warning(f"Found {path} but failed to read credentials: {e}")

    return None, None


def get_s3_client(region='us-east-1', access_key=None, secret_key=None):
    """Creates a boto3 S3 client using resolved credentials."""
    resolved_access, resolved_secret = resolve_aws_credentials(access_key, secret_key)
    if resolved_access and resolved_secret:
        return boto3.client(
            's3',
            aws_access_key_id=resolved_access,
            aws_secret_access_key=resolved_secret,
            region_name=region
        )
    return boto3.client('s3', region_name=region)


def run_local_pipeline(
    model_path='Member 03/best_model_HistGradientBoosting.joblib',
    holdout_path='Member 03/processed/cell2cellholdout_clean.csv',
    output_path='datasets/predictions/cell2cellholdout_predictions.csv'
):
    """Executes the complete Member 03 pipeline on local files."""
    logger.info("Executing Member 03 pipeline in LOCAL mode")
    from batch_inference import run_batch_inference
    predictions = run_batch_inference(
        model_path=model_path,
        holdout_path=holdout_path,
        output_path=output_path
    )
    return {
        'status': 'SUCCESS',
        'mode': 'local',
        'evaluated_records': len(predictions),
        'predicted_churners': int((predictions['ChurnPrediction'] == 1).sum()),
        'output_file': output_path
    }


def run_cloud_pipeline(
    bucket_name='churnpredictionaws-team-2026',
    region='us-east-1',
    local_model_path='Member 03/best_model_HistGradientBoosting.joblib',
    access_key=None,
    secret_key=None
):
    """
    Connects to Amazon S3:
    1. Uploads the trained ML model pipeline to s3://<bucket>/models/
    2. Downloads cleaned holdout data from s3://<bucket>/processed/real/cell2cellholdout_clean.csv
    3. Runs model inference (churn predictions and risk probability scores)
    4. Uploads predictions to s3://<bucket>/output/predictions/cell2cellholdout_predictions.csv
    under least-privilege Member 03 IAM permissions.
    """
    logger.info("Executing Member 03 ML pipeline in AWS CLOUD mode")
    logger.info(f"Target S3 Bucket: {bucket_name} (Region: {region})")

    try:
        s3 = get_s3_client(region=region, access_key=access_key, secret_key=secret_key)
    except NoCredentialsError:
        logger.error("AWS credentials not found. Provide --aws-access-key-id and --aws-secret-access-key or place admin_accessKeys.csv in the project folder.")
        raise

    model_s3_key = 'models/best_model_HistGradientBoosting.joblib'
    processed_holdout_key = 'processed/real/cell2cellholdout_clean.csv'
    predictions_s3_key = 'output/predictions/cell2cellholdout_predictions.csv'

    # Step 1: Upload model artifact to S3 models/
    if not os.path.exists(local_model_path):
        if os.path.exists('best_model_HistGradientBoosting.joblib'):
            local_model_path = 'best_model_HistGradientBoosting.joblib'
        else:
            raise FileNotFoundError(f"Local model not found at {local_model_path}")

    logger.info(f"Uploading trained model to s3://{bucket_name}/{model_s3_key} ...")
    try:
        s3.upload_file(local_model_path, bucket_name, model_s3_key)
        logger.info("Model artifact uploaded successfully!")
    except ClientError as e:
        logger.error(f"Failed to upload model to S3: {e}")
        raise

    # Step 2: Download preprocessed holdout dataset from S3
    logger.info(f"Downloading s3://{bucket_name}/{processed_holdout_key} ...")
    try:
        holdout_obj = s3.get_object(Bucket=bucket_name, Key=processed_holdout_key)
        holdout_df = pd.read_csv(io.BytesIO(holdout_obj['Body'].read()))
        logger.info(f"Holdout dataset downloaded successfully: {holdout_df.shape} ({len(holdout_df):,} records)")
    except ClientError as e:
        logger.error(f"Failed to download holdout dataset from S3: {e}")
        raise

    # Step 3: Run model inference
    logger.info("Loading model from local artifact for fast execution...")
    model = joblib.load(local_model_path)

    logger.info("Generating churn predictions and probability scores...")
    preds = model.predict(holdout_df)
    probas = model.predict_proba(holdout_df)[:, 1]

    # Format predictions for Member 04 Athena integration
    output_df = holdout_df.copy()
    output_df['ChurnPrediction'] = preds.astype(int)
    output_df['ChurnProbability'] = np.round(probas, 4)
    output_df['RiskSegment'] = pd.cut(
        output_df['ChurnProbability'],
        bins=[-0.01, 0.35, 0.65, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )

    # Step 4: Stream and upload prediction results to S3
    logger.info(f"Uploading holdout predictions to s3://{bucket_name}/{predictions_s3_key} ...")
    csv_buffer = io.StringIO()
    output_df.to_csv(csv_buffer, index=False)
    
    try:
        s3.put_object(
            Bucket=bucket_name,
            Key=predictions_s3_key,
            Body=csv_buffer.getvalue().encode('utf-8')
        )
        logger.info("Predictions successfully written to S3!")
    except ClientError as e:
        logger.error(f"Failed to upload predictions to S3: {e}")
        raise

    # Summary metrics
    total = len(output_df)
    churners = int((output_df['ChurnPrediction'] == 1).sum())
    logger.info("=" * 60)
    logger.info("CLOUD INFERENCE PIPELINE SUCCESSFUL")
    logger.info(f"Evaluated: {total:,} records")
    logger.info(f"Predicted Churn: {churners:,} ({churners/total*100:.2f}%)")
    logger.info(f"S3 Model Artifact: s3://{bucket_name}/{model_s3_key}")
    logger.info(f"S3 Prediction Output: s3://{bucket_name}/{predictions_s3_key}")
    logger.info("Ready for Member 04 Amazon Athena queries!")
    logger.info("=" * 60)

    return {
        'status': 'SUCCESS',
        'mode': 'cloud',
        'bucket': bucket_name,
        'model_s3_key': model_s3_key,
        'predictions_s3_key': predictions_s3_key,
        'total_records': total,
        'predicted_churners': churners
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Member 03 End-to-End ML & Cloud Pipeline")
    parser.add_argument('--mode', choices=['local', 'cloud'], default='local',
                        help="Execution mode: 'local' (offline files) or 'cloud' (AWS S3)")
    parser.add_argument('--bucket', default='churnpredictionaws-team-2026',
                        help="Target S3 Bucket name")
    parser.add_argument('--region', default='us-east-1',
                        help="AWS Region")
    parser.add_argument('--model-path', default='Member 03/best_model_HistGradientBoosting.joblib')
    parser.add_argument('--holdout-path', default='Member 03/processed/cell2cellholdout_clean.csv')
    parser.add_argument('--output-path', default='datasets/predictions/cell2cellholdout_predictions.csv')
    parser.add_argument('--aws-access-key-id', default=None, help="AWS Access Key ID")
    parser.add_argument('--aws-secret-access-key', default=None, help="AWS Secret Access Key")
    args = parser.parse_args()

    # Ensure python import path has Member 03 directory
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    if args.mode == 'local':
        run_local_pipeline(
            model_path=args.model_path,
            holdout_path=args.holdout_path,
            output_path=args.output_path
        )
    else:
        run_cloud_pipeline(
            bucket_name=args.bucket,
            region=args.region,
            local_model_path=args.model_path,
            access_key=args.aws_access_key_id,
            secret_key=args.aws_secret_access_key
        )
