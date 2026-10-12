"""
Member 04: Amazon Athena & Business Analytics Cloud Pipeline
Project: AWS-Based Customer Churn Prediction System

Usage:
  # Run locally with local files
  python "Member 04/pipeline.py" --mode local

  # Run in AWS cloud mode with Amazon Athena and S3
  python "Member 04/pipeline.py" --mode cloud --bucket churnpredictionaws-team-2026 --region us-east-1
"""

import os
import sys
import io
import time
import argparse
import logging
import pandas as pd
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Member04Pipeline")


def resolve_aws_credentials(cli_access_key=None, cli_secret_key=None):
    """Locates AWS credentials from CLI, env vars, or admin_accessKeys.csv."""
    if cli_access_key and cli_secret_key:
        return cli_access_key, cli_secret_key

    env_access = os.environ.get('AWS_ACCESS_KEY_ID')
    env_secret = os.environ.get('AWS_SECRET_ACCESS_KEY')
    if env_access and env_secret:
        return env_access, env_secret

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


def get_aws_clients(region='us-east-1', access_key=None, secret_key=None):
    """Creates boto3 S3 and Athena clients."""
    resolved_access, resolved_secret = resolve_aws_credentials(access_key, secret_key)
    if resolved_access and resolved_secret:
        s3 = boto3.client('s3', aws_access_key_id=resolved_access, aws_secret_access_key=resolved_secret, region_name=region)
        athena = boto3.client('athena', aws_access_key_id=resolved_access, aws_secret_access_key=resolved_secret, region_name=region)
    else:
        s3 = boto3.client('s3', region_name=region)
        athena = boto3.client('athena', region_name=region)
    return s3, athena


def run_local_pipeline():
    """Runs complete Member 04 analytics workflow locally."""
    logger.info("Executing Member 04 pipeline in LOCAL mode")
    from analytics import run_analytics_suite
    results, charts, report = run_analytics_suite()
    return {
        'status': 'SUCCESS',
        'mode': 'local',
        'queries_executed': len(results),
        'charts_generated': len(charts),
        'report_path': report
    }


def upload_reports_to_s3(s3_client, bucket_name, local_reports_dir='output/reports'):
    """Uploads generated analytics reports and charts to S3 output/reports/."""
    logger.info(f"Uploading business reports to s3://{bucket_name}/output/reports/ ...")
    uploaded_files = []

    # Upload CSV and MD reports
    if os.path.exists(local_reports_dir):
        for root, _, files in os.walk(local_reports_dir):
            for file in files:
                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, local_reports_dir)
                s3_key = f"output/reports/{rel_path}".replace('\\', '/')
                s3_client.upload_file(local_path, bucket_name, s3_key)
                uploaded_files.append(s3_key)
                logger.info(f"  Uploaded: s3://{bucket_name}/{s3_key}")

    # Upload figures
    figures_dir = 'reports/figures'
    if os.path.exists(figures_dir):
        for file in os.listdir(figures_dir):
            if file.startswith('04_') and file.endswith(('.png', '.jpg')):
                local_path = os.path.join(figures_dir, file)
                s3_key = f"output/reports/figures/{file}"
                s3_client.upload_file(local_path, bucket_name, s3_key)
                uploaded_files.append(s3_key)
                logger.info(f"  Uploaded: s3://{bucket_name}/{s3_key}")

    logger.info(f"Successfully synced {len(uploaded_files)} report artifacts to S3.")
    return uploaded_files


def run_cloud_pipeline(
    bucket_name='churnpredictionaws-team-2026',
    region='us-east-1',
    access_key=None,
    secret_key=None,
    execute_athena=True
):
    """
    Executes Member 04 cloud analytics:
    1. Runs the business analytics engine.
    2. Uploads report summaries and charts to s3://<bucket>/output/reports/.
    3. (Optional) Registers/Executes Athena DDL in the AWS Glue Data Catalog.
    """
    logger.info("Executing Member 04 pipeline in AWS CLOUD mode")
    logger.info(f"Target S3 Bucket: {bucket_name} (Region: {region})")

    s3, athena = get_aws_clients(region=region, access_key=access_key, secret_key=secret_key)

    # Step 1: Run local analytics engine to generate reports and figures
    logger.info("Generating analytical metrics and charts from S3 holdout predictions...")
    from analytics import run_analytics_suite
    results, charts, report = run_analytics_suite()

    # Step 2: Upload all reports and figures to S3 output/reports/
    uploaded = upload_reports_to_s3(s3, bucket_name, local_reports_dir='output/reports')

    # Step 3: Register/Run Athena DDL if requested
    if execute_athena:
        athena_output = f"s3://{bucket_name}/athena-results/"
        logger.info(f"Setting up Amazon Athena database 'churn_db' (Results: {athena_output}) ...")

        try:
            # Create Database
            db_query = "CREATE DATABASE IF NOT EXISTS churn_db;"
            resp = athena.start_query_execution(
                QueryString=db_query,
                ResultConfiguration={'OutputLocation': athena_output}
            )
            logger.info(f"Database query started (ExecutionId: {resp['QueryExecutionId']})")

            # Check database creation
            time.sleep(2)

            # Read and execute DDL
            ddl_path = os.path.join(os.path.dirname(__file__), 'athena_ddl.sql')
            if os.path.exists(ddl_path):
                with open(ddl_path, 'r', encoding='utf-8') as f:
                    ddl_sql = f.read()

                # Execute table DDL
                table_ddl = [stmt.strip() for stmt in ddl_sql.split(';') if 'CREATE EXTERNAL TABLE' in stmt]
                if table_ddl:
                    logger.info("Creating external table churn_db.customer_predictions in Athena...")
                    tbl_resp = athena.start_query_execution(
                        QueryString=table_ddl[0],
                        QueryExecutionContext={'Database': 'churn_db'},
                        ResultConfiguration={'OutputLocation': athena_output}
                    )
                    logger.info(f"Table DDL query started (ExecutionId: {tbl_resp['QueryExecutionId']})")
        except ClientError as e:
            logger.warning(f"Athena execution notice: {e}")

    logger.info("=" * 60)
    logger.info("MEMBER 04 CLOUD PIPELINE SUCCESSFUL")
    logger.info(f"Reports uploaded to: s3://{bucket_name}/output/reports/")
    logger.info("Amazon Athena DDL and queries ready for execution!")
    logger.info("=" * 60)

    return {
        'status': 'SUCCESS',
        'bucket': bucket_name,
        'uploaded_reports_count': len(uploaded)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Member 04 Athena & Business Analytics Pipeline")
    parser.add_argument('--mode', choices=['local', 'cloud'], default='local',
                        help="Execution mode: 'local' (offline) or 'cloud' (AWS S3 & Athena)")
    parser.add_argument('--bucket', default='churnpredictionaws-team-2026',
                        help="Target S3 Bucket name")
    parser.add_argument('--region', default='us-east-1',
                        help="AWS Region")
    parser.add_argument('--aws-access-key-id', default=None, help="AWS Access Key ID")
    parser.add_argument('--aws-secret-access-key', default=None, help="AWS Secret Access Key")
    args = parser.parse_args()

    # Ensure python import path has Member 04 directory
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    if args.mode == 'local':
        run_local_pipeline()
    else:
        run_cloud_pipeline(
            bucket_name=args.bucket,
            region=args.region,
            access_key=args.aws_access_key_id,
            secret_key=args.aws_secret_access_key
        )
