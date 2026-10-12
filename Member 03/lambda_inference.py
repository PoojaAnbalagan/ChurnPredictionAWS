"""
Member 03: Serverless Inference Lambda Handler
Project: AWS-Based Customer Churn Prediction System

Purpose:
  AWS Lambda function that loads the pre-trained HistGradientBoostingClassifier
  pipeline from Amazon S3 (cached in /tmp) to serve:
  1. Real-time inference via REST API / API Gateway (single or batch customer records).
  2. Event-driven batch inference triggered by S3 ObjectCreated events in processed/real/.

Environment Variables:
  MODEL_BUCKET : Name of the S3 bucket (default: churnpredictionaws-team-2026)
  MODEL_KEY    : Key to the model artifact (default: models/best_model_HistGradientBoosting.joblib)
"""

import os
import io
import json
import logging
import boto3
import joblib
import pandas as pd
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)

MODEL_BUCKET = os.environ.get('MODEL_BUCKET', 'churnpredictionaws-team-2026')
MODEL_KEY = os.environ.get('MODEL_KEY', 'models/best_model_HistGradientBoosting.joblib')
LOCAL_MODEL_PATH = '/tmp/best_model_HistGradientBoosting.joblib'

s3 = boto3.client('s3')
_cached_model = None


def get_model():
    """Caches model in Lambda execution environment memory across warm invocations."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    if not os.path.exists(LOCAL_MODEL_PATH):
        logger.info(f"Downloading model artifact from s3://{MODEL_BUCKET}/{MODEL_KEY} to {LOCAL_MODEL_PATH}...")
        s3.download_file(MODEL_BUCKET, MODEL_KEY, LOCAL_MODEL_PATH)
        logger.info("Model download complete.")

    logger.info("Loading model into memory...")
    _cached_model = joblib.load(LOCAL_MODEL_PATH)
    return _cached_model


def classify_risk(probability):
    if probability < 0.35:
        return 'Low Risk'
    elif probability < 0.65:
        return 'Medium Risk'
    else:
        return 'High Risk'


def lambda_handler(event, context):
    """
    AWS Lambda entry point supporting:
    - HTTP POST / API Gateway invocation (JSON payload)
    - S3 ObjectCreated event (Batch inference trigger)
    """
    logger.info(f"Received Lambda event: {json.dumps(event)[:200]}...")

    # Pattern 1: S3 Event Trigger (Member 02 uploaded new processed dataset)
    if 'Records' in event and len(event['Records']) > 0 and 's3' in event['Records'][0]:
        record = event['Records'][0]['s3']
        src_bucket = record['bucket']['name']
        src_key = record['object']['key']

        logger.info(f"Triggered by S3 ObjectCreated event: s3://{src_bucket}/{src_key}")
        if 'holdout' not in src_key.lower():
            return {
                'statusCode': 200,
                'body': json.dumps({'message': f"Skipped non-holdout file {src_key}"})
            }

        # Download dataset from S3
        obj = s3.get_object(Bucket=src_bucket, Key=src_key)
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))

        model = get_model()
        preds = model.predict(df)
        probas = model.predict_proba(df)[:, 1]

        df['ChurnPrediction'] = preds.astype(int)
        df['ChurnProbability'] = np.round(probas, 4)
        df['RiskSegment'] = df['ChurnProbability'].apply(classify_risk)

        output_key = 'output/predictions/cell2cellholdout_predictions.csv'
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        s3.put_object(Bucket=src_bucket, Key=output_key, Body=csv_buffer.getvalue().encode('utf-8'))

        logger.info(f"Successfully processed {len(df)} records and saved to s3://{src_bucket}/{output_key}")
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Holdout batch prediction completed successfully',
                'records_processed': len(df),
                'output_location': f"s3://{src_bucket}/{output_key}"
            })
        }

    # Pattern 2: API Gateway / Direct Ingestion (Single or Batch JSON)
    try:
        body = event.get('body', event)
        if isinstance(body, str):
            body = json.loads(body)

        if isinstance(body, dict):
            records = [body]
        elif isinstance(body, list):
            records = body
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Invalid request format. Must be JSON object or list.'})
            }

        input_df = pd.DataFrame(records)
        model = get_model()

        preds = model.predict(input_df)
        probas = model.predict_proba(input_df)[:, 1]

        results = []
        for i, row in input_df.iterrows():
            cid = row.get('CustomerID', i)
            p = float(preds[i])
            prob = float(round(probas[i], 4))
            results.append({
                'CustomerID': cid,
                'ChurnPrediction': int(p),
                'ChurnProbability': prob,
                'RiskSegment': classify_risk(prob)
            })

        logger.info(f"Successfully predicted {len(results)} records.")
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'status': 'SUCCESS',
                'predictions': results
            })
        }

    except Exception as e:
        logger.error(f"Inference error: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
