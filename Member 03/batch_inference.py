"""
Member 03: Batch Inference Engine
Project: AWS-Based Customer Churn Prediction System

Performs batch churn inference on the cleaned Cell2Cell holdout dataset
using the optimized HistGradientBoostingClassifier model pipeline.
Generates predictions and risk segments formatted for Amazon Athena (Member 04).
"""

import os
import sys
import argparse
import logging
import joblib
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Member03BatchInference")


def run_batch_inference(
    model_path='Member 03/best_model_HistGradientBoosting.joblib',
    holdout_path='Member 03/processed/cell2cellholdout_clean.csv',
    output_path='datasets/predictions/cell2cellholdout_predictions.csv'
):
    """
    Runs batch predictions using the saved scikit-learn Pipeline on holdout data.
    """
    if not os.path.exists(model_path):
        # Fallback path if run from Member 03 directory
        if os.path.exists('best_model_HistGradientBoosting.joblib'):
            model_path = 'best_model_HistGradientBoosting.joblib'
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")

    if not os.path.exists(holdout_path):
        if os.path.exists('processed/cell2cellholdout_clean.csv'):
            holdout_path = 'processed/cell2cellholdout_clean.csv'
        elif os.path.exists('../datasets/processed/cell2cellholdout_clean.csv'):
            holdout_path = '../datasets/processed/cell2cellholdout_clean.csv'
        elif os.path.exists('datasets/processed/cell2cellholdout_clean.csv'):
            holdout_path = 'datasets/processed/cell2cellholdout_clean.csv'
        else:
            raise FileNotFoundError(f"Holdout dataset not found at {holdout_path}")

    logger.info(f"Loading trained ML model pipeline from: {model_path}")
    model = joblib.load(model_path)

    logger.info(f"Loading cleaned holdout dataset from: {holdout_path}")
    holdout_df = pd.read_csv(holdout_path)
    logger.info(f"Holdout dataset shape: {holdout_df.shape} ({len(holdout_df):,} records)")

    # Execute predictions using pipeline
    logger.info("Executing model inference (classes + probabilities)...")
    churn_preds = model.predict(holdout_df)
    churn_probas = model.predict_proba(holdout_df)[:, 1]

    # Construct the output dataframe preserving identifiers for Member 04 Athena queries
    output_df = holdout_df.copy()
    output_df['ChurnPrediction'] = churn_preds.astype(int)
    output_df['ChurnProbability'] = np.round(churn_probas, 4)
    
    # Categorize risk tiers for business analytics
    output_df['RiskSegment'] = pd.cut(
        output_df['ChurnProbability'],
        bins=[-0.01, 0.35, 0.65, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save primary predictions file
    output_df.to_csv(output_path, index=False)
    logger.info(f"Predictions successfully saved to: {output_path}")

    # Also save a copy inside Member 03/output/predictions/ if different
    alt_output_path = os.path.join(os.path.dirname(model_path), 'output', 'predictions', 'cell2cellholdout_predictions.csv')
    if os.path.abspath(alt_output_path) != os.path.abspath(output_path):
        os.makedirs(os.path.dirname(alt_output_path), exist_ok=True)
        output_df.to_csv(alt_output_path, index=False)
        logger.info(f"Synced prediction copy saved to: {alt_output_path}")

    # Log summary statistics
    total_customers = len(output_df)
    predicted_churners = int((output_df['ChurnPrediction'] == 1).sum())
    churn_rate = (predicted_churners / total_customers) * 100
    
    logger.info("=" * 60)
    logger.info("BATCH INFERENCE SUMMARY METRICS")
    logger.info("=" * 60)
    logger.info(f"Total Evaluated Customers: {total_customers:,}")
    logger.info(f"Predicted Churners:       {predicted_churners:,} ({churn_rate:.2f}%)")
    logger.info(f"Predicted Retained:       {total_customers - predicted_churners:,} ({100 - churn_rate:.2f}%)")
    logger.info("Risk Segmentation Distribution:")
    for segment, count in output_df['RiskSegment'].value_counts().items():
        pct = (count / total_customers) * 100
        logger.info(f"  - {segment:<12}: {count:>6,} customers ({pct:>5.2f}%)")
    logger.info("=" * 60)

    return output_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Member 03 Batch Churn Inference")
    parser.add_argument('--model-path', default='Member 03/best_model_HistGradientBoosting.joblib')
    parser.add_argument('--holdout-path', default='Member 03/processed/cell2cellholdout_clean.csv')
    parser.add_argument('--output-path', default='datasets/predictions/cell2cellholdout_predictions.csv')
    args = parser.parse_args()

    run_batch_inference(
        model_path=args.model_path,
        holdout_path=args.holdout_path,
        output_path=args.output_path
    )
