# churnPredictionAWS

An end-to-end cloud-based churn prediction system built on AWS, combining 
real-world telecom customer data (Cell2Cell dataset, ~71K records) to predict customer churn and generate actionable 
business insights.

## Pipeline Overview
1. **Data Ingestion (AWS)** — Raw datasets uploaded to S3, with event-driven 
   Lambda functions for validation and CloudWatch logging.
2. **EDA & Preprocessing** — Data quality checks, cleaning, and comparison of 
   real vs. synthetic data distributions.
3. **Machine Learning** — Feature engineering and training of classification 
   models to predict churn, evaluated via accuracy, precision, recall, F1, 
   and ROC-AUC.
4. **Business Analytics** — SQL queries via AWS Athena on combined 
   predictions + customer data to surface churn patterns, key customer 
   segments, and retention recommendations.

## Architecture
Raw CSVs → S3 (`raw/`) → S3 Event Trigger → Lambda (validate + log) → 
CloudWatch → S3 (`processed/`, `output/`) → Athena → Business Report

## Team
| Member | Responsibility |
|---|---|
| Member 1 | AWS Infrastructure & Cloud Pipeline |
| Member 2 | EDA & Data Preprocessing |
| Member 3 | Machine Learning / Churn Prediction |
| Member 4 | SQL & Business Analytics |
