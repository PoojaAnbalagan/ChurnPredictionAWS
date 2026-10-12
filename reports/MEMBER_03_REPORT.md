# Member 03: Machine Learning & Cloud Churn Prediction Report
**Project**: AWS-Based Customer Churn Prediction System  
**Dataset**: Cell2Cell Telecom Dataset (~71,000 records)  
**Author**: Member 03  
**Target AWS Bucket**: `s3://churnpredictionaws-team-2026/`  

---

## 1. Executive Summary & Objective

In this end-to-end cloud churn prediction system, **Member 03** is responsible for developing, tuning, and deploying the machine learning intelligence that predicts customer churn. Member 03 interfaces directly with **Member 02** (who supplies cleaned and preprocessed datasets) and **Member 04** (who ingests model predictions into Amazon Athena for business analytics and strategic reporting).

### Key Responsibilities:
1. **Ingest Cleaned Datasets**: Consume `cell2celltrain_clean.csv` (51,047 records) and `cell2cellholdout_clean.csv` (20,000 records) from `Amazon S3 · processed/real/`.
2. **Feature Engineering & Preprocessing Pipeline**: Construct leakage-free `scikit-learn` transformation pipelines handling categorical encoding and numeric normalization.
3. **Model Exploration & Hyperparameter Optimization**: Train, cross-validate, and evaluate multiple classification algorithms (Logistic Regression, Random Forest, and Histogram-based Gradient Boosting).
4. **Model Serialization**: Export the champion model (`best_model_HistGradientBoosting.joblib`) containing the complete preprocessing and classification graph.
5. **Batch Holdout Inference**: Score 20,000 unlabelled holdout customers with binary churn classifications, probability risk scores, and business risk tiers (`Low Risk`, `Medium Risk`, `High Risk`).
6. **AWS Cloud Integration & IAM**:
   - Store model artifacts under `s3://churnpredictionaws-team-2026/models/`.
   - Store holdout predictions under `s3://churnpredictionaws-team-2026/output/predictions/`.
   - Implement least-privilege IAM policies, automated `boto3` pipelines, and an event-driven serverless AWS Lambda inference handler.

---

## 2. Model Development & Evaluation

### 2.1 Handling Class Imbalance
The training dataset contains an imbalance ratio of **2.47 : 1** (71.18% Retained vs. 28.82% Churned). To ensure the model does not bias heavily towards the majority class:
- Stratified 5-fold cross-validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) was utilized.
- Cost-sensitive learning (`class_weight='balanced'`) was configured across all candidate pipelines.
- Primary evaluation was guided by **ROC-AUC** and **F1-Score** rather than raw accuracy.

### 2.2 Model Architecture Comparison

| Model Architecture | Cross-Validation ROC-AUC | Handling of Mixed Features | Inference Speed | Selected Status |
|---|---|---|---|---|
| **Logistic Regression** (L1/L2 penalized) | ~0.6120 | Requires strict dense scaling | Very Fast | Baseline |
| **Random Forest Classifier** | ~0.6380 | Handles non-linear splits well | Moderate (High memory) | Evaluated |
| **HistGradientBoostingClassifier** | **0.6720+** | Native binning, non-linear interactions | **Fast & Lightweight** | **CHAMPION MODEL** |

### 2.3 Selected Champion Model: `HistGradientBoostingClassifier`
The Histogram-based Gradient Boosting Classifier achieved the superior performance across cross-validation folds. It excels at telecommunications tabular data due to:
- Native tree binning for high sample sizes (~51K rows).
- Early stopping to prevent overfitting on noisy usage spikes.
- Optimized hyperparameters:
  - `learning_rate`: ~0.1445
  - `max_iter`: 766
  - `max_depth`: 5
  - `max_leaf_nodes`: 17
  - `min_samples_leaf`: 60
  - `l2_regularization`: ~0.0667

---

## 3. Holdout Batch Inference Results

Batch scoring was executed across all **20,000 records** in `cell2cellholdout_clean.csv`.

### Inference Summary:
- **Total Evaluated Customers**: 20,000
- **Predicted Churners**: 8,884 customers (**44.42%**)
- **Predicted Retained**: 11,116 customers (**55.58%**)

### Business Risk Segmentation:
Predictions were enriched with risk probability thresholds tailored for retention marketing:

| Risk Tier | Churn Probability Range | Customer Count | Percentage | Recommended Business Action |
|---|---|---|---|---|
| **Low Risk** | `0.00% – 34.99%` | 4,692 | 23.46% | Standard customer lifecycle marketing |
| **Medium Risk** | `35.00% – 64.99%` | 12,926 | 64.63% | Proactive satisfaction checks, plan upgrades |
| **High Risk** | `65.00% – 100.0%` | 2,382 | 11.91% | **Immediate retention outreach**, device upgrade offers |

The resulting file was generated and formatted at:
`datasets/predictions/cell2cellholdout_predictions.csv` and `Member 03/output/predictions/cell2cellholdout_predictions.csv`.

---

## 4. AWS Cloud Integration

### 4.1 S3 Storage Architecture
```
s3://churnpredictionaws-team-2026/
├── processed/real/
│   ├── cell2celltrain_clean.csv       <- Read by Member 03 (Training)
│   └── cell2cellholdout_clean.csv     <- Read by Member 03 (Inference)
├── models/
│   └── best_model_HistGradientBoosting.joblib <- Uploaded by Member 03
└── output/predictions/
    └── cell2cellholdout_predictions.csv        <- Generated by Member 03 (Read by Member 04)
```

### 4.2 Automated Python CLI Cloud Pipeline
A dedicated script (`Member 03/pipeline.py`) provides end-to-end cloud automation:
```bash
# Execute cloud sync with Amazon S3
python "Member 03/pipeline.py" --mode cloud --bucket churnpredictionaws-team-2026 --region us-east-1
```
**Cloud Actions Executed**:
1. Connects to `churnpredictionaws-team-2026` S3 bucket.
2. Uploads the trained model artifact to `models/best_model_HistGradientBoosting.joblib`.
3. Downloads the latest cleaned holdout data from `processed/real/cell2cellholdout_clean.csv`.
4. Executes model inference with churn classes and continuous risk probabilities.
5. Streams and uploads `output/predictions/cell2cellholdout_predictions.csv` directly to S3.

### 4.3 Serverless Real-Time Inference (AWS Lambda)
To support real-time application calls and automated event pipelines, Member 03 created `Member 03/lambda_inference.py`:
- **Cold-Start Optimization**: Downloads the model from S3 to `/tmp` once and caches it in memory across warm invocations.
- **Dual Invocation Support**:
  1. **REST API / API Gateway**: Takes customer JSON feature payloads and returns real-time churn predictions with risk levels.
  2. **S3 Event Trigger**: Triggered when Member 02 writes new cleaned files to `processed/real/`, automatically triggering batch predictions.

### 4.4 AWS IAM Least-Privilege Policy
A security policy (`Member 03/iam_policy_member03.json`) was engineered following the principle of least privilege:
- `s3:GetObject` restricted strictly to `processed/*`.
- `s3:PutObject` restricted strictly to `models/*` and `output/predictions/*`.
- Full Amazon CloudWatch Logging permissions for monitoring and auditing.

---

## 5. Handover to Member 04 (Amazon Athena)

The generated predictions dataset in `output/predictions/` is fully prepared for Member 04's Athena table definition.

### Sample Athena External Table DDL:
```sql
CREATE EXTERNAL TABLE IF NOT EXISTS churn_db.customer_predictions (
  CustomerID INT,
  ServiceArea STRING,
  MonthlyRevenue DOUBLE,
  MonthlyMinutes DOUBLE,
  TotalRecurringCharge DOUBLE,
  CurrentEquipmentDays DOUBLE,
  ChurnPrediction INT,
  ChurnProbability DOUBLE,
  RiskSegment STRING
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION 's3://churnpredictionaws-team-2026/output/predictions/'
TBLPROPERTIES ("skip.header.line.count"="1");
```

Member 04 can now run Athena SQL queries such as:
```sql
-- Churn Rate by Risk Tier
SELECT RiskSegment, COUNT(*) AS TotalCustomers, AVG(ChurnProbability) AS AvgRisk
FROM churn_db.customer_predictions
GROUP BY RiskSegment;

-- Top Service Areas with Highest High-Risk Churn Concentration
SELECT ServiceArea, COUNT(*) AS HighRiskCustomers
FROM churn_db.customer_predictions
WHERE RiskSegment = 'High Risk'
GROUP BY ServiceArea
ORDER BY HighRiskCustomers DESC
LIMIT 10;
```

---

## 6. Deliverables & Artifacts Summary

| Artifact Path | Description |
|---|---|
| `Member 03/best_model_HistGradientBoosting.joblib` | Serialized champion model with preprocessing pipeline |
| `Member 03/batch_inference.py` | Local and batch scoring engine |
| `Member 03/pipeline.py` | Dual Local & S3 Cloud automated execution pipeline |
| `Member 03/lambda_inference.py` | AWS Lambda serverless real-time & event-driven inference handler |
| `Member 03/iam_policy_member03.json` | AWS IAM Least-Privilege Policy for Member 03 |
| `datasets/predictions/cell2cellholdout_predictions.csv` | Scored holdout records (20,000 customers) ready for Athena |
| `Member 03/output/predictions/` | Local synchronized prediction deliverables |
| `reports/MEMBER_03_REPORT.md` | Comprehensive ML & Cloud Architecture Report |
