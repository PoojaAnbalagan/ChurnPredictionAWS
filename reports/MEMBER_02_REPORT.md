# Member 02: Exploratory Data Analysis (EDA) & Data Preprocessing Report
**Project**: AWS-Based Customer Churn Prediction System  
**Dataset**: Cell2Cell Telecom Dataset (~71,000 records)  
**Author**: Member 02  
**Target AWS Bucket**: `s3://churnpredictionaws-team-2026/`  

---

## 1. Executive Summary & Objective

In this end-to-end cloud churn prediction architecture, **Member 02** bridges raw cloud data ingestion (Member 01) and predictive machine learning / analytics (Members 03 & 04).

### Key Responsibilities:
1. **Read raw datasets** from `Amazon S3 · raw/real/` (`cell2celltrain.csv` and `cell2cellholdout.csv`).
2. **Conduct comprehensive Exploratory Data Analysis (EDA)** to quantify target class imbalance, inspect distributions, and identify churn indicators.
3. **Perform Data Cleaning**: Handle missing values with zero data leakage (fit on train, transform on holdout), resolve corrupted/mixed values, and ensure type consistency.
4. **Data Preprocessing & Encoding**: Transform binary flags, encode multi-class variables, and preserve relational keys (`CustomerID`, `ServiceArea`) for Member 04's Athena SQL queries.
5. **Write Cleaned Datasets** to `Amazon S3 · processed/real/`:
   - `cell2celltrain_clean.csv` (51,047 rows × 69 columns)
   - `cell2cellholdout_clean.csv` (20,000 rows × 69 columns)
6. **Cloud Integration & IAM**: Configure least-privilege IAM policies, automated `boto3` pipelines, AWS Lambda event handlers, and AWS Glue ETL scripts.

---

## 2. Exploratory Data Analysis (EDA) Findings

### 2.1 Target Class Imbalance (`Churn`)
- **Dataset Size**: 51,047 training records, 20,000 holdout records.
- **Target Variable**: `Churn` (Yes / No).
- **Distribution**:
  - **No (Retained)**: 36,336 customers (71.18%)
  - **Yes (Churned)**: 14,711 customers (28.82%)
- **Imbalance Ratio**: **2.47 : 1**.
  > *Machine Learning Note for Member 03*: Class weighting (`scale_pos_weight` in XGBoost/LightGBM or `class_weight='balanced'` in Random Forest/Logistic Regression) or SMOTE / threshold tuning should be used to optimize ROC-AUC and F1-score.

### 2.2 Missing Value Analysis
14 features contained missing values in the raw dataset. Because missingness was non-trivial across demographic and behavioral features, systematic median/mode imputation was applied:

| Feature | Missing Count (Train) | Missing % | Imputation Strategy | Rationale |
|---|---|---|---|---|
| `AgeHH1` & `AgeHH2` | 909 | 1.78% | Train Median (36.0 & 0.0) | Skewed distribution; median preserves central tendency |
| `PercChangeRevenues` | 367 | 0.72% | Train Median (-0.3) | Avoids distortion from extreme revenue outliers |
| `PercChangeMinutes` | 367 | 0.72% | Train Median (-5.0) | Negative median indicates general minute usage drops |
| `MonthlyRevenue` | 156 | 0.31% | Train Median ($48.16) | Standard telecommunications ARPU median |
| `MonthlyMinutes` | 156 | 0.31% | Train Median (355.0 min) | Robust to extreme high-minute users |
| `TotalRecurringCharge` | 156 | 0.31% | Train Median ($44.99) | Baseline recurring monthly subscription fee |
| `OverageMinutes`, `RoamingCalls`, `DirectorAssistedCalls` | 156 | 0.31% | Train Median (0.0) | Sparsely used specialized call services |
| `ServiceArea` | 24 | 0.05% | Constant `'UNKNOWN'` | Essential categorical key for Member 04 Athena queries |
| `Handsets`, `HandsetModels`, `CurrentEquipmentDays` | 1 | 0.00% | Train Median | Single isolated record nulls |

### 2.3 Top Feature Correlations with Churn
- **Positively Correlated with Churn**:
  1. `CurrentEquipmentDays` (+0.1037): Customers with older phones churn significantly more as devices age or battery degrades.
  2. `RetentionCalls` (+0.0653): Customers calling retention teams are already dissatisfied.
  3. `RetentionOffersAccepted` (+0.0350): Prior at-risk customers who took offers remain high churn risks.
  4. `UniqueSubs` (+0.0345): More lines on account correlate with complex billing churn.
- **Negatively Correlated with Churn (Retention Drivers)**:
  1. `TotalRecurringCharge` (-0.0613): Higher commitment plans have lower churn.
  2. `MonthlyMinutes` (-0.0502): Highly engaged, active callers churn less.
  3. `OffPeakCallsInOut` (-0.0408) & `PeakCallsInOut` (-0.0400): Strong everyday calling habits correlate with loyalty.

---

## 3. Data Cleaning and Preprocessing Pipeline

### 3.1 Preprocessing Transformations
1. **Relational Identifiers**:
   - `CustomerID` preserved as int identifier (no scaling).
   - `ServiceArea` retained as categorical string (missing imputed to `'UNKNOWN'`) so Member 04's Athena query *"Churn by service area"* operates smoothly.
2. **Corrupted / Mixed Value Handling**:
   - `HandsetPrice`: Contains numeric strings and `'Unknown'`. Cleaned by converting `'Unknown'` to NaN and imputing with the training median ($60.00).
3. **Binary Feature Mapping (Yes/No -> 1/0)**:
   - 15 features converted to binary integer flags: `ChildrenInHH`, `HandsetRefurbished`, `HandsetWebCapable`, `TruckOwner`, `RVOwner`, `BuysViaMailOrder`, `RespondsToMailOffers`, `OptOutMailings`, `NonUSTravel`, `OwnsComputer`, `HasCreditCard`, `NewCellphoneUser`, `NotNewCellphoneUser`, `OwnsMotorcycle`, `MadeCallToRetentionTeam`.
   - `Homeownership`: Mapped `Known` -> 1, `Unknown` -> 0.
4. **Ordinal Feature Encoding**:
   - `CreditRating`: 7 levels mapped to integer risk rank:  
     `{'1-Highest': 1, '2-High': 2, '3-Good': 3, '4-Medium': 4, '5-Low': 5, '6-VeryLow': 6, '7-Lowest': 7}`.
5. **One-Hot Encoding**:
   - Applied to multi-class categoricals:
     - `PrizmCode` (`Suburban`, `Town`, `Other`, `Rural`)
     - `Occupation` (`Professional`, `Crafts`, `Other`, `Self`, `Retired`, `Student`, `Homemaker`, `Clerical`)
     - `MaritalStatus` (`Yes`, `No`, `Unknown`)
   - Encoded strictly with binary integers (0/1).
6. **Target Column**:
   - `cell2celltrain_clean.csv`: `Churn` encoded as `1` (Yes) and `0` (No).
   - `cell2cellholdout_clean.csv`: `Churn` preserved as null/empty for Member 03 prediction.

### 3.2 Leakage-Free Schema Alignment
- Preprocessing parameters (medians, modes, dummy column schema) are learned **strictly from the training dataset**.
- The exact same schema is applied to `cell2cellholdout.csv`, ensuring both output files have **69 columns in identical order**.

---

## 4. How to Integrate Member 02 into the AWS Cloud

Integration with the cloud architecture is supported across multiple patterns:

### Method 1: Python CLI Pipeline with `boto3` (Direct S3 Integration)
Run the automated cloud pipeline:
```bash
python member02/pipeline.py --mode cloud --bucket churnpredictionaws-team-2026 --region us-east-1
```
**Cloud Workflow**:
1. Connects to `churnpredictionaws-team-2026` S3 bucket.
2. Downloads `raw/real/cell2celltrain.csv` and `raw/real/cell2cellholdout.csv`.
3. Runs `Cell2CellPreprocessor` cleaning & encoding.
4. Streams and uploads `processed/real/cell2celltrain_clean.csv` and `processed/real/cell2cellholdout_clean.csv` back to S3.
5. Emits CloudWatch-formatted execution metadata.

### Method 2: Serverless AWS Lambda Function
- Code: `member02/lambda_preprocessor.py`
- Trigger options:
  - **S3 ObjectCreated Event**: Triggered when Member 01 uploads files to `raw/real/`.
  - **Direct Invocation**: Called via Member 01's validation Lambda upon successful validation.
  - **AWS Step Functions**: Orchestrated state machine executing Member 01 -> Member 02 -> Member 03 -> Member 04.

### Method 3: Enterprise AWS Glue ETL Job
- Code: `member02/glue_etl_job.py`
- Deployed as an AWS Glue Python-shell or PySpark job for automated batch preprocessing directly in the AWS Data Lake.

### Method 4: Least-Privilege AWS IAM Policy
- Policy: `member02/iam_policy_member02.json`
- Grants Member 02:
  - `s3:GetObject` on `arn:aws:s3:::churnpredictionaws-team-2026/raw/*`
  - `s3:PutObject` on `arn:aws:s3:::churnpredictionaws-team-2026/processed/*`
  - Full CloudWatch logging permissions.

---

## 5. Artifacts and Generated Deliverables

| File / Folder | Purpose |
|---|---|
| `member02/eda.py` | Complete Exploratory Data Analysis script |
| `member02/preprocess.py` | Data Cleaning and Preprocessing engine |
| `member02/pipeline.py` | Dual Local & S3 Cloud automated execution pipeline |
| `member02/lambda_preprocessor.py` | Serverless AWS Lambda preprocessing handler |
| `member02/glue_etl_job.py` | AWS Glue serverless ETL batch script |
| `member02/iam_policy_member02.json` | AWS IAM Least-Privilege Policy for Member 02 |
| `notebooks/02_eda_and_data_preprocessing.ipynb` | Interactive Jupyter Notebook with visualization |
| `datasets/processed/cell2celltrain_clean.csv` | Cleaned 51,047 training records ready for ML |
| `datasets/processed/cell2cellholdout_clean.csv` | Cleaned 20,000 holdout records ready for prediction |
| `reports/figures/` | High-resolution EDA visualization charts |
| `reports/eda_summary.json` | Machine-readable EDA summary metrics |
| `reports/preprocessing_metadata.json` | Preprocessing audit log and column mapping |
