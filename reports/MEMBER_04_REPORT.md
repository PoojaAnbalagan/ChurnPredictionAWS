# Member 04: Amazon Athena & Business Analytics Report
**Project**: AWS-Based Customer Churn Prediction System  
**Dataset**: Cell2Cell Telecom Dataset (~71,000 records) & Holdout Predictions (20,000 records)  
**Author**: Member 04  
**Target AWS Bucket**: `s3://churnpredictionaws-team-2026/`  

---

## 1. Executive Summary & Objective

In this end-to-end cloud churn prediction architecture, **Member 04** is responsible for translating technical machine learning outputs into strategic business intelligence. Member 04 queries the predictions and customer features stored in Amazon S3 using **Amazon Athena** (serverless distributed SQL) to surface churn patterns, identify high-risk customer segments, quantify financial exposure, and provide actionable retention recommendations.

### Key Responsibilities:
1. **Schema & Table Definitions (AWS Glue & Athena)**: Create external tables in the AWS Glue Data Catalog mapping S3 prediction datasets (`s3://churnpredictionaws-team-2026/output/predictions/`).
2. **Serverless SQL Execution**: Execute SQL queries in Amazon Athena to analyze customer churn risk across behavioral, geographic, and financial dimensions.
3. **Business Segment Discovery**: Group customers into actionable risk tiers (`Low Risk`, `Medium Risk`, `High Risk`) and quantify total monthly recurring revenue at risk.
4. **Key Driver Analytics**:
   - Churn concentration across telecom service areas (`ServiceArea`).
   - Hardware aging effects (`CurrentEquipmentDays`, refurbished status).
   - Retention team ROI & offer acceptance behavior (`MadeCallToRetentionTeam`, `RetentionOffersAccepted`).
   - Socioeconomic patterns (`CreditRating`, `IncomeGroup`, `HandsetPrice`).
5. **Reporting & Cloud Integration**: Generate executive summaries, automated figures, query CSV tables, and sync all reports to `s3://churnpredictionaws-team-2026/output/reports/`.

---

## 2. Amazon Athena Architecture & DDL

### 2.1 Serverless Data Lake Schema
```
Amazon S3 · output/predictions/
└── cell2cellholdout_predictions.csv (20,000 scored customer records)
          │
          ▼ (Metadata via AWS Glue Data Catalog)
Amazon Athena: churn_db.customer_predictions
          │
          ▼ (Serverless SQL Queries)
Amazon S3 · output/reports/
├── EXECUTIVE_BUSINESS_REPORT.md
├── q1_risk_distribution.csv ... q7_high_value_targets.csv
└── figures/ (Visual charts & risk breakdowns)
```

### 2.2 Table Creation DDL (`Member 04/athena_ddl.sql`)
```sql
CREATE DATABASE IF NOT EXISTS churn_db;

CREATE EXTERNAL TABLE IF NOT EXISTS churn_db.customer_predictions (
  CustomerID INT,
  ServiceArea STRING,
  MonthlyRevenue DOUBLE,
  MonthlyMinutes DOUBLE,
  TotalRecurringCharge DOUBLE,
  DirectorAssistedCalls DOUBLE,
  OverageMinutes DOUBLE,
  RoamingCalls DOUBLE,
  PercChangeMinutes DOUBLE,
  PercChangeRevenues DOUBLE,
  CurrentEquipmentDays INT,
  HandsetRefurbished INT,
  MadeCallToRetentionTeam INT,
  RetentionOffersAccepted INT,
  CreditRating INT,
  IncomeGroup INT,
  HandsetPrice DOUBLE,
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

---

## 3. Core Business Insights

### 3.1 Overall Portfolio Health & Revenue Exposure

Query 1 evaluated the portfolio of 20,000 customers scored by Member 03's champion model:

| Risk Segment | Customer Count | Percentage | Avg Churn Probability | Monthly Revenue at Risk | Annualized Revenue Exposure |
|---|---|---|---|---|---|
| **High Risk** (`> 65%`) | **2,382** | **11.91%** | **70.62%** | **$139,320.19** | **$1,671,842.28** |
| **Medium Risk** (`35% – 65%`) | 12,926 | 64.63% | 46.85% | $747,543.82 | $8,970,525.84 |
| **Low Risk** (`< 35%`) | 4,692 | 23.46% | 24.32% | $275,541.36 | $3,306,496.32 |
| **Total Portfolio** | 20,000 | 100.0% | 44.42% | $1,162,405.37 | $13,948,864.44 |

> **Key Takeaway**: Nearly **$1.67 Million in annual recurring revenue** is at imminent risk from the top 11.91% high-risk accounts.

---

### 3.2 Behavioral Profiles: Loyal vs Churn-Prone Users

| Risk Segment | Avg Monthly Revenue (ARPU) | Avg Monthly Minutes | Avg Recurring Charge | Avg Overage Minutes | Avg Equipment Age (Days) |
|---|---|---|---|---|---|
| **High Risk** | **$58.49** | 522.6 min | $46.83 | 42.8 min | **468 days** (~1.3 years) |
| **Medium Risk** | $57.83 | 518.2 min | $46.81 | 39.5 min | 402 days (~1.1 years) |
| **Low Risk** | $58.73 | 526.0 min | $47.38 | 38.6 min | **318 days** (~0.87 years) |

> **Key Insight**: High-risk customers have significantly older handsets (**468 days vs. 318 days**) and higher overage bill shock, despite paying similar recurring base rates.

---

### 3.3 Equipment Age as a Primary Churn Trigger

Analysis revealed phone hardware aging is one of the strongest churn catalysts:
- **Under 6 Months**: Churn probability is lowest (**~38.1%**).
- **6 to 12 Months**: Churn probability rises to **~41.4%**.
- **1 to 2 Years**: Churn probability climbs to **~47.8%**.
- **Over 2 Years**: Churn probability peaks at **~51.2%**.
- **Refurbished Phone Owners**: Exhibit an additional **2.5% to 4.0% higher churn propensity** across all age brackets compared to original phone buyers.

---

### 3.4 Top Geographic Markets with Highest High-Risk Churn

Geographical analysis revealed distinct high-risk customer clusters:
1. `NYCBRO917` (New York / Brooklyn): 74 High-Risk Customers (43.2% Churn Rate)
2. `DALFTW817` (Dallas / Fort Worth): 68 High-Risk Customers (46.1% Churn Rate)
3. `LOSLAX213` (Los Angeles): 61 High-Risk Customers (44.8% Churn Rate)
4. `CHICHI312` (Chicago): 58 High-Risk Customers (45.3% Churn Rate)
5. `HOUHOU713` (Houston): 53 High-Risk Customers (42.9% Churn Rate)

---

### 3.5 Retention Team Dynamics & Offer Acceptance

- Customers who **called the retention team** have an average churn probability of **~53.6%**, compared to **~44.1%** for customers who never called.
- When customers **accepted a retention offer**, their subsequent churn probability remains elevated at **~49.2%**.
- **Diagnosis**: Retention efforts are currently **reactive** (initiated only when a customer is already prepared to cancel). Proactive outreach must precede the retention call.

---

## 4. Strategic Retention Action Plan

1. **Preemptive 365-Day Handset Upgrade Campaign**:
   - Send personalized upgrade vouchers to customers reaching 10-12 months of phone age before battery or hardware dissatisfaction peaks.
2. **VIP Concierge for Top 50 High-Value Accounts**:
   - Direct outreach to the highest-revenue accounts identified in `q7_high_value_targets.csv` (subscribers generating $150–$300+/month).
3. **Bill-Shock Overage Buffer**:
   - Automatically notify customers when approaching overage minutes and suggest plan adjustments to prevent bill shock churn.
4. **Targeted Regional Offers**:
   - Run competitive retention promotions in New York, Dallas, and Los Angeles metro areas.

---

## 5. Deliverables & Cloud Artifacts Summary

| Artifact Path | Description |
|---|---|
| `Member 04/athena_ddl.sql` | Amazon Athena external database & table creation DDL |
| `Member 04/queries.sql` | Suite of 7 analytical business queries for Athena |
| `Member 04/analytics.py` | Analytics engine executing SQL queries and generating charts |
| `Member 04/pipeline.py` | Dual-mode Local and AWS Cloud pipeline syncing to Athena & S3 |
| `Member 04/iam_policy_member04.json` | AWS IAM Least-Privilege Policy for Member 04 |
| `output/reports/EXECUTIVE_BUSINESS_REPORT.md` | Executive Summary report synced to `s3://churnpredictionaws-team-2026/output/reports/` |
| `output/reports/*.csv` | 7 CSV query result tables |
| `output/reports/figures/` | Visual risk segmentation and service area charts synced to S3 |
| `reports/MEMBER_04_REPORT.md` | Comprehensive Business Analytics Report |
