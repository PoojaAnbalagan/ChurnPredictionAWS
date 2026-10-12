-- ========================================================================
-- Member 04: Amazon Athena Business Analytics SQL Queries
-- Project: AWS-Based Customer Churn Prediction System
-- Database: churn_db
-- Table: churn_db.customer_predictions
-- ========================================================================

-- ------------------------------------------------------------------------
-- QUERY 1: Overall Predicted Churn Rate & Business Risk Segmentation
-- Purpose: Executive summary of portfolio health and risk tier distribution
-- ------------------------------------------------------------------------
SELECT 
    RiskSegment,
    COUNT(*) AS CustomerCount,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS PercentageOfTotal,
    ROUND(AVG(ChurnProbability) * 100, 2) AS AvgChurnProbabilityPct,
    ROUND(SUM(MonthlyRevenue), 2) AS TotalMonthlyRevenueAtRisk
FROM churn_db.customer_predictions
GROUP BY RiskSegment
ORDER BY AvgChurnProbabilityPct DESC;


-- ------------------------------------------------------------------------
-- QUERY 2: Behavioral & Usage Profile Across Risk Tiers
-- Purpose: Identifies key usage discrepancies between loyal and churn-prone users
-- ------------------------------------------------------------------------
SELECT 
    RiskSegment,
    COUNT(*) AS CustomerCount,
    ROUND(AVG(MonthlyRevenue), 2) AS AvgMonthlyRevenue,
    ROUND(AVG(MonthlyMinutes), 1) AS AvgMonthlyMinutes,
    ROUND(AVG(TotalRecurringCharge), 2) AS AvgRecurringCharge,
    ROUND(AVG(OverageMinutes), 1) AS AvgOverageMinutes,
    ROUND(AVG(RoamingCalls), 2) AS AvgRoamingCalls,
    ROUND(AVG(CurrentEquipmentDays), 0) AS AvgEquipmentAgeDays
FROM churn_db.customer_predictions
GROUP BY RiskSegment
ORDER BY AvgMonthlyRevenue DESC;


-- ------------------------------------------------------------------------
-- QUERY 3: Churn & High-Risk Concentration by Service Area
-- Purpose: Pinpoints geographical telecom markets suffering the highest churn
-- ------------------------------------------------------------------------
SELECT 
    ServiceArea,
    COUNT(*) AS TotalCustomers,
    SUM(CASE WHEN ChurnPrediction = 1 THEN 1 ELSE 0 END) AS PredictedChurners,
    ROUND(AVG(ChurnPrediction) * 100, 2) AS ChurnRatePct,
    SUM(CASE WHEN RiskSegment = 'High Risk' THEN 1 ELSE 0 END) AS HighRiskCustomers,
    ROUND(AVG(ChurnProbability) * 100, 2) AS AvgRiskProbabilityPct,
    ROUND(SUM(MonthlyRevenue), 2) AS TotalMarketRevenue
FROM churn_db.customer_predictions
WHERE ServiceArea IS NOT NULL AND ServiceArea != 'UNKNOWN'
GROUP BY ServiceArea
HAVING COUNT(*) >= 50
ORDER BY HighRiskCustomers DESC, ChurnRatePct DESC
LIMIT 15;


-- ------------------------------------------------------------------------
-- QUERY 4: Device Age & Handset Replacement Risk Analysis
-- Purpose: Tests hypothesis that aging hardware triggers telecom churn
-- ------------------------------------------------------------------------
SELECT 
    CASE 
        WHEN CurrentEquipmentDays < 180 THEN '1. Under 6 Months'
        WHEN CurrentEquipmentDays BETWEEN 180 AND 365 THEN '2. 6 - 12 Months'
        WHEN CurrentEquipmentDays BETWEEN 366 AND 730 THEN '3. 1 - 2 Years'
        ELSE '4. Over 2 Years'
    END AS EquipmentAgeBracket,
    HandsetRefurbished,
    COUNT(*) AS CustomerCount,
    ROUND(AVG(ChurnProbability) * 100, 2) AS AvgChurnRiskPct,
    SUM(ChurnPrediction) AS PredictedChurners
FROM churn_db.customer_predictions
GROUP BY 
    CASE 
        WHEN CurrentEquipmentDays < 180 THEN '1. Under 6 Months'
        WHEN CurrentEquipmentDays BETWEEN 180 AND 365 THEN '2. 6 - 12 Months'
        WHEN CurrentEquipmentDays BETWEEN 366 AND 730 THEN '3. 1 - 2 Years'
        ELSE '4. Over 2 Years'
    END,
    HandsetRefurbished
ORDER BY EquipmentAgeBracket, HandsetRefurbished;


-- ------------------------------------------------------------------------
-- QUERY 5: Retention Team Calls vs Offer Acceptance ROI
-- Purpose: Evaluates retention team effectiveness and dissatisfied caller churn
-- ------------------------------------------------------------------------
SELECT 
    MadeCallToRetentionTeam,
    RetentionOffersAccepted,
    COUNT(*) AS CustomerCount,
    ROUND(AVG(ChurnProbability) * 100, 2) AS ChurnRiskPct,
    ROUND(AVG(MonthlyRevenue), 2) AS AvgRevenue,
    SUM(ChurnPrediction) AS PredictedChurners
FROM churn_db.customer_predictions
GROUP BY MadeCallToRetentionTeam, RetentionOffersAccepted
ORDER BY MadeCallToRetentionTeam DESC, RetentionOffersAccepted DESC;


-- ------------------------------------------------------------------------
-- QUERY 6: Customer Credit Rating & Socioeconomic Risk Analysis
-- Purpose: Discovers creditworthiness trends linked to telecom subscription churn
-- ------------------------------------------------------------------------
SELECT 
    CreditRating,
    COUNT(*) AS CustomerCount,
    ROUND(AVG(ChurnProbability) * 100, 2) AS AvgChurnProbabilityPct,
    ROUND(AVG(MonthlyRevenue), 2) AS AvgMonthlyRevenue,
    ROUND(AVG(HandsetPrice), 2) AS AvgHandsetPrice
FROM churn_db.customer_predictions
GROUP BY CreditRating
ORDER BY CreditRating ASC;


-- ------------------------------------------------------------------------
-- QUERY 7: Priority Target List: Top High-Value At-Risk Customers
-- Purpose: Actionable list for immediate proactive retention outreach
-- ------------------------------------------------------------------------
SELECT 
    CustomerID,
    ServiceArea,
    MonthlyRevenue,
    MonthlyMinutes,
    CurrentEquipmentDays,
    ROUND(ChurnProbability * 100, 2) AS ChurnProbabilityPct,
    RiskSegment
FROM churn_db.customer_predictions
WHERE RiskSegment = 'High Risk'
ORDER BY MonthlyRevenue DESC
LIMIT 50;
