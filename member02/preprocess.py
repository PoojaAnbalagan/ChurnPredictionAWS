"""
Member 02: Data Cleaning and Preprocessing Module
Project: AWS-Based Customer Churn Prediction System
Dataset: Cell2Cell Telecom Dataset

This module:
1. Imputes missing values without data leakage (fit on train, transform on both train & holdout).
2. Cleans corrupted/mixed types (e.g. HandsetPrice 'Unknown' -> median numeric).
3. Encodes binary flags (Yes/No -> 1/0).
4. Encodes multi-class categoricals (CreditRating ordinal 1..7, One-Hot for Occupation, PrizmCode, MaritalStatus).
5. Retains CustomerID and ServiceArea for downstream Member 03 ML & Member 04 Athena SQL analytics.
6. Guarantees 100% column schema alignment between cleaned train and holdout datasets.
"""

import os
import json
import numpy as np
import pandas as pd


BINARY_COLS = [
    'ChildrenInHH', 'HandsetRefurbished', 'HandsetWebCapable', 'TruckOwner',
    'RVOwner', 'BuysViaMailOrder', 'RespondsToMailOffers', 'OptOutMailings',
    'NonUSTravel', 'OwnsComputer', 'HasCreditCard', 'NewCellphoneUser',
    'NotNewCellphoneUser', 'OwnsMotorcycle', 'MadeCallToRetentionTeam'
]

CREDIT_RATING_MAP = {
    '1-Highest': 1,
    '2-High': 2,
    '3-Good': 3,
    '4-Medium': 4,
    '5-Low': 5,
    '6-VeryLow': 6,
    '7-Lowest': 7
}

ONE_HOT_COLS = ['PrizmCode', 'Occupation', 'MaritalStatus']


class Cell2CellPreprocessor:
    def __init__(self):
        self.medians = {}
        self.modes = {}
        self.final_columns = []
        self.metadata = {}

    def fit(self, train_df: pd.DataFrame):
        """Learns imputation medians and modes strictly from training data."""
        print("Fitting preprocessor on training data...")
        df = train_df.copy()

        # 1. Clean HandsetPrice to compute its median
        handset_numeric = pd.to_numeric(
            df['HandsetPrice'].replace('Unknown', np.nan), errors='coerce'
        )
        self.medians['HandsetPrice'] = float(handset_numeric.median())

        # 2. Compute numeric medians for all numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns.drop(['CustomerID'], errors='ignore')
        for col in num_cols:
            self.medians[col] = float(df[col].median())

        # 3. Categorical modes
        self.modes['ServiceArea'] = 'UNKNOWN'
        self.modes['CreditRating'] = 4  # median rating: 4-Medium
        self.modes['PrizmCode'] = 'Other'
        self.modes['Occupation'] = 'Other'
        self.modes['MaritalStatus'] = 'Unknown'

        return self

    def _transform_core(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Transforms a dataframe (train or holdout) using learned parameters."""
        data = df.copy()

        # 1. Preserve CustomerID & ServiceArea
        customer_ids = data['CustomerID']
        service_areas = data['ServiceArea'].fillna('UNKNOWN')

        # 2. Target Variable
        if 'Churn' in data.columns and is_train:
            churn_series = data['Churn'].map({'Yes': 1, 'No': 0})
        elif 'Churn' in data.columns and not is_train:
            churn_series = pd.Series(np.nan, index=data.index, name='Churn')
        else:
            churn_series = pd.Series(np.nan, index=data.index, name='Churn')

        # 3. Clean HandsetPrice
        data['HandsetPrice'] = pd.to_numeric(
            data['HandsetPrice'].replace('Unknown', np.nan), errors='coerce'
        ).fillna(self.medians['HandsetPrice'])

        # 4. Binary Features (Yes/No -> 1/0)
        for col in BINARY_COLS:
            if col in data.columns:
                data[col] = data[col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

        # 5. Homeownership (Known -> 1, Unknown -> 0)
        if 'Homeownership' in data.columns:
            data['Homeownership'] = data['Homeownership'].map({'Known': 1, 'Unknown': 0}).fillna(0).astype(int)

        # 6. CreditRating (Ordinal 1 to 7)
        if 'CreditRating' in data.columns:
            data['CreditRating'] = data['CreditRating'].map(CREDIT_RATING_MAP).fillna(self.modes['CreditRating']).astype(int)

        # 7. Impute Remaining Numeric Features
        for col, median_val in self.medians.items():
            if col in data.columns:
                data[col] = data[col].fillna(median_val)

        # 8. Categoricals for One-Hot Encoding
        for col in ONE_HOT_COLS:
            if col in data.columns:
                data[col] = data[col].fillna(self.modes[col])

        # Generate One-Hot Encoded Dummies (0/1 integers)
        dummies = pd.get_dummies(data[ONE_HOT_COLS], prefix=ONE_HOT_COLS, drop_first=False, dtype=int)

        # Drop original raw categorical columns
        cols_to_drop = ['CustomerID', 'Churn', 'ServiceArea', 'Homeownership'] + ONE_HOT_COLS
        remaining_numeric = data.drop(columns=[c for c in cols_to_drop if c in data.columns])

        # Combine: CustomerID, ServiceArea, remaining_numeric, dummies, Churn
        transformed = pd.concat([
            customer_ids,
            service_areas,
            remaining_numeric,
            dummies,
            churn_series
        ], axis=1)

        return transformed

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fits on train and returns cleaned train dataset."""
        self.fit(train_df)
        cleaned_train = self._transform_core(train_df, is_train=True)
        # Record final columns schema
        self.final_columns = cleaned_train.columns.tolist()
        return cleaned_train

    def transform_holdout(self, holdout_df: pd.DataFrame) -> pd.DataFrame:
        """Transforms holdout set and guarantees identical columns with train set."""
        if not self.final_columns:
            raise ValueError("Preprocessor has not been fitted yet! Call fit_transform first.")
        
        cleaned_holdout = self._transform_core(holdout_df, is_train=False)

        # Reindex to guarantee exact same column order and presence
        for col in self.final_columns:
            if col not in cleaned_holdout.columns:
                cleaned_holdout[col] = 0

        cleaned_holdout = cleaned_holdout[self.final_columns]
        return cleaned_holdout


def run_preprocessing(train_path='datasets/cell2celltrain.csv',
                      holdout_path='datasets/cell2cellholdout.csv',
                      output_dir='datasets/processed',
                      metadata_dir='reports'):
    """
    Main function to execute the full data cleaning and preprocessing pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    print("=" * 60)
    print("MEMBER 02: RUNNING DATA CLEANING & PREPROCESSING PIPELINE")
    print("=" * 60)

    # 1. Load Raw Datasets
    print(f"Loading raw train dataset: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Loading raw holdout dataset: {holdout_path}")
    holdout_df = pd.read_csv(holdout_path)

    # 2. Check Duplicates
    train_dups = train_df.duplicated(subset=['CustomerID']).sum()
    holdout_dups = holdout_df.duplicated(subset=['CustomerID']).sum()
    print(f"Duplicate CustomerIDs - Train: {train_dups}, Holdout: {holdout_dups}")
    if train_dups > 0:
        train_df = train_df.drop_duplicates(subset=['CustomerID'])
    if holdout_dups > 0:
        holdout_df = holdout_df.drop_duplicates(subset=['CustomerID'])

    # 3. Fit and Transform
    preprocessor = Cell2CellPreprocessor()
    clean_train = preprocessor.fit_transform(train_df)
    clean_holdout = preprocessor.transform_holdout(holdout_df)

    # 4. Verify Integrity
    assert list(clean_train.columns) == list(clean_holdout.columns), "Column mismatch between train and holdout!"
    # Check nulls except Churn in holdout
    train_nulls = clean_train.isnull().sum().sum()
    holdout_nulls_ex_churn = clean_holdout.drop(columns=['Churn']).isnull().sum().sum()
    print(f"Post-cleaning Null Values - Train: {train_nulls}, Holdout (excl Churn): {holdout_nulls_ex_churn}")
    assert train_nulls == 0, f"Unexpected null values in cleaned train: {train_nulls}"
    assert holdout_nulls_ex_churn == 0, f"Unexpected null values in cleaned holdout: {holdout_nulls_ex_churn}"

    # 5. Save Processed CSVs
    train_out_path = os.path.join(output_dir, 'cell2celltrain_clean.csv')
    holdout_out_path = os.path.join(output_dir, 'cell2cellholdout_clean.csv')

    print(f"Exporting cleaned train dataset to: {train_out_path}")
    clean_train.to_csv(train_out_path, index=False)

    print(f"Exporting cleaned holdout dataset to: {holdout_out_path}")
    clean_holdout.to_csv(holdout_out_path, index=False)

    # 6. Save Preprocessing Metadata
    meta = {
        'train_raw_shape': list(train_df.shape),
        'holdout_raw_shape': list(holdout_df.shape),
        'train_clean_shape': list(clean_train.shape),
        'holdout_clean_shape': list(clean_holdout.shape),
        'total_features': len(clean_train.columns),
        'column_list': clean_train.columns.tolist(),
        'train_churn_distribution': {
            'retained_0': int((clean_train['Churn'] == 0).sum()),
            'churned_1': int((clean_train['Churn'] == 1).sum()),
            'churn_rate': float(clean_train['Churn'].mean())
        },
        'imputation_medians': preprocessor.medians,
        'status': 'SUCCESS'
    }

    meta_file = os.path.join(metadata_dir, 'preprocessing_metadata.json')
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Exported preprocessing metadata to: {meta_file}")

    print("=" * 60)
    print(f"MEMBER 02 PREPROCESSING COMPLETE: Cleaned datasets ready for Member 03 & 04!")
    print(f"  Train: {clean_train.shape[0]:,} rows x {clean_train.shape[1]} cols")
    print(f"  Holdout: {clean_holdout.shape[0]:,} rows x {clean_holdout.shape[1]} cols")
    print("=" * 60)

    return clean_train, clean_holdout


if __name__ == '__main__':
    run_preprocessing()
