"""
Member 02: Exploratory Data Analysis (EDA) Module
Project: AWS-Based Customer Churn Prediction System
Dataset: Cell2Cell Telecom Dataset (~71K records)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda(train_path='datasets/cell2celltrain.csv', 
            output_dir='reports', 
            fig_dir='reports/figures'):
    """
    Executes comprehensive exploratory data analysis on the raw training dataset.
    Generates summary metrics, distributions, correlations, and visualization plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 60)
    print("MEMBER 02: STARTING EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    # 1. Load Dataset
    print(f"Loading raw training data from: {train_path}")
    df = pd.read_csv(train_path)
    n_rows, n_cols = df.shape
    print(f"Dataset shape: {n_rows:,} rows, {n_cols} columns")

    # 2. Target Variable Analysis
    churn_counts = df['Churn'].value_counts(dropna=False)
    churn_pct = df['Churn'].value_counts(normalize=True, dropna=False) * 100
    churn_rate = (df['Churn'] == 'Yes').mean()

    print("\n--- Churn Target Distribution ---")
    for val, count in churn_counts.items():
        print(f"  {val}: {count:,} ({churn_pct[val]:.2f}%)")
    print(f"Class Imbalance Ratio (No : Yes): {churn_counts.get('No', 0) / churn_counts.get('Yes', 1):.2f} : 1")

    # 3. Missing Value Analysis
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / n_rows) * 100
    missing_df = pd.DataFrame({
        'missing_count': missing_counts,
        'missing_pct': missing_pct
    })
    missing_features = missing_df[missing_df['missing_count'] > 0].sort_values(by='missing_count', ascending=False)
    print(f"\n--- Features with Missing Values: {len(missing_features)} ---")
    for feat, row in missing_features.iterrows():
        print(f"  {feat:<25}: {int(row['missing_count']):>5} ({row['missing_pct']:.2f}%)")

    # 4. Correlation Analysis
    # Convert binary churn to 0/1 for correlation
    df['Churn_Numeric'] = df['Churn'].map({'Yes': 1, 'No': 0})
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(['CustomerID', 'Churn_Numeric'], errors='ignore')
    
    correlations = df[numeric_cols].apply(lambda x: df['Churn_Numeric'].corr(x)).sort_values()
    print("\n--- Top Negative Correlations with Churn ---")
    print(correlations.head(5).to_string())
    print("\n--- Top Positive Correlations with Churn ---")
    print(correlations.tail(5).to_string())

    # 5. Generate Visualizations
    print("\nGenerating EDA visualization plots...")
    sns.set_theme(style='whitegrid', font_scale=1.1)

    # Plot 1: Churn Class Balance
    plt.figure(figsize=(7, 5))
    colors = ['#2E86AB', '#E84855']
    ax = sns.countplot(x='Churn', data=df, palette=colors)
    plt.title('Customer Churn Class Distribution (Train Dataset)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Churn Status', fontsize=12)
    plt.ylabel('Customer Count', fontsize=12)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height):,}\n({height/n_rows*100:.1f}%)',
                    (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', color='white', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plot1_path = os.path.join(fig_dir, 'churn_distribution.png')
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"  Saved: {plot1_path}")

    # Plot 2: Missing Values Chart
    if len(missing_features) > 0:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=missing_features['missing_pct'], y=missing_features.index, palette='viridis')
        plt.title('Missing Value Percentage by Feature', fontsize=14, fontweight='bold', pad=12)
        plt.xlabel('Percentage Missing (%)', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        for p in ax.patches:
            width = p.get_width()
            ax.annotate(f'{width:.2f}%', (width + 0.05, p.get_y() + p.get_height()/2.),
                        ha='left', va='center', fontsize=9)
        plt.tight_layout()
        plot2_path = os.path.join(fig_dir, 'missing_values_bar.png')
        plt.savefig(plot2_path, dpi=300)
        plt.close()
        print(f"  Saved: {plot2_path}")

    # Plot 3: Current Equipment Days vs Churn
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Churn', y='CurrentEquipmentDays', data=df, palette=colors, showmeans=True,
                meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black"})
    plt.title('Equipment Age (Current Equipment Days) vs Churn Status', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Churn Status', fontsize=12)
    plt.ylabel('Current Equipment Days', fontsize=12)
    plt.tight_layout()
    plot3_path = os.path.join(fig_dir, 'equipment_days_vs_churn.png')
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"  Saved: {plot3_path}")

    # Plot 4: Monthly Revenue vs Churn
    plt.figure(figsize=(8, 5))
    # Filter extreme outliers for clean visualization
    q99 = df['MonthlyRevenue'].quantile(0.99)
    rev_subset = df[df['MonthlyRevenue'] < q99]
    sns.kdeplot(data=rev_subset, x='MonthlyRevenue', hue='Churn', common_norm=False, fill=True, palette=colors)
    plt.title('Monthly Revenue Distribution by Churn Status (<99th percentile)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Monthly Revenue ($)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.tight_layout()
    plot4_path = os.path.join(fig_dir, 'monthly_revenue_vs_churn.png')
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    print(f"  Saved: {plot4_path}")

    # Plot 5: Top Correlations Bar Plot
    plt.figure(figsize=(10, 6))
    top_corrs = pd.concat([correlations.head(6), correlations.tail(6)])
    bar_colors = ['#E84855' if v < 0 else '#2E86AB' for v in top_corrs.values]
    ax = sns.barplot(x=top_corrs.values, y=top_corrs.index, palette=bar_colors)
    plt.title('Top Feature Correlations with Churn', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Pearson Correlation Coefficient', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    plt.tight_layout()
    plot5_path = os.path.join(fig_dir, 'feature_correlations_with_churn.png')
    plt.savefig(plot5_path, dpi=300)
    plt.close()
    print(f"  Saved: {plot5_path}")

    # 6. Save EDA Summary Report
    eda_summary = {
        'total_rows': n_rows,
        'total_columns': n_cols,
        'churn_counts': {str(k): int(v) for k, v in churn_counts.items()},
        'churn_rate': float(churn_rate),
        'missing_columns_count': len(missing_features),
        'missing_features': {feat: {'count': int(row['missing_count']), 'pct': float(row['missing_pct'])}
                             for feat, row in missing_features.iterrows()},
        'top_positive_correlations': {k: float(v) for k, v in correlations.tail(5).items()},
        'top_negative_correlations': {k: float(v) for k, v in correlations.head(5).items()}
    }

    summary_file = os.path.join(output_dir, 'eda_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(eda_summary, f, indent=2)
    print(f"\nEDA summary JSON exported to: {summary_file}")
    print("=" * 60)
    print("MEMBER 02: EDA COMPLETE")
    print("=" * 60)

    return eda_summary


if __name__ == '__main__':
    run_eda()
