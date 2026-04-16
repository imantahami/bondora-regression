# 📈 Bondora P2P Loan Strategic Analysis & Regression

This project performs a strategic analysis of the **Bondora P2P loans** dataset, focusing on predicting interest rates and identifying risk factors. By applying multiple linear regression and custom feature engineering, the model quantifies how borrower demographics and financial liabilities impact loan pricing.

## 📂 Dataset
* **Source**: [Kaggle - Bondora P2P Loans](https://www.kaggle.com/datasets/marcobeyer/bondora-p2p-loans)
* **Scope**: Analysis of 400,000+ historical loan records including borrower income, existing liabilities, and credit ratings.

## 🚀 Key Features & Updates (2026)
* **Strategic Feature Engineering**: 
  - Created `DebtToIncome` ratio to assess borrower leverage.
  - Defined `IsRisky` loans based on high debt ratios (>35%) and low employment stability.
* **Modern Data Pipeline**: Fully compatible with Pandas 2.0+ (Copy-on-Write) and optimized handling of mixed-type categorical data.
* **Statistical Inference**: Implementation of 95% Confidence Intervals for funding proportions and detailed OLS summaries.

## 📊 Model Performance
The current Multiple Linear Regression (OLS) model achieves:
* **R² Score: 0.618** (Explains 61.8% of interest rate variance).
* **Significant Predictors**: Credit Rating, Refinance Liabilities, and Existing Liabilities were found to be the most critical drivers of interest rate adjustments.

## 🛠️ Getting Started

### Prerequisites
Install the required dependencies:
```bash
pip install pandas matplotlib seaborn statsmodels scipy numpy