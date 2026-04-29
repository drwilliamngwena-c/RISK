# Credit Risk Modeling: Basel II/III (PD, LGD, EAD) - Comprehensive Guide

This project implements an end-to-end credit risk modeling framework based on the Basel II/III pillars using Python. It covers the end-to-end pipeline from data preprocessing to Expected Loss (EL) estimation.

## Project Structure

- `DATA_PROCESSING/`: Modular notebooks for missing values, outlier handling, categorical encoding, and data splitting.
- `MODELS/`: Notebooks for baseline model training (Logistic/Linear Regression) and evaluation metrics.
- `Credit_Risk_Model_Definition.ipynb`: Documentation of dataset schema and Basel targets.
- `Credit_Risk_Model_Interpretations.ipynb`: Business drivers and regulatory alignment logic.

## Key Components

1.  **PD (Probability of Default)**: Modeled via Logistic Regression to predict binary default status.
2.  **LGD (Loss Given Default)**: Modeled via Linear Regression to estimate recovery rates on defaulted loans.
3.  **EAD (Exposure at Default)**: Modeled via Linear Regression to predict total exposure at time of default.

## Data Preprocessing

Data preprocessing is crucial for building robust credit risk models. This includes handling missing values, managing outliers, and encoding categorical features.

### Example: Outlier Capping
We cap extreme values, such as `person_income`, to prevent them from disproportionately influencing our models.

```python
def cap_outliers(series, percentile=0.99):
    upper_limit = series.quantile(percentile)
    return series.clip(upper=upper_limit)

df['person_income'] = cap_outliers(df['person_income'])
```

### Example: Categorical Encoding
Categorical features like `person_home_ownership` are converted into a numerical format using One-Hot Encoding, which is suitable for linear models.

```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cols = encoder.fit_transform(df[['person_home_ownership']])
encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(['person_home_ownership']))
df_final = pd.concat([df.drop(columns=['person_home_ownership']), encoded_df], axis=1)
```

## Exploratory Data Analysis (EDA) Insights

EDA helps understand the data distribution and relationships between variables impacting credit risk.

### Income Distribution by Loan Status

This visualization (generated in the notebook) highlights how income levels correlate with loan default status. Typically, lower income brackets show a higher density of defaults, providing key insights for the PD model.

![Income Distribution by Loan Status (Conceptual)](https://via.placeholder.com/400x200?text=Income+Distribution+by+Loan+Status)
*(Note: This is a placeholder image. The actual plot is generated in `6_Exploratory_Data_Analysis.ipynb`)*

## Model Development

We train baseline models for PD, LGD, and EAD using interpretable algorithms to meet regulatory requirements.

### PD Model Training (Logistic Regression)

The Probability of Default (PD) model uses Logistic Regression to predict the likelihood of a borrower defaulting.

```python
from sklearn.linear_model import LogisticRegression

pd_model = LogisticRegression(max_iter=1000)
pd_model.fit(X_train, y_train_pd)
```

## Expected Loss (EL) Estimation

Expected Loss is a critical metric for regulatory capital calculation, defined by the formula: `EL = PD × LGD × EAD`.

### EL Calculation Steps

```python
# PD: Probability of the '1' class (Default)
pd_probs = pd_model.predict_proba(X_test)[:, 1]

# LGD: (1 - Predicted Recovery Rate), clipped to [0, 1]
predicted_recovery = lgd_model.predict(X_test)
lgd_estimate = np.clip(1 - predicted_recovery, 0, 1)

# EAD: Predicted Exposure, clipped to non-negative values
ead_estimate = np.maximum(ead_model.predict(X_test), 0)

el_estimates = pd_probs * lgd_estimate * ead_estimate
```

### Expected Loss Distribution

The distribution of estimated Expected Loss values provides insights into the potential financial impact of defaults across the portfolio.

![Expected Loss Distribution (Conceptual)](https://via.placeholder.com/400x200?text=Expected+Loss+Distribution)
*(Note: This is a placeholder image. The actual plot is generated in the Integrated Basel Modeling & Expected Loss Analysis section)*


## Model Performance Summary

An overview of the model performance using key metrics like AUC-ROC for PD and RMSE for LGD/EAD.

![Model Performance Summary](model_performance_summary.png)

## Getting Started

1. Run the preprocessing notebooks in `DATA_PROCESSING/`.
2. Train the models using the training notebook in `MODELS/`.
3. Evaluate performance and interpret coefficients to ensure regulatory transparency.
