# Interpretable Time Series Forecasting using SHAP for Complex Regression Model

# Project Overview

  This project focuses on developing an interpretable high-frequency time-series forecasting model using LightGBM, combined with SHAP (Shapley Additive ExPlanations) to explain both global and local feature contributions.
The primary objective was not only to achieve strong forecasting accuracy but also to understand why the model behaves the way it does—particularly during critical forecasting windows such as sudden demand shifts or peak periods.

# Framework and Method

1. Data merging and cleaning (filling in missing values)
2. Data visualisation
3. Feature engineering (transforming categorical features)
4. Modelling and prediction
- Multivariate Time Series model
- LightGBM algorithm# Dataset Preparation & Feature Engineering

# Dataset Preparation & Feature Engineering

## Data Source

https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

## Data Understanding

**Store data:**
- store_nbr: the store at which the products are sold
- family: the type of product sold
- sales: the total sales for a product family at a particular store at a given date
- onpromotion: the total number of items in a product family that were being promoted at a store at a given date.
- cluster: a grouping of similar stores

**Holiday data:**
- type: holiday type
- locale: scope of the holiday

**Macro indicator:**
- oil price: Ecuador is an oil-dependent country and it's economical health is highly vulnerable to shocks in oil prices.

# Model Training - LightGBM

A gradient boosting model (LightGBM Regressor) was trained using:
- 80/20 time-based split for train/validation
- RMSE used as the primary baseline metric
- Early stopping for generalization

## Model Performance

- RMSE: 201.5717
- MAE: 55.2395

The error values reflect reasonable performance for the dataset’s scale, indicating the model learned both short-term and mid-term structure in the series.

# SHAP Analysis - Global Feature Importance

Two global interpretability views were compared:

## SHAP Mean Absolute Contributions

| Rank | Feature           | SHAP Mean Abs |
| ---- | ----------------- | ------------- |
| 1    | `rolling_7_mean`  | 242.4853      |
| 2    | `lag_1`           | 166.4345      |
| 3    | `lag_7`           | 104.2822      |
| 4    | `transactions`    | 41.5455       |
| 5    | `onpromotion`     | 38.3749       |
| 6    | `lag_14`          | 31.2030       |
| 7    | `rolling_14_mean` | 28.8118       |
| 8    | `day`             | 15.0077       |
| 9    | `cluster`         | 14.0537       |
| 10   | `lag_28`          | 13.0875       |

**Interpretation:**
- The model is heavily dependent on recent rolling averages and lags.
- External regressors (transactions, onpromotion) meaningfully shift predictions, but only after internal demand structure is captured.
- Calendar/cluster features help refine granularity but are not primary drivers

## LightGBM Native Feature Importance (Gain)

| Rank | Feature           | Gain      |
| ---- | ----------------- | --------- |
| 1    | `rolling_7_mean`  | 1.0666e13 |
| 2    | `lag_7`           | 8.1683e12 |
| 3    | `lag_1`           | 3.8510e12 |
| 4    | `lag_14`          | 1.5028e12 |
| 5    | `rolling_14_mean` | 5.5017e11 |
| 6    | `transactions`    | 5.4499e11 |
| 7    | `lag_28`          | 3.2900e11 |
| 8    | `day`             | 1.6710e11 |
| 9    | `dayofweek`       | 1.2527e11 |
| 10   | `onpromotion`     | 1.1989e11 |

**Key Takeaways:**
- Both SHAP and LightGBM agree on the top 3: rolling_7_mean, lag_7, lag_1.
- Gain gives higher emphasis to splitting efficiency, while SHAP captures per-prediction contribution magnitude.
- dayofweek ranks higher in model gain but does not appear in SHAP top 10—indicating frequent but small-impact splits.

## Local SHAP Analysis – High-Impact Forecast Windows

Ten forecast points identified as:
- Outliers
- Peaks
- Sudden shifts
- High-impact operational periods
were analyzed using SHAP force plots and decision paths.

## Insights from Local Explanations

Across the 10 critical timestamps:

**1.Sudden Peaks**
- Driven primarily by lag_1, rolling_7_mean, and onpromotion bursts.
- Promotions amplify already rising demand signals
**2.Unexpected drops**
- Strong negative SHAP values from lag_1 and transactions
- Suggest reduced customer movement or stock-outs.
**3.Holiday effects**
- Though not top globally, holiday-related features create large local SHAP spikes.
**4.Multi-feature interactions**
- (lag_7 × rolling_7_mean) interaction dominates most turning points.
- Calendar interactions (day × onpromotion) appear during weekday campaigns.
**5.Cluster-specific behavior**
- “cluster” sharply shifts predictions for two stores, showing location-specific pattern deviation.

# Global vs SHAP Comparison

| Feature        | SHAP Rank | Gain Rank | Explanation                                          |
| -------------- | --------- | --------- | ---------------------------------------------------- |
| `transactions` | 4         | 6         | Rare but high local influence during spikes.         |
| `onpromotion`  | 5         | 10        | Strong local impact but splits not always efficient. |
| `dayofweek`    | —         | 9         | Used often but has small marginal effect.            |

These discrepancies highlight why SHAP is needed—gain alone cannot explain local or interaction dynamics.

#Summary

This project successfully demonstrates how SHAP can provide transparent explanations for a complex gradient-boosted time-series forecasting model. The LightGBM model shows strong reliance on recent demand patterns, while SHAP reveals nuanced interactions and localized feature effects.

The combination of:
- Strong baseline performance,
- Robust global explanations, and
- Deep local interpretability
provides a defensible, operationally useful forecasting solution suitable for real-world deployment where justifying model behavior is essential.


