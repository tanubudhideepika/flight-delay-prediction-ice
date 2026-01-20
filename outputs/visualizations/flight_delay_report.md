---
title: "Flight Delay Prediction: Analysis, Modeling, and Insights"
author: "Deepika Tanubudhi"
date: "2026"
geometry: margin=1in
fontsize: 11pt
---

# 1. Data Exploration & Analysis

## 1.1 Dataset Overview

The dataset contains domestic U.S. flight records from 2017–2018, focusing on completed flights with valid arrival delay information. 

A binary target variable was defined using a 15-minute arrival delay threshold, consistent with industry standards used in airline performance reporting.

---

## 1.2 Delay Distribution and Class Imbalance

![Flight Delay Distribution](viz01_delay_distributions.png)

The delay distribution is highly imbalanced. Approximately **85–87% of flights arrive on time**, while only **13–15% are delayed** by 15 minutes or more.  

The arrival delay histogram shows a long right tail, with rare but severe delays extending several hours. This confirms that flight delays are infrequent but impactful events, motivating the use of precision–recall–based evaluation metrics during modeling.

---

## 1.3 Temporal Patterns

![Temporal Delay Patterns](viz02_temporal_analysis.png)

Delays are more frequent on **Mondays, Thursdays, and Fridays**, indicating higher congestion during business travel periods. Weekends, particularly Saturdays, show the lowest delay rates.

Monthly analysis reveals strong seasonality:
- Delay rates peak during **late spring and summer (May–July)**
- Lowest delay rates occur in **November**
- A secondary increase appears in December due to holiday travel

---

## 1.4 Time-of-Day Effects

![Delay by Hour](viz02_temporal_analysis.png)

Early morning departures (5–8 AM) have the lowest delay rates. Delay probability increases steadily throughout the day, reflecting accumulated upstream delays and airport congestion.

Late-night flights exhibit higher delay rates but lower flight volumes, suggesting reduced recovery capacity.

---

## 1.5 Day–Hour Interaction Effects

![Day vs Hour Heatmap](viz03_day_hour_heatmap.png)

The heatmap highlights compound effects:
- Weekday evenings experience the highest delays
- Early mornings remain consistently reliable across all days

These interactions justify the inclusion of time-of-day and weekday interaction features.

---

## 1.6 Airport and Geographic Patterns

![Geographic Delay Patterns](viz04_geographic_analysis.png)

Certain airports and states consistently exhibit higher delay rates independent of traffic volume. Large hub airports dominate flight volume but are not always the worst performers in delay percentage, suggesting operational efficiency differences.

---

## 1.7 Route-Level Analysis

![Route Analysis](viz05_route_analysis.png)

Delay rates vary significantly by route. Some high-volume routes perform better than average, while certain low-volume routes experience elevated delays. This motivated route-level historical risk features.

---

## 1.8 Carrier Performance

![Carrier Analysis](viz06_carrier_analysis.png)

Carriers show distinct delay profiles. High-volume carriers are not necessarily the worst performers, while smaller carriers display higher volatility. This supports the inclusion of carrier-level historical features.

---

## 1.9 Distance and Delay Relationship

![Distance Analysis](viz07_distance_analysis.png)

Distance alone shows weak linear correlation with delay probability. However, long-haul flights display greater variability and extreme delays, particularly when combined with seasonal effects.

---

## 1.10 Correlation and Anomaly Analysis

![Feature Correlation Matrix](viz08_correlation_matrix.png)

Most individual features exhibit weak linear correlation with the delay outcome, reinforcing the need for non-linear modeling approaches.

![Arrival Delay Outliers](viz09_anomaly_detection.png)

Extreme delays are rare but genuine operational events. These outliers were retained to preserve real-world behavior.

---

# 2. Machine Learning Model Development

## 2.1 Target Definition

Flights were labeled as delayed if arrival delay was **≥ 15 minutes**, aligning with standard aviation performance benchmarks.

---

## 2.2 Feature Engineering Strategy

Feature groups included:
- Temporal indicators (month, weekday, hour)
- Route and carrier identifiers
- Airport traffic and hub indicators
- Distance categories
- Train-only historical risk encodings
- Interaction features capturing compound effects

All target-derived features were computed **only on training data** to avoid leakage.

---

## 2.3 Models Implemented

Three models were developed:
- Logistic Regression (baseline, interpretable)
- XGBoost (non-linear baseline)
- CatBoost (native categorical handling)

A **time-based train/test split** was used to simulate real-world deployment.

---

# 3. Model Evaluation & Selection

## 3.1 Evaluation Metrics and Business Rationale

Flight delay prediction is a highly imbalanced classification problem, with delayed flights representing approximately 13–15% of observations. In this context, traditional accuracy metrics are misleading, as a naive model predicting all flights as on time would achieve high accuracy without providing value.

To address this, the following metrics were used:
- **Precision–Recall AUC (PR-AUC)** as the primary metric, as it better reflects model performance on the minority (delayed) class.data
- **ROC-AUC** as a secondary metric to evaluate overall ranking ability.
- **Brier score** to assess the quality and reliability of predicted probabilities, which is critical for user-facing risk estimates.

This combination allows evaluation of both ranking performance and probability calibration, aligning with the needs of a travel decision-support application.

---

## 3.2 Model Comparison

| Model | ROC-AUC | PR-AUC | Brier |
|------|--------|--------|-------|
| CatBoost | 0.693 | 0.299 | 0.186 |
| XGBoost | 0.679 | 0.290 | 0.180 |
| Logistic Regression | 0.672 | 0.279 | 0.109 |

CatBoost achieved the strongest PR-AUC and was selected as the final model.

---

## 3.3 Threshold Selection

A probability threshold of **0.30** was chosen to prioritize recall (~90%), ensuring that most delayed flights are flagged even at the cost of some false positives.

This choice reflects a deliberate trade-off:
- Higher recall reduces the likelihood of missing delayed flights.
- Lower precision increases false positives, which are acceptable in this context because the application provides risk awareness, not hard guarantees.

---

# 4. Model Interpretation & Insights

## 4.1 Key Factors Influencing Delays

Key contributors include:
- Historical delay behavior of routes, carriers, and airports
- Time-of-day congestion effects
- Seasonal travel demand
- Hub-to-hub routing patterns

Delays arise from interacting operational factors rather than a single dominant cause.

---

## 4.2 User-Facing Insights

The model enables actionable insights such as:
- Higher delay risk for late-afternoon departures
- Persistent reliability differences across routes
- Seasonal congestion warnings

---

## 4.3 Translation into App Features

These insights can be translated into:
- Delay risk scores during booking
- Proactive delay alerts
- Buffer time recommendations for connections
- Time-of-day travel suggestions

---

# 5. Limitations 

Limitations include the absence of weather, air traffic control, and aircraft rotation data. Future work could integrate real-time operational data to further improve predictive performance.

---

# Conclusion

This project demonstrates an end-to-end, approach to flight delay prediction, combining exploratory analysis, leakage-safe feature engineering, robust modeling, and interpretable insights suitable for in a travel application.
