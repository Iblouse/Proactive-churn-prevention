# Proactive Churn Prevention: A Complete ML Pipeline with Survival Analysis and A/B Testing

**Portfolio Project | Dual-Model Architecture | Production-Ready Statistical Framework**

---

## Overview

This notebook implements a complete churn prevention system that goes beyond binary classification to answer: **When should we intervene?**

![Executive Summary](viz/Executive_summary.png)

*Complete system overview: Risk scoring, timing prediction, A/B testing, and ROI optimization*

**Results at a Glance:**
| Metric | Value |
|--------|-------|
| Churn AUC | 0.6612 |
| Survival C-Index | 0.6645 |
| Best Intervention Lift | 54.4% (p < 0.0001) |
| Best ROI | 158.8x |
| Revenue Protected | ~$264K |

---

## 1. Setup and Configuration

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lifelines import CoxPHFitter
from scipy import stats

RANDOM_SEED = 42
EXPERIMENT_SEED = 11
OBSERVATION_WINDOW = 120  # days
```

**Dataset:** 6,000 synthetic customers | 21.0% churn rate | $1,931 mean CLV

---

## 2. The Problem: $2.54M at Risk

![Risk Distribution](viz/01_risk_distribution.png)

*Nearly 47% of customers (2,825) fall into High or Critical risk tiers, representing $2.54M in CLV exposure*

---

## 3. Model 1: Churn Classification

```python
def train_churn_model(X_train, y_train):
    model = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)
    return model
```

**Results:**
- AUC-ROC: **0.6612**
- Best Threshold: 0.5
- Recall: 66.3% | Precision: 29.3% | F1: 0.406

![Threshold Analysis](viz/03_threshold_analysis.png)

*Precision-recall trade-off across thresholds. Selected 0.5 for simplicity.*

**Note:** I intentionally used default settings. The goal is ranking customers by risk, not maximizing AUC.

---

## 4. Model 2: Survival Analysis

```python
def train_survival_model(df, features):
    survival_df = df[features + ['duration', 'churned']].copy()
    survival_df = survival_df.rename(columns={'churned': 'event'})
    
    cph = CoxPHFitter()
    cph.fit(survival_df, duration_col='duration', event_col='event')
    return cph
```

**Results:**
- Concordance Index: **0.6645**

![Survival Curves](viz/02_survival_curves.png)

*Churn occurs gradually: 8% by day 30, 13% by day 60, 21% by day 120*

**Top Hazard Ratios:**
| Feature | Hazard Ratio |
|---------|-------------|
| support_tickets_90d | 1.208 |
| is_inactive | 1.117 |
| has_payment_issues | 1.077 |

---

## 5. The Actionability Gap

**Key insight:** The best predictors aren't always the best intervention targets.

![Feature Importance vs Actionability](viz/04_feature_importance_comparison.png)

*Traditional importance (left) vs. combined scoring with actionability (right)*

`tenure_months` is our strongest predictor but we can't change it. `engagement_score` becomes #1 target when actionability is factored in.

![Priority Matrix](viz/05_four_quadrant_matrix.png)

*Four-quadrant matrix: Focus on Priority Targets (high importance + high actionability)*

---

## 6. Intervention Window Derivation

```python
def derive_intervention_window(df, risk_threshold=0.5):
    high_risk = df[df['churn_probability'] >= risk_threshold]
    predicted_days = high_risk['predicted_days_until_churn']
    
    return {
        'optimal_start': int(predicted_days.quantile(0.50) * 0.5),  # Day 45
        'optimal_peak': int(predicted_days.quantile(0.25)),         # Day 93
        'optimal_end': int(predicted_days.quantile(0.50)),          # Day 95
    }
```

![Intervention Window](viz/06_intervention_timing.png)

*Optimal window (Day 45-95) derived from survival model predictions*

![Revenue Impact](viz/07_revenue_impact.png)

*Revenue impact by timing window - optimal timing captures majority of potential savings*

---

## 7. A/B Testing Framework

```python
class ABTestManager:
    def __init__(self, baseline_rate, significance_level=0.05):
        self.baseline_rate = baseline_rate
        self.alpha = significance_level
        
    def calculate_significance(self, results, control='control'):
        n_comparisons = len(results) - 1
        bonferroni_alpha = self.alpha / n_comparisons  # 0.0125
        
        # Chi-square test for each variant vs control
        # Returns p-values and significance flags
```

**Configuration:**
- 5 variants x 900 customers = 4,500 total
- Bonferroni-adjusted alpha: **0.0125**

![A/B Test Results](viz/08_ab_test_results.png)

*Call achieves 54.4% churn reduction (p < 0.0001)*

| Variant | Churn Rate | Lift | p-value | Significant? |
|---------|------------|------|---------|--------------|
| Control | 21.7% | - | - | baseline |
| Email | 17.6% | +19.0% | 0.033 | No |
| Discount | 15.6% | +28.2% | 0.001 | Yes |
| **Call** | **9.9%** | **+54.4%** | **<0.0001** | **Yes** |
| Combined | 15.0% | +30.8% | <0.001 | Yes |

---

## 8. ROI Analysis

```python
def calculate_roi(absolute_reduction, cost, avg_clv=1931):
    return (avg_clv * absolute_reduction) / cost
```

![Lift vs ROI](viz/10_lift_vs_roi.png)

*Email has 158.8x ROI despite lower lift - optimal strategy is tiered*

| Channel | Absolute Diff | Cost | ROI |
|---------|--------------|------|-----|
| **Email** | 4.1pp | $0.50 | **158.8x** |
| Discount | 6.1pp | $10.00 | 11.8x |
| Call | 11.8pp | $35.00 | 6.5x |
| Combined | 6.7pp | $45.50 | 2.8x |

**Strategic insight:** Email for volume, Call for VIPs.

---

## 9. Summary

### Model Performance
| Model | Metric | Value |
|-------|--------|-------|
| Classification | AUC-ROC | 0.6612 |
| Survival | C-Index | 0.6645 |

### Business Outcomes
| Metric | Value |
|--------|-------|
| Optimal Window | Day 45-95 |
| Best Lift | Call (+54.4%) |
| Best ROI | Email (158.8x) |
| Revenue Protected | ~$264K |

### Key Takeaways
1. Survival analysis adds timing information classification cannot provide
2. Bonferroni correction essential for multiple comparisons
3. ROI tells a different story than lift
4. AUC 0.66 is sufficient when goal is ranking

---

## Technical Capabilities Demonstrated

| Capability | Implementation |
|------------|----------------|
| Risk Scoring | Logistic Regression |
| Timing Prediction | Cox Proportional Hazards |
| Experiment Design | A/B testing, Bonferroni |
| Agent Orchestration | Google ADK |
| ROI Analysis | Cost modeling |

---

**Upvote if you found this useful!**

*Tags: survival-analysis, churn-prediction, ab-testing, portfolio-project*
