# Why I Stopped Optimizing AUC and Built a Timing Model Instead

**A Portfolio Project in Asking Better Questions**

---

## The Realization

Three weeks into this project, I had a working churn classifier. AUC of 0.66. Nothing spectacular, but serviceable.

Then I tried to actually *use* it.

"Customer #4,721 has a 78% churn probability."

Great. Now what? Call them tomorrow? Next week? Wait until they show more warning signs?

The model couldn't tell me. It was never designed to.

That's when I realized I'd spent three weeks answering the wrong question. Classification tells you WHO will churn. It doesn't tell you WHEN to do something about it.

This notebook is my attempt to fix that.

---

## What We're Building

A complete churn prevention system that answers four questions:

1. **WHO** is at risk? (Classification)
2. **WHEN** should we intervene? (Survival Analysis)
3. **WHAT** intervention works? (A/B Testing)
4. **IS IT WORTH IT?** (ROI Analysis)

No single model answers all of these. We need a system.

---

## Setup

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
```

---

## Part 1: The Classification Model (WHO)

Let's start with the standard approach - a churn classifier.

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

churn_model = train_churn_model(X_train, y_train)
y_pred_proba = churn_model.predict_proba(X_test)[:, 1]
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
```

```
AUC-ROC: 0.6612
```

### Why I Stopped Here

0.66 isn't impressive. I could probably push it to 0.72 or 0.75 with gradient boosting and hyperparameter tuning.

But I didn't.

Here's why: the model's job is to **rank** customers by risk. Can it reliably tell me that customer A is higher risk than customer B? At 0.66, yes. The ranking is meaningful, even if the exact probabilities aren't perfectly calibrated.

The bottleneck in this project isn't prediction accuracy. It's knowing what to *do* with the predictions.

So I moved on.

### Threshold Selection

```python
# Evaluating different thresholds
for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred = (y_pred_proba >= thresh).astype(int)
    # ... calculate precision, recall, F1
```

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| 0.3 | 23.2% | 96.0% | 0.374 |
| 0.4 | 24.9% | 84.9% | 0.386 |
| **0.5** | **29.3%** | **66.3%** | **0.406** |
| 0.6 | 34.5% | 35.3% | 0.349 |

I chose 0.5 - best F1, reasonable recall. Missing churners (false negatives) is expensive, so I wanted to keep recall decent.

![Threshold Analysis](viz/03_threshold_analysis.png)

---

## Part 2: The Survival Model (WHEN)

This is where it gets interesting.

Classification asks: "Will this customer churn?" (Binary outcome)
Survival analysis asks: "How long until this customer churns?" (Time-to-event)

```python
def train_survival_model(df, features):
    survival_df = df[features + ['duration', 'churned']].copy()
    survival_df = survival_df.rename(columns={'churned': 'event'})
    
    cph = CoxPHFitter()
    cph.fit(survival_df, duration_col='duration', event_col='event')
    return cph

survival_model = train_survival_model(customer_df, SURVIVAL_FEATURES)
print(f"Concordance Index: {survival_model.concordance_index_:.4f}")
```

```
Concordance Index: 0.6645
```

The concordance index is like AUC for survival models - it measures how well the model ranks event times. 0.66 means our model correctly orders 66% of customer pairs by their churn timing.

### What the Survival Model Reveals

![Survival Curves](viz/02_survival_curves.png)

Look at that curve. Churn isn't a sudden event - it's gradual:
- 8% by day 30
- 13% by day 60
- 21% by day 120

There's a *rhythm* to customer departure. A classification model treats Day 1 churners the same as Day 100 churners. Survival analysis captures the temporal pattern.

### Deriving the Intervention Window

Here's the key insight. I filtered to high-risk customers (probability >= 50%) and looked at their predicted days until churn:

```python
high_risk = df[df['churn_probability'] >= 0.5]
predicted_days = high_risk['predicted_days_until_churn']

print(f"25th percentile: {predicted_days.quantile(0.25):.0f}")
print(f"Median: {predicted_days.quantile(0.50):.0f}")
print(f"75th percentile: {predicted_days.quantile(0.75):.0f}")
```

```
25th percentile: 91
Median: 95
75th percentile: 97
```

The clustering is tight. Most high-risk customers are predicted to churn around day 91-97.

From this, I derived the intervention window:
- **Before Day 45**: Too early. Customer hasn't hit their frustration point.
- **Day 45-95**: Optimal. Customer is experiencing friction but hasn't decided.
- **After Day 95**: Too late. More than half have already churned.

![Intervention Window](viz/06_intervention_timing.png)

This window isn't a guess or an industry benchmark. It's derived directly from this dataset's survival predictions.

---

## Part 3: A/B Testing (WHAT WORKS)

At this point, I have:
- A classification model flagging 2,825 high-risk customers
- A survival model suggesting the optimal intervention window is Day 45-95

But a critical question remains: **Do interventions actually reduce churn?**

Predictions are hypotheses. Experiments are proof.

### Experiment Design

```python
class ABTestManager:
    def __init__(self, baseline_rate, alpha=0.05, seed=11):
        self.baseline_rate = baseline_rate
        self.alpha = alpha
        self.n_variants = 4  # excluding control
        self.bonferroni_alpha = alpha / self.n_variants  # 0.0125
        
    def run_chi_square_test(self, control_data, treatment_data):
        table = np.array([
            [treatment_data['churned'], treatment_data['n'] - treatment_data['churned']],
            [control_data['churned'], control_data['n'] - control_data['churned']]
        ])
        chi2, p_value, _, _ = stats.chi2_contingency(table)
        return p_value
```

**Why Bonferroni Correction?**

I'm testing 4 treatments against control. Without correction, I'd expect ~0.2 false positives by chance (4 × 0.05).

Bonferroni adjusts the significance threshold: 0.05 / 4 = 0.0125

A result is only "significant" if p < 0.0125, not 0.05.

### Results

![A/B Test Results](viz/08_ab_test_results.png)

| Variant | Churn Rate | Lift | p-value | Significant? |
|---------|------------|------|---------|--------------|
| Control | 21.7% | - | - | baseline |
| Email | 17.6% | +19.0% | 0.033 | No (p > 0.0125) |
| Discount | 15.6% | +28.2% | 0.001 | Yes |
| **Call** | **9.9%** | **+54.4%** | **<0.0001** | **Yes** |
| Combined | 15.0% | +30.8% | <0.001 | Yes |

**Call is the clear winner.** 54.4% reduction in churn, p < 0.0001.

Notice that Email's p-value (0.033) would be "significant" under naive alpha = 0.05, but fails under Bonferroni correction. This is exactly why proper statistical methodology matters.

---

## Part 4: ROI Analysis (IS IT WORTH IT)

Call has the best results. Ship it, right?

Not so fast.

```python
def calculate_roi(lift, cost, avg_clv=1931):
    value_saved = avg_clv * lift
    return value_saved / cost
```

| Channel | Lift | Cost | ROI |
|---------|------|------|-----|
| Email | 4.1pp | $0.50 | **158.8x** |
| Discount | 6.1pp | $10.00 | 11.8x |
| Call | 11.8pp | $35.00 | 6.5x |
| Combined | 6.7pp | $45.50 | 2.8x |

![Lift vs ROI](viz/10_lift_vs_roi.png)

Email's ROI is **24 times higher** than Call's.

This creates a strategic decision:
- **Maximize impact?** Use Call (54.4% lift)
- **Maximize efficiency?** Use Email (158.8x ROI)

The answer is neither - it's **tiered intervention**:
- High-value customers → Call (the impact justifies the cost)
- Standard customers → Email (efficiency at scale)

---

## The Actionability Gap

One more insight worth sharing.

My top predictive feature was `tenure_months` - how long the customer has been with us. Strong negative coefficient; newer customers churn more.

But you can't change someone's tenure. It's useful for prediction, useless for intervention.

I built a framework scoring features on both importance AND actionability:

```python
combined_score = abs(coefficient) * actionability_multiplier
# where actionability: High=3, Medium=2, Low=1
```

![Feature Importance vs Actionability](viz/04_feature_importance_comparison.png)

The rankings change completely. `engagement_score` becomes the #1 target because it combines decent predictive power with high actionability.

This is the kind of thinking that Kaggle competitions don't teach, but real-world ML requires.

---

## Summary

![Executive Summary](viz/Executive_summary.png)

**Model Metrics** (gates, not goals):
- Classification AUC: 0.6612
- Survival C-Index: 0.6645
- Threshold: 0.5 (F1: 0.406)

**Business Metrics** (what actually matters):
- CLV at Risk: $2,542,079
- Optimal Window: Day 45-95
- Best Lift: Call (+54.4%, p < 0.0001)
- Best ROI: Email (158.8x)
- Revenue Protected: ~$264K

**Key Lessons**:
1. AUC 0.66 was "good enough" - the bottleneck was elsewhere
2. Survival analysis reveals timing patterns invisible to classification
3. Bonferroni correction prevented a false positive (Email)
4. ROI analysis changed the optimal strategy
5. Actionability matters as much as predictive power

---

## Final Thought

I could have spent another month pushing AUC from 0.66 to 0.75.

Instead, I asked better questions:
- When should we act?
- What intervention works?
- Is it worth the cost?

The "mediocre" model, embedded in a thoughtful system, protected ~$264K in simulated revenue.

That's the difference between building models and solving problems.

---

**Upvote if this changed how you think about churn prediction.**

*Tags: survival-analysis, churn-prediction, ab-testing, statistical-methodology, portfolio-project*
