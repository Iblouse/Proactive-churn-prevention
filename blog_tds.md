# Beyond Churn Prediction: Building a Complete Retention System with Survival Analysis and A/B Testing

*A portfolio project demonstrating how to answer "when to intervene" - not just "who will churn"*

---

## Introduction

Churn prediction is one of the most common machine learning applications. But here's what most tutorials miss: **predicting churn is only half the problem.**

For this portfolio project, I built a complete system that answers the operational question that actually matters: **When is the optimal time to intervene?**

![Executive Summary](viz/Executive_summary.png)

*The complete system: from risk identification to validated intervention strategies*

**What you'll learn:**
1. Why timing matters more than prediction accuracy
2. Adding survival analysis to predict *when* (not just *if*)
3. Deriving data-driven intervention windows
4. Validating interventions with proper A/B testing
5. Calculating ROI to optimize channel selection

---

## Part 1: The $2.54 Million Problem

I created a synthetic dataset of 6,000 customers to demonstrate this methodology.

![Risk Distribution](viz/01_risk_distribution.png)

*Nearly 47% of customers (2,825) fall into High or Critical risk tiers*

| Metric | Value |
|--------|-------|
| Total Customers | 6,000 |
| Churn Rate | 21.0% |
| High-Risk Customers | 2,825 (47.1%) |
| Mean CLV | $1,931 |
| **Total CLV at Risk** | **$2,542,079** |

---

## Part 2: Why Timing Is Everything

Consider a customer with 80% churn probability. Your model correctly identifies them as high-risk. Now what?

| Timing | Outcome |
|--------|---------|
| Too Early (Day 20) | Customer hasn't experienced friction - outreach feels random |
| **Optimal (Day 70)** | Customer experiencing issues but hasn't decided to leave |
| Too Late (Day 100) | Customer has mentally checked out - too late |

**The key insight:** Timing isn't a nice-to-have. It's fundamental to whether your intervention works.

---

## Part 3: The Two-Model Architecture

### Model 1: Churn Classification (Who?)

**Approach:** Logistic Regression with L2 regularization

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.6612 |
| Best Threshold | 0.5 |
| Recall @ 0.5 | 66.3% |
| Precision | 29.3% |
| F1 | 0.406 |

![Threshold Analysis](viz/03_threshold_analysis.png)

*Precision-recall trade-off across thresholds*

**Note:** I intentionally used default settings. The model's job is to *rank* customers by risk, and 0.66 is sufficient for that.

### Model 2: Survival Analysis (When?)

**Approach:** Cox Proportional Hazards

Survival analysis predicts:
- Probability of "surviving" to any time point
- Expected time until the event
- Which features accelerate or delay the event

![Survival Curves](viz/02_survival_curves.png)

*Churn occurs gradually: 8% by day 30, 13% by day 60, 21% by day 120*

| Metric | Value |
|--------|-------|
| Concordance Index | 0.6645 |

**Top Hazard Ratios:**
| Feature | Hazard Ratio | Interpretation |
|---------|-------------|----------------|
| support_tickets_90d | 1.208 | 20.8% higher hazard |
| is_inactive | 1.117 | 11.7% higher hazard |
| has_payment_issues | 1.077 | 7.7% higher hazard |

---

## Part 4: The Actionability Gap

**Critical insight:** The best predictors aren't always the best intervention targets.

![Feature Importance vs Actionability](viz/04_feature_importance_comparison.png)

*Traditional importance (left) vs. combined scoring with actionability (right)*

Our strongest predictor is `tenure_months` (coefficient = -0.50), but we can't change how long someone has been a customer.

Features like `engagement_score` have smaller coefficients but **high actionability** - we can influence them through product tours and feature education.

**Combined Scoring Formula:**
```
Combined Score = |Coefficient| x Actionability Multiplier
```
Where: High = 3.0, Medium = 2.0, Low = 1.0

![Priority Matrix](viz/05_four_quadrant_matrix.png)

*Focus on Priority Targets: high importance + high actionability*

---

## Part 5: Deriving the Optimal Intervention Window

Rather than hardcoding timing, I calculated windows from survival model predictions for high-risk customers (n=2,825):

| Percentile | Predicted Days |
|------------|----------------|
| 25th | 91 |
| Median | 95 |
| 75th | 97 |

**Derived Window:**
| Zone | Day Range | Rationale |
|------|-----------|-----------|
| Too Early | 0-45 | Customer hasn't experienced friction |
| **Optimal** | **45-95** | Customer receptive, hasn't decided to leave |
| Peak | ~93 | Maximum intervention leverage |
| Too Late | 95+ | More than half have already churned |

![Intervention Window](viz/06_intervention_timing.png)

*The optimal window derived from survival model predictions - not assumptions*

![Revenue Impact](viz/07_revenue_impact.png)

*Revenue impact by timing - optimal window captures majority of potential savings*

---

## Part 6: A/B Testing (Validating That Interventions Work)

Model predictions are hypotheses. Only experiments can validate them.

### Experiment Design

| Variant | Description | Cost |
|---------|-------------|------|
| Control | No intervention | $0 |
| Email | Automated re-engagement | $0.50 |
| Discount | 10% off offer | $10.00 |
| Call | Personal phone outreach | $35.00 |
| Combined | Multi-channel | $45.50 |

**Configuration:**
- Sample size: 900 per variant (4,500 total)
- Significance level: alpha = 0.05
- **Bonferroni correction:** adjusted alpha = 0.0125

### Why Bonferroni Correction?

With 4 treatment variants, we're running 4 hypothesis tests. Without correction, we'd expect false positives by chance.

```
Adjusted alpha = 0.05 / 4 = 0.0125
```

### Results

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

## Part 7: The ROI Analysis

Call has the highest lift. But ROI tells a different story:

![Lift vs ROI](viz/10_lift_vs_roi.png)

*Email has 158.8x ROI despite lower lift - the optimal strategy is tiered*

**ROI Formula:**
```
ROI = (Average CLV x Absolute Reduction) / Cost per Customer
```

| Channel | Absolute Diff | Cost | ROI |
|---------|--------------|------|-----|
| **Email** | 4.1pp | $0.50 | **158.8x** |
| Discount | 6.1pp | $10.00 | 11.8x |
| Call | 11.8pp | $35.00 | 6.5x |
| Combined | 6.7pp | $45.50 | 2.8x |

**Strategic Insight:** This isn't either/or - it's a tiered strategy:
- **Email** for broad outreach (best ROI)
- **Call** for high-value customers (best lift)

---

## Results Summary

### Model Metrics (Gating Checks)

| Metric | Value | Status |
|--------|-------|--------|
| Churn AUC | 0.6612 | Adequate for ranking |
| Survival C-Index | 0.6645 | Adequate for timing |
| Optimal Threshold | 0.5 (F1: 0.406) | Balanced trade-off |

### Business Metrics (What Actually Matters)

| Metric | Value |
|--------|-------|
| CLV at Risk | $2,542,079 |
| Optimal Window | Day 45-95 |
| Best Lift | Call (+54.4%) |
| Best ROI | Email (158.8x) |
| Customers Saved | ~137 |
| Revenue Protected | ~$264K |

---

## Key Takeaways

1. **Timing comes from data, not assumptions.** The Day 45-95 window was derived from survival model predictions.

2. **Always correct for multiple testing.** Bonferroni adjustment prevents false positives.

3. **ROI trumps lift for resource allocation.** Call has 3x the lift but Email has 24x better ROI.

4. **Model metrics are gating checks.** AUC 0.66 is sufficient - business metrics matter.

5. **Experiment before you operationalize.** Validate interventions before building production systems.

---

## Technical Capabilities Demonstrated

| Capability | Implementation | Outcome |
|------------|----------------|---------|
| Risk Scoring | Logistic Regression | Prioritized customer lists |
| Timing Prediction | Cox Proportional Hazards | Optimal intervention windows |
| Experiment Design | A/B testing, Bonferroni | Validated channel effectiveness |
| Agent Orchestration | Google ADK | Automated routing at scale |
| ROI Analysis | Cost modeling | Resource allocation guidance |

---

## Conclusion

This project demonstrates that adequate models (AUC 0.66) combined with rigorous methodology (survival analysis + A/B testing) can deliver measurable business outcomes.

The model's AUC is 0.66. That's fine. What matters is:
- The optimal window is **derived from data**
- The winning channel is **validated by experiment**
- The ROI is **calculated from test data**
- The system is **usable by cross-functional teams**

This is what applied ML looks like: adequate models, rigorous experimentation, measurable outcomes.

---

*The complete implementation, including all code and visualizations, is available in the accompanying Jupyter notebook.*
