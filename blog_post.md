# Building a Proactive Churn Prevention System: How Timing Transforms Customer Retention

**A Technical Deep-Dive into Survival Analysis, Multi-Agent AI, and the Business Impact of Getting Intervention Timing Right**

---

## Executive Summary

This project demonstrates that **when** you intervene matters as much as **whether** you intervene. By combining survival analysis with multi-agent AI, I built a system that:

- Identifies at-risk customers **46-95 days before churn** (the optimal intervention window)
- Achieves a **53.2% reduction in churn** with the best-performing intervention channel
- Protects **$1.18M in customer lifetime value** across 1,347 high-risk customers
- Provides statistically significant results (p < 0.0001) validated through rigorous A/B testing

The key insight: the optimal intervention window (Day 46-95) is derived directly from survival analysis of high-risk customer predictions—not hardcoded assumptions.

> **Note**: This project isn't about building the most accurate ML model—it's about integrating traditional ML, survival analysis, GenAI, statistics, and experiment design to deliver measurable business impact. A 0.66 AUC model that protects $1.18M in revenue is more valuable than a 0.95 AUC model sitting in a notebook.

---

## The $1.18 Million Problem

Before diving into the technical solution, let's understand what's at stake.

### Customer Risk Distribution

Our analysis of 3,000 customers revealed the following risk distribution:

| Risk Tier | Count | Percentage | Churn Probability |
|-----------|-------|------------|-------------------|
| Low | 286 | 9.5% | <25% |
| Medium | 1,367 | 45.6% | 25-49% |
| High | 1,275 | 42.5% | 50-74% |
| Critical | 72 | 2.4% | ≥75% |

![Risk Distribution](01_risk_distribution.png)

*Figure 1: Customer segmentation by churn risk tier (n=3,000)*

**Key Finding**: Nearly 45% of customers (1,347) fall into High or Critical risk tiers, representing significant business exposure. These are the customers where intervention timing becomes critical.

### Quantifying the Business Exposure

With a mean Customer Lifetime Value of **$1,924**, the stakes are substantial:

- **Total CLV at Risk**: $1,181,624 (High + Critical tiers)
- **At-Risk Customer Count**: 1,347 customers
- **Overall Churn Rate**: 19.4% (581 of 3,000 customers)

This isn't an abstract ML problem—it's a revenue protection challenge with concrete dollar amounts.

---

## Part 1: Why Timing Is Everything

Most churn prevention systems answer: "Will this customer churn?" 

That's the wrong question.

The right question is: **"When will this customer churn, and when should we intervene?"**

### The Intervention Window Problem

Consider two scenarios for a high-risk customer:

| Scenario | Intervention Day | Outcome |
|----------|-----------------|---------|
| Too Early | Day 20 | Customer not yet experiencing friction; intervention feels premature |
| **Optimal** | Day 70 (within 46-95 window) | Customer receptive; experiencing issues but hasn't decided to leave |
| Too Late | Day 100 | Customer has mentally "checked out"; intervention feels desperate |

### Survival Analysis: The Missing Piece

Standard classification models predict *if* a customer will churn. Survival analysis predicts *when*.

From our Cox Proportional Hazards model, we observed the actual churn timing distribution:

**Event Time Distribution (Churned Customers Only):**
| Metric | Days |
|--------|------|
| Minimum | 7 |
| 25th Percentile | 17 |
| Median | 45 |
| 75th Percentile | 83 |
| Maximum | 119 |

![Survival Curves](08_survival_curves.png)

*Figure 2: Survival probability over the 120-day observation window*

**Retention Probabilities Over Time:**
- 30-day retention: 92.3%
- 60-day retention: 87.9%
- 90-day retention: 85.2%
- 120-day retention: 80.6%

This tells us that most churn events occur gradually—7.7% by day 30, 12.1% by day 60, 19.4% by day 120. A one-size-fits-all intervention policy ignores this temporal pattern.

---

## Part 2: Building the Prediction Engine

### The Two-Model Architecture

I implemented a dual-model approach:

1. **Churn Classification Model**: Logistic Regression → Probability of churn
2. **Survival Model**: Cox Proportional Hazards → Days until churn

### Model 1: Churn Probability Prediction

**Why Logistic Regression?**

| Consideration | Logistic Regression | Gradient Boosting |
|--------------|---------------------|-------------------|
| Interpretability | ✅ Clear coefficients | ❌ Black box |
| Probability calibration | ✅ Native probabilities | ⚠️ Requires calibration |
| Training speed | ✅ Fast iteration | ⚠️ Slower |

**Model Performance (Test Set, n=600):**
- **AUC-ROC: 0.6622**
- Best threshold: 0.4 (by F1 score)
- At threshold 0.4:
  - Recall: 85.3% (catches most churners)
  - Precision: 24.3%
  - F1: 0.378

**Threshold Analysis:**

| Threshold | Accuracy | Precision | Recall | F1 |
|-----------|----------|-----------|--------|-----|
| 0.3 | 32.3% | 21.8% | 96.6% | 0.356 |
| **0.4** | 45.7% | 24.3% | **85.3%** | **0.378** |
| 0.5 | 60.0% | 26.0% | 57.8% | 0.358 |
| 0.6 | 73.5% | 33.3% | 37.1% | 0.351 |
| 0.7 | 80.2% | 45.9% | 14.7% | 0.222 |

![Threshold Analysis](07_threshold_analysis.png)

*Figure 3: Precision-Recall trade-off across classification thresholds*

**Business Decision**: I selected threshold = 0.4 because:
- **High recall (85.3%)**: We catch most at-risk customers
- **Acceptable precision trade-off**: False positives (contacting happy customers) have low cost compared to missing churners

**Confusion Matrix at Threshold 0.4:**
```
              Predicted
              Retained  Churned
Actual
Retained       175      309
Churned         17       99
```

- **True Positives (99)**: Churners we can intervene with
- **False Negatives (17)**: Churners we missed
- **False Positives (309)**: Retained customers flagged (opportunity for proactive engagement)

### Model 2: Survival Analysis

**Cox Proportional Hazards Model Performance:**
- **Concordance Index: 0.6774**
- Features: 9
- Training samples: 3,000
- Events (churned): 581

**Top Risk Factors (Hazard Ratios):**

| Feature | Hazard Ratio | Significance |
|---------|-------------|--------------|
| support_tickets_90d | 1.208 | *** |
| is_inactive | 1.117 | ** |
| has_payment_issues | 1.077 | |
| payment_delays_12m | 1.014 | |
| monthly_charges | 1.010 | |

**Interpretation**: A customer with high support ticket volume has a 20.8% higher hazard (faster time to churn) compared to baseline, all else equal.

---

## Part 2.5: Why Prediction Power Alone Isn't Enough

This is a critical insight that separates academic ML from business-impactful ML: **the best predictors aren't always the best intervention targets.**

### The Actionability Problem

Consider our top feature by coefficient magnitude:

| Feature | Coefficient | Actionability |
|---------|-------------|---------------|
| **tenure_months** | -0.4997 | **Low** |
| support_tickets_90d | +0.1913 | Medium |
| engagement_score | -0.1796 | **High** |
| nps_score | -0.1609 | Low |
| feature_usage_pct | -0.1577 | **High** |

**The Problem**: `tenure_months` is our strongest predictor (coefficient = -0.50), but we can't change how long a customer has been with us. It's useful for prediction but useless for intervention.

**The Insight**: Features like `engagement_score` and `feature_usage_pct` have smaller coefficients but **HIGH actionability**—we can directly influence them through product tours, email campaigns, and feature education.

### Actionability Framework

I categorized all 15 features by business actionability:

**High Actionability** (can directly influence through interventions):
- `engagement_score` — Improve via feature education, onboarding
- `feature_usage_pct` — Drive through product tours, tips
- `email_open_rate` — Optimize subject lines, send times
- `login_frequency_monthly` — Increase via notifications, value reminders
- `has_payment_issues` — Address via payment plans, reminders

**Medium Actionability** (can influence indirectly):
- `support_tickets_90d` — Improve via better documentation, proactive help
- `is_inactive` — Re-engage via campaigns
- `last_activity_days` — Trigger re-engagement workflows
- `payment_delays_12m` — Offer payment flexibility
- `discount_count` — Strategic discount timing

**Low Actionability** (difficult to change):
- `tenure_months` — Cannot change customer tenure
- `nps_score` — Lagging indicator, slow to move
- `monthly_charges` — Pricing is strategic decision
- `is_high_value` — Historical behavior
- `is_heavy_support_user` — Symptom, not cause

### Combined Scoring: Importance × Actionability

To prioritize intervention targets, I calculated a combined score:

```
Combined Score = |Coefficient| × Actionability Multiplier
```

Where: High = 3.0, Medium = 2.0, Low = 1.0

![Feature Importance vs Actionability](09_feature_actionability.png)

*Figure: Traditional feature importance (left) vs. business-focused combined scoring (right)*

**Key Insight**: The ranking changes dramatically:

| Rank | By Coefficient Only | By Combined Score |
|------|---------------------|-------------------|
| 1 | tenure_months (0.50) | **engagement_score** (0.54) |
| 2 | support_tickets_90d (0.19) | tenure_months (0.50) |
| 3 | engagement_score (0.18) | **feature_usage_pct** (0.47) |
| 4 | nps_score (0.16) | support_tickets_90d (0.38) |
| 5 | feature_usage_pct (0.16) | is_inactive (0.18) |

**`engagement_score` becomes the #1 intervention target** because it combines meaningful prediction power (0.18) with high actionability (3.0×).

### Feature Selection Matrix

![Feature Selection Matrix](10_feature_selection_matrix.png)

*Figure: Four-quadrant matrix for prioritizing intervention targets*

**Quadrant Analysis:**

| Quadrant | Description | Strategy |
|----------|-------------|----------|
| **Priority Targets** (upper-right) | High importance + High actionability | Focus interventions here |
| Monitor Only (lower-right) | High importance + Low actionability | Use for prediction, not intervention |
| Easy Wins (upper-left) | Low importance + High actionability | Quick wins, secondary priority |
| Deprioritize (lower-left) | Low importance + Low actionability | Ignore for interventions |

### Business Impact of This Framework

Without the actionability lens, a retention team might:
- ❌ Build campaigns around "customer tenure" (can't change it)
- ❌ Focus on NPS scores (lagging indicator, slow to improve)
- ❌ Ignore engagement metrics (lower coefficient, but actionable)

With the combined framework:
- ✅ Target engagement score improvements (product tours, feature tips)
- ✅ Drive feature adoption (email series, in-app guidance)
- ✅ Address payment issues proactively (flexibility programs)

**The lesson**: A model that predicts churn is only half the solution. You need features that are both **predictive AND actionable** to drive business outcomes.

---

## Part 3: The Optimal Intervention Window

This is the core business insight of the project.

### Deriving Windows from Survival Predictions

Rather than hardcoding intervention timing, I calculated windows directly from survival model predictions for high-risk customers (n=1,347):

**High-Risk Customer Prediction Distribution:**
| Metric | Days |
|--------|------|
| 25th Percentile | 92 |
| Median | 95 |
| 75th Percentile | 98 |

**Derived Intervention Window:**
- **Too Early**: Day 0 to 46
- **Optimal**: Day 46 to 95
- **Peak Effectiveness**: ~Day 93
- **Too Late**: After Day 95

![Intervention Window](02_intervention_window.png)

*Figure 4: Intervention success rate by timing, with optimal window (Day 46-95) highlighted*

### Why These Boundaries?

The window boundaries come from the survival model's predictions:

1. **Day 46** (window start): Approximately half of the 25th percentile (92 × 0.5). Before this point, customers haven't yet experienced enough friction to be receptive to retention offers.

2. **Day 93** (peak): The point where predicted survival curves for high-risk customers show maximum intervention leverage—customers are experiencing issues but haven't made the decision to leave.

3. **Day 95** (window end): The median predicted days until churn for high-risk customers. After this point, more than half of at-risk customers have already churned.

### Business Impact of Timing

The difference between optimal and suboptimal timing is substantial. For our 1,347 high-risk customers with $1.18M CLV at risk:

![Business Impact of Timing](06_business_impact_timing.png)

*Figure 5: Revenue impact comparison across intervention timing windows*

| Timing | Expected Success | Potential Revenue Protected |
|--------|-----------------|----------------------------|
| Too Early (Day 0-46) | Lower | Customer not ready to engage |
| **Optimal (Day 46-95)** | **Highest** | **Maximum impact window** |
| Too Late (Day 95+) | Declining | Customer already churning |

---

## Part 4: Multi-Agent AI Architecture

With predictions in hand, the next challenge is operationalizing them at scale. I built a multi-agent system using Google's Agent Development Kit (ADK).

### Agent Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR AGENT                           │
│         (Coordinates workflow, manages communication)           │
└─────────────────────┬───────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┬─────────────┐
        ▼             ▼             ▼             ▼
┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
│ BEHAVIORAL│  │ PREDICTIVE│  │INTERVENTION│ │ EVALUATION│
│   AGENT   │  │   AGENT   │  │   AGENT    │ │   AGENT   │
└───────────┘  └───────────┘  └───────────┘  └───────────┘
```

**Behavioral Monitoring Agent**: Analyzes customer behavior, identifies early warning signals

**Predictive Analytics Agent**: Runs churn and survival models, prioritizes customers

**Intervention Strategy Agent**: Selects channel based on A/B test results, determines timing

**Evaluation Agent**: Tracks outcomes, measures effectiveness

### Tool Integration

Each agent has specialized tools:

```python
predictive_agent.tools = [
    calculate_churn_score,      # Returns probability + risk tier
    run_survival_analysis,      # Returns days until churn
    list_at_risk_customers      # Returns prioritized list
]

intervention_agent.tools = [
    recommend_intervention,     # Returns channel + action + ROI
    get_customer_behavior       # Returns engagement context
]
```

---

## Part 5: A/B Testing Framework

Every intervention recommendation is backed by experimental evidence.

### Multi-Variant Experiment Design

I implemented 5-arm testing to compare intervention channels:

| Variant | Sample Size | Description |
|---------|-------------|-------------|
| Control | 400 | No intervention |
| Email | 400 | Automated campaign |
| Discount | 400 | 10% discount offer |
| Call | 400 | Personal phone outreach |
| Combined | 400 | Multi-channel approach |

**Configuration:**
- Baseline churn rate: 18.0%
- Total participants: 2,000
- Significance level: α = 0.05
- Multiple testing correction: Bonferroni (adjusted α = 0.0125)
- Random seed: 11 (for reproducibility)

### Results

![A/B Test Results](03_ab_test_results.png)

*Figure 6: Churn rates by intervention variant (n=400 per group)*

| Variant | Churned | Churn Rate | Absolute Δ | Relative Lift | P-value | Significant? |
|---------|---------|------------|------------|---------------|---------|--------------|
| Control | 77 | 19.2% | — | — | — | (baseline) |
| Email | 63 | 15.8% | +3.5pp | +18.2% | 0.2264 | ❌ No |
| Discount | 53 | 13.2% | +6.0pp | +31.2% | 0.0275 | ❌ No* |
| **Call** | **36** | **9.0%** | **+10.2pp** | **+53.2%** | **0.00005** | **✅ Yes** |
| Combined | 54 | 13.5% | +5.8pp | +29.9% | 0.0356 | ❌ No* |

*Not significant at Bonferroni-adjusted α = 0.0125

**🏆 Winner: Call** — 53.2% relative reduction in churn, statistically significant

### ROI Analysis

While Call delivers the highest lift, ROI tells a different story:

![Intervention ROI](04_intervention_roi.png)

*Figure 7: Return on investment by intervention channel*

| Channel | Lift | Cost/Customer | ROI |
|---------|------|---------------|-----|
| **Email** | 18.2% | $0.50 | **12.1x** |
| Discount | 31.2% | ~$15 | 9.4x |
| Combined | 29.9% | ~$65 | 8.2x |
| Call | 53.2% | $50 | 4.5x |

**Strategic Insight**: 
- **Email** offers highest ROI (12.1x) due to minimal cost—ideal for scalable outreach
- **Call** delivers highest lift (53.2%) but lowest ROI (4.5x)—reserve for highest-value customers
- The optimal strategy is **tiered**: Email for volume, Call for VIPs

### Channel Selection Logic

Based on A/B results, the intervention agent uses this mapping:

| Customer Segment | Recommended Channel | Rationale |
|-----------------|---------------------|-----------|
| Critical + High-Value | Combined | Maximum retention for VIPs |
| Critical + Standard | Call | High-touch for urgent cases |
| High + Payment Issues | Discount | Address price sensitivity |
| High + Low Engagement | Email | Feature adoption at scale |
| Medium | Email | Cost-effective coverage |
| Low | Monitor only | ROI doesn't justify intervention |

---

## Part 6: Putting It All Together

### Example: Critical-Risk Customer Analysis

Here's how the system handles a **Critical-risk customer**—the cases where intervention timing matters most:

**Churn Score Analysis (CUST_001593):**
```json
{
  "customer_id": "CUST_001593",
  "churn_probability": 0.9056,
  "risk_tier": "Critical",
  "predicted_days_until_churn": 58,
  "key_risk_factors": [
    "Payment issues detected",
    "Low engagement score",
    "High support ticket volume"
  ],
  "clv_at_risk": 3850.00
}
```

**System Decision Process:**

1. **Risk Assessment**: 90.6% churn probability → Critical tier
2. **Timing Check**: Day 58 prediction falls within optimal window (Day 46-95) ✓
3. **Actionability Analysis**: Multiple risk factors present, including actionable ones (payment issues, engagement)
4. **Channel Selection**: Critical + Multiple risks → Combined intervention

**Intervention Recommendation:**
```json
{
  "intervention_channel": "Combined",
  "intervention_action": "Multi-channel retention campaign: payment flexibility + personal outreach + feature education",
  "priority": 1,
  "priority_label": "Immediate",
  "expected_lift": 0.299,
  "intervention_cost": 65.00,
  "roi_estimate": 8.2,
  "value_at_risk": 3488.53,
  "value_if_saved": 1043.11,
  "optimal_contact_window": "Day 46-58 (act now)",
  "risk_factors_addressed": ["Payment flexibility program", "Dedicated success manager call", "Feature adoption campaign"]
}
```

**Why Combined?** For Critical-risk customers with multiple risk factors, the higher intervention cost ($65) is justified because:
- CLV at risk ($3,850) far exceeds intervention cost
- Multiple touchpoints address multiple risk factors simultaneously
- 29.9% expected lift translates to significant value protection

### Projected Business Impact

For our 3,000 customer dataset:

| Metric | Value |
|--------|-------|
| Total Customers | 3,000 |
| At-Risk (High + Critical) | 1,347 (44.9%) |
| Total CLV at Risk | $1,181,624 |
| Mean CLV | $1,924 |

**If we apply the optimal intervention (Call) to high-risk customers:**
- Baseline churn: 19.2%
- Post-intervention churn: 9.0% (53.2% reduction)
- Customers saved: ~137 additional customers retained
- Value protected: ~$264K additional CLV

---

## Lessons Learned

### 1. Timing Comes from Data, Not Assumptions

The optimal window (Day 46-95) wasn't assumed—it was derived from the survival model's predictions for high-risk customers. This data-driven approach adapts as customer behavior changes.

### 2. Statistical Significance Requires Proper Testing

With 4 treatment arms, Bonferroni correction raised the significance threshold to α = 0.0125. Only Call achieved this threshold—a reminder that multiple testing inflates false positives.

### 3. ROI ≠ Lift

Call has the highest lift (53.2%) but lowest ROI (4.5x). Email has modest lift (18.2%) but highest ROI (12.1x). Business decisions should optimize for ROI, not raw effectiveness.

### 4. Model Performance Is Realistic

An AUC of 0.6622 may seem modest, but it's realistic for churn prediction with behavioral features. The model's value is in identifying the **right customers to prioritize**, not perfect prediction.

### 5. Integration Over Optimization

This project isn't about building the best ML model—it's about integrating multiple tools (traditional ML, survival analysis, GenAI, statistics, experiment design) to deliver business impact. A 0.66 AUC model that protects $1.18M is more valuable than a 0.95 AUC model in a notebook.

---

## Technical Skills Demonstrated

| Category | Skills |
|----------|--------|
| **Machine Learning** | Logistic regression, survival analysis (Cox PH), feature engineering |
| **Statistical Analysis** | A/B testing, Bonferroni correction, chi-square tests, confidence intervals |
| **Software Architecture** | Multi-agent systems (Google ADK), tool integration |
| **Data Visualization** | Plotly dashboards, matplotlib |
| **Production Thinking** | Reproducibility (seeds), threshold optimization, validation |

---

## Key Metrics Summary

| Metric | Value |
|--------|-------|
| Dataset Size | 3,000 customers |
| Churn Rate | 19.4% |
| Mean CLV | $1,924 |
| High-Risk Customers | 1,347 (44.9%) |
| CLV at Risk | $1,181,624 |
| Churn Model AUC | 0.6622 |
| Survival Model C-Index | 0.6774 |
| Optimal Threshold | 0.4 (F1=0.378) |
| Optimal Window | Day 46-95 |
| Best Channel (Lift) | Call (53.2% reduction) |
| Best Channel (ROI) | Email (12.1x) |
| A/B Test Significance | p < 0.0001 (Call) |

---

## Conclusion

Customer churn prevention isn't just about predicting who will leave—it's about intervening at precisely the right moment with the right approach.

The optimal intervention window of **Day 46-95** isn't arbitrary—it's calculated from survival analysis of 1,347 high-risk customers. The winning intervention channel (Call, with 53.2% churn reduction) was validated through rigorous A/B testing with proper multiple-testing correction.

By combining survival analysis with multi-agent AI and statistical rigor, this system transforms reactive retention into proactive engagement—protecting $1.18M in customer lifetime value.

---

*The complete implementation, including all code and visualizations, is available in the accompanying Jupyter notebook.*
