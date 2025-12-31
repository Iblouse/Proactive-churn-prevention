# I Stopped Chasing AUC and Started Asking "When?" - A Portfolio Project That Changed How I Think About Churn Prevention

*Building a complete retention system, not just a classifier*

---

Most churn prediction tutorials focus on one thing: building a model that predicts who will churn.

I wanted to go further.

For this portfolio project, I set out to answer a different question - the one that actually matters for business impact:

**"When exactly should we intervene?"**

The result is a complete churn prevention system that reduced churn by 54% and protected $2.54M in simulated customer lifetime value.

![Executive Summary](viz/Executive_summary.png)

*The complete system: from risk identification to validated intervention strategies*

---

## The Problem Setup

I created a synthetic dataset of 6,000 customers to demonstrate this methodology. Nearly half of them - 2,825 - showed signs of churn risk, representing **$2.54 million** in CLV exposure.

![Risk Distribution](viz/01_risk_distribution.png)

*Figure 1: Nearly 47% of customers fall into High or Critical risk tiers*

---

## The Two-Model Architecture

To answer both "who" and "when," I implemented two complementary models:

**Model 1: Churn Classification** (Logistic Regression)
- AUC-ROC: 0.6612
- Recall @ 0.5: 66.3% | Precision: 29.3% | F1: 0.406

**Model 2: Survival Analysis** (Cox Proportional Hazards)
- Concordance Index: 0.6645

![Survival Curves](viz/02_survival_curves.png)

*Figure 2: Churn occurs gradually over 120 days - timing interventions matters*

---

## Finding the Optimal Intervention Window

I analyzed predicted days until churn for high-risk customers and derived the intervention window directly from the data:

- **Too Early**: Days 0-45
- **Optimal**: Days 45-95
- **Peak Effectiveness**: Day 93
- **Too Late**: After Day 95

![Intervention Window](viz/06_intervention_timing.png)

*Figure 3: The optimal window (Day 45-95) derived from survival model predictions*

---

## Validating with A/B Testing

Model predictions are hypotheses. A/B tests are proof.

I designed a 5-arm experiment with 900 customers per variant, applying Bonferroni correction (adjusted alpha = 0.0125):

![A/B Test Results](viz/08_ab_test_results.png)

*Figure 4: Call achieves 54.4% churn reduction (p < 0.0001)*

**Winner: Call** with 54.4% relative lift, statistically significant.

---

## The ROI Analysis

Call delivered the biggest impact. But ROI tells a different story:

![Intervention ROI](viz/10_lift_vs_roi.png)

*Figure 5: Email has 158.8x ROI despite lower lift - the optimal strategy is tiered*

| Channel | Lift | Cost | ROI |
|---------|------|------|-----|
| **Email** | +19.0% | $0.50 | **158.8x** |
| Discount | +28.2% | $10.00 | 11.8x |
| Call | +54.4% | $35.00 | 6.5x |
| Combined | +30.8% | $45.50 | 2.8x |

**Strategic insight**: Email for broad outreach, Call for high-value customers.

---

## Results Summary

| Category | Metric | Value |
|----------|--------|-------|
| **Model** | Churn AUC | 0.6612 |
| **Model** | Survival C-Index | 0.6645 |
| **Business** | CLV at Risk | $2,542,079 |
| **Business** | Optimal Window | Day 45-95 |
| **Business** | Best Lift | Call (+54.4%) |
| **Business** | Best ROI | Email (158.8x) |
| **Business** | Revenue Protected | ~$264K |

---

## Technical Capabilities Demonstrated

| Capability | Implementation |
|------------|----------------|
| Risk Scoring | Logistic Regression, threshold optimization |
| Timing Prediction | Cox Proportional Hazards survival analysis |
| Agent Orchestration | Google ADK multi-agent system |
| Experiment Design | A/B testing with Bonferroni correction |
| ROI Analysis | Cost modeling, channel comparison |

---

## Key Takeaways

1. **Timing Comes from Data**: The Day 45-95 window was derived from survival model predictions
2. **Statistical Rigor Matters**: Bonferroni correction prevents false positives
3. **ROI Beats Lift**: Call has highest impact but Email has best ROI
4. **Model Metrics Are Gating Checks**: AUC 0.66 was sufficient - business metrics matter

This is what applied ML looks like: adequate models, rigorous experimentation, measurable outcomes.

---

*The complete implementation, including all code and visualizations, is available in the accompanying Jupyter notebook.*
