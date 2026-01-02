# Why Most Churn Models Solve the Wrong Problem (And How to Fix It)

*Lessons from a decade of building ML systems that actually get used*

---

I have built churn models for over a decade now. Different industries, different scales, different algorithms. The technical part is straightforward: train a classifier, optimize the threshold, validate on held-out data.

But here is what I have noticed: the model is rarely the problem.

The problem is that classification models answer the wrong question. They tell you WHO will churn. They do not tell you WHEN to intervene. And that gap between prediction and action is where most retention efforts fall apart.

For this portfolio project, I wanted to demonstrate what a complete solution looks like: one that bridges that gap with survival analysis, A/B testing, and ROI modeling.

Here is the approach, and what it reinforces about applied ML.

---

## The Problem With Standard Churn Prediction

Let me show you what I mean.

![Risk Distribution](viz/01_risk_distribution.png)

I created a synthetic dataset of 6,000 customers for this portfolio project. After running my classifier, I had 2,825 customers flagged as "high risk," nearly half the customer base. Each one represents about $1,931 in lifetime value. That is **$2.54 million** sitting in a spreadsheet labeled "probably going to leave."

Now imagine you are the Customer Success lead. What do you do with this list?

- Call all 2,825? You do not have the bandwidth.
- Email everyone? Feels impersonal for your VIP accounts.
- Offer discounts? That eats into margins.

But here is the deeper problem: **when do you reach out?**

### The Timing Trap

Think about the last subscription you canceled.

There was probably a period, maybe a few weeks before you clicked "cancel," when you were frustrated but still persuadable. If someone had reached out then, with the right message, you might have stayed.

But if they'd contacted you the day after you'd made up your mind? Too late. You'd already mentally moved on.

That is the timing trap:
- **Too early**: Customer has not experienced friction yet. Your outreach feels random.
- **Too late**: They have already decided to leave. Your discount feels desperate.
- **Just right**: They are frustrated but have not committed. This is your window.

Standard classification models do not tell you where that window is. They give you a probability score and wish you luck.

I wanted to do better.

---

## Solution Part 1: Adding Survival Analysis

Here is the key insight that changed my approach:

**Classification asks:** "Will this customer churn?" (Yes/No)
**Survival analysis asks:** "How long until this customer churns?" (Time)

These are fundamentally different questions, and they require different models.

### What is Survival Analysis?

Survival analysis is a statistical technique originally developed for medical research to predict how long patients survive after treatment. It applies beautifully to any "time-to-event" problem: customer churn, employee attrition, equipment failure.

The Cox Proportional Hazards model I implemented gives us:
- **Survival curves**: Probability of "surviving" (not churning) over time
- **Hazard ratios**: Which features speed up or slow down the time to churn
- **Predictions**: Expected days until churn for each customer

### What the Survival Model Revealed

![Survival Curves](viz/02_survival_curves.png)

Look at that curve. Churn is not a sudden event. It is gradual:
- 8% churned by day 30
- 13% by day 60
- 21% by day 120

There is a rhythm to customer departure. A pattern that unfolds over months. My classification model treated all churners the same and could not see this temporal structure.

### Finding the Optimal Window

Here is where it gets practical.

I took my 2,825 high-risk customers and asked the survival model: "For each of these, when do you predict they will churn?"

The answers clustered tightly:
- 25th percentile: Day 91
- Median: Day 95  
- 75th percentile: Day 97

From this distribution, I derived the intervention window:
- **Before Day 45**: Too early. Customer has not hit frustration yet.
- **Day 45-95**: Optimal. Customer experiencing issues, still persuadable.
- **After Day 95**: Too late. More than half have already churned.

![Intervention Window](viz/06_intervention_timing.png)

**This window came from the data, not from intuition or industry benchmarks.** That is the power of survival analysis: the timing adapts to your specific customer behavior.

---

## Solution Part 2: Validating With A/B Testing

At this point, I had:
1. A classification model identifying WHO is at risk
2. A survival model telling me WHEN to act

But a crucial question remained: **Do interventions actually work?**

### Why This Matters

It is tempting to assume that "contacting at-risk customers" reduces churn. But without evidence, that is just a hypothesis. Maybe your outreach annoys people. Maybe your discount attracts the wrong customers. Maybe the effect is real but too small to justify the cost.

Predictions are hypotheses. Experiments are proof.

### The Experiment

I designed a 5-arm A/B test:

| Variant | Description | Cost |
|---------|-------------|------|
| Control | No intervention | $0 |
| Email | Automated campaign | $0.50 |
| Discount | 10% off offer | $10.00 |
| Call | Personal outreach | $35.00 |
| Combined | All channels | $45.50 |

900 customers per group. 4,500 total.

### Why Bonferroni Correction Matters

With 4 treatment variants, I am running 4 hypothesis tests against control. Without correction, I'd expect about 0.2 false positives by chance (4 tests × 5% error rate).

Bonferroni correction adjusts the significance threshold:

```
Adjusted α = 0.05 / 4 = 0.0125
```

A result is only "significant" if p < 0.0125, not the usual 0.05.

This matters. Watch what happens:

### The Results

![A/B Test Results](viz/08_ab_test_results.png)

| Variant | Churn Rate | Lift vs Control | p-value | Significant? |
|---------|------------|-----------------|---------|--------------|
| Control | 21.7% | - | - | baseline |
| Email | 17.6% | +19.0% | 0.033 | **No** |
| Discount | 15.6% | +28.2% | 0.001 | Yes |
| **Call** | **9.9%** | **+54.4%** | **<0.0001** | **Yes** |
| Combined | 15.0% | +30.8% | <0.001 | Yes |

Notice Email. Its p-value of 0.033 would be "significant" under naive α = 0.05. But under Bonferroni correction, it fails.

Without proper methodology, I might have declared Email effective when the evidence was actually inconclusive. This is exactly why statistical rigor matters.

**Call is the clear winner**: 54.4% churn reduction, p < 0.0001.

---

## Solution Part 3: ROI Changes Everything

Call won the A/B test. Ship it, right?

Not so fast.

I calculated ROI for each channel:

```
ROI = (Customer Lifetime Value × Lift) / Cost per Customer
```

![Lift vs ROI](viz/10_lift_vs_roi.png)

| Channel | Lift | Cost | ROI |
|---------|------|------|-----|
| Email | 4.1pp | $0.50 | **158.8x** |
| Discount | 6.1pp | $10.00 | 11.8x |
| Call | 11.8pp | $35.00 | 6.5x |
| Combined | 6.7pp | $45.50 | 2.8x |

Email's ROI is **24 times higher** than Call's.

This creates a strategic question: Do you maximize impact (Call) or efficiency (Email)?

### The Tiered Solution

The answer is neither. You build a tiered system:

| Customer Segment | Recommended Action | Why |
|-----------------|-------------------|-----|
| Critical risk + High CLV | Call | Impact justifies cost |
| High risk + Standard CLV | Email | Efficiency at scale |
| Medium risk | Email | Low cost, positive ROI |
| Low risk | Monitor only | ROI does not justify any action |

This is the kind of nuanced recommendation that a single "best model" cannot provide.

---

## Bonus Insight: The Actionability Gap

One more lesson worth sharing.

My strongest predictive feature was `tenure_months`, how long the customer had been with us. Strong negative coefficient; newer customers churn more.

But here is the problem: **you cannot change someone's tenure.**

It is useful for prediction, useless for intervention.

I built a framework scoring features on both dimensions:

![Feature Importance vs Actionability](viz/04_feature_importance_comparison.png)

When I factored in actionability, the rankings changed completely. `engagement_score` jumped to #1, not because it is the best predictor, but because it combines decent predictive power with high actionability. We can actually influence engagement through product tours, feature education, targeted content.

This is the gap between academic ML and applied ML. In a Kaggle competition, you optimize for prediction accuracy. In the real world, you need features you can actually do something about.

---

## What This Project Reinforces

![Executive Summary](viz/Executive_summary.png)

Building this system reinforced several principles I have come to value over my career:

### 1. Model metrics are gates, not goals

The AUC was 0.66. "Mediocre" by competition standards. But the model's job was to *rank* customers by risk, and 0.66 does that fine. The real validation came from the A/B test: did acting on predictions actually improve retention?

I stopped optimizing when the model was "good enough" and focused on the questions that actually mattered.

### 2. The right question beats the right algorithm

I could have spent another month pushing AUC from 0.66 to 0.75. Instead, I asked: "When should we intervene? Which channel works? Is it worth the cost?"

Those questions delivered more value than any hyperparameter tuning ever would.

### 3. Statistical rigor protects you from yourself

Bonferroni correction prevented me from declaring Email "significant" when it was not. Proper methodology is not bureaucratic overhead. It is protection against false confidence.

### 4. Systems beat models

No single model answered all the questions. I needed:
- Classification for WHO
- Survival analysis for WHEN
- A/B testing for WHAT WORKS
- ROI analysis for IS IT WORTH IT

The value was not in any individual component. It was in how they worked together.

---

## The Results

For my simulated 6,000 customer dataset:

**Model Metrics** (the gates):
- Classification AUC: 0.6612
- Survival C-Index: 0.6645

**Business Metrics** (what matters):
- $2.54M CLV at risk identified
- Day 45-95 optimal intervention window
- 54.4% churn reduction validated (p < 0.0001)
- ~$264K revenue protected
- ~137 customers saved

---

## Final Thought

Here is what a decade of applied ML has taught me:

**Predicting churn is easy. Preventing it is hard.**

Prevention requires answering questions that classification models were not designed for:
- When should we act?
- What intervention works?
- Is it worth the cost?
- Which features can we actually influence?

Those questions require more than a good model. They require a complete system and the experience to know which questions to ask in the first place.

The model's AUC was 0.66. The system protected $264K.

That is the difference between building models and solving problems.

---

*The complete implementation, including all code, models, and visualizations, is available in the accompanying Jupyter notebook. Built with Python, scikit-learn, lifelines (survival analysis), and Google ADK.*
