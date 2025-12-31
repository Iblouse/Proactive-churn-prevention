# LinkedIn Post: Proactive Churn Prevention System (Portfolio Project)

---

## VERSION 1: Visual Results-Focused (~300 words)

---

**I stopped optimizing my churn model at AUC 0.66.**

Here's why that was the right decision for this portfolio project:

Most churn tutorials focus on prediction accuracy. I wanted to demonstrate something different: **a complete business solution**.

The question I answered wasn't "who will churn?" - it was **"when should we intervene?"**

![Executive Summary](viz/Executive_summary.png)

**What I built:**

A dual-model architecture:
- Logistic Regression for churn probability (AUC: 0.6612)
- Cox Proportional Hazards for timing (C-Index: 0.6645)

**The key insight:**

![Intervention Window](viz/06_intervention_timing.png)

The optimal intervention window is **Day 45-95** - derived from survival model predictions, not assumptions.

**A/B test results (n=900 per variant, Bonferroni-corrected):**

![A/B Test Results](viz/08_ab_test_results.png)

- Call: **54.4% churn reduction** (p < 0.0001)
- Email: 19.0% reduction, but **158.8x ROI**

**The strategy: Tiered intervention**

![Lift vs ROI](viz/10_lift_vs_roi.png)

- Email for broad outreach (best ROI)
- Call for high-value customers (best lift)

**Business impact:**
- $2.54M CLV at risk
- ~137 customers saved
- ~$264K revenue protected

**Technical stack:**
- Python, scikit-learn, lifelines
- Google ADK, Vertex AI
- A/B testing with Bonferroni correction

Model metrics are gating checks, not goals. AUC 0.66 was sufficient - business outcomes matter.

Full project with code in portfolio.

#MachineLearning #DataScience #ChurnPrediction #ABTesting #Portfolio

---

## VERSION 2: Single Image Post (~150 words)

---

**Portfolio Project: Churn Prevention System**

Built a system that answers "when to intervene" - not just "who will churn."

![Executive Summary](viz/Executive_summary.png)

**Results:**
- 54.4% churn reduction (best channel)
- 158.8x ROI (most efficient channel)
- Day 45-95 optimal window (data-derived)
- ~$264K revenue protected

**Technical approach:**
- Dual-model: Classification + Survival Analysis
- A/B testing with Bonferroni correction
- Multi-agent AI orchestration

The model's AUC is 0.66. That's intentional - I focused on the complete system, not hyperparameter tuning.

Full code available.

#DataScience #MachineLearning #Portfolio

---

## VERSION 3: Carousel Script (10 slides with chart recommendations)

---

**Slide 1 (Hook):**
Portfolio Project:
Proactive Churn Prevention System

[Use: Executive_summary.png]

**Slide 2:**
THE WRONG QUESTION:
"Will this customer churn?"

THE RIGHT QUESTION:
"When should we intervene?"

**Slide 3:**
THE PROBLEM:
6,000 customers
2,825 high-risk (47%)
$2.54M CLV at risk

[Use: 01_risk_distribution.png]

**Slide 4:**
TWO-MODEL ARCHITECTURE:

Classification: WHO will churn
- AUC: 0.66

Survival Analysis: WHEN they'll churn
- C-Index: 0.66

**Slide 5:**
OPTIMAL INTERVENTION WINDOW:
Day 45-95

Derived from survival model predictions

[Use: 06_intervention_timing.png]

**Slide 6:**
A/B TEST DESIGN:
5 variants x 900 customers
Bonferroni correction (alpha = 0.0125)

[Use: 08_ab_test_results.png]

**Slide 7:**
RESULTS - LIFT:
Call: +54.4% (p < 0.0001) - WINNER
Discount: +28.2%
Combined: +30.8%
Email: +19.0%

**Slide 8:**
RESULTS - ROI:
Email: 158.8x - BEST EFFICIENCY
Discount: 11.8x
Call: 6.5x
Combined: 2.8x

[Use: 10_lift_vs_roi.png]

**Slide 9:**
THE STRATEGY:
Tiered intervention
- Email: Broad outreach (158.8x ROI)
- Call: High-value customers (54.4% lift)

**Slide 10:**
KEY LESSON:
Model metrics are gating checks
Business metrics are the goal

AUC 0.66 was good enough
$264K protected is what matters

---

## RECOMMENDED IMAGES FOR LINKEDIN

For maximum impact, use these charts:

1. **Executive_summary.png** - Best for main post image (shows complete system)
2. **08_ab_test_results.png** - Clear A/B test visualization
3. **10_lift_vs_roi.png** - Shows the ROI insight
4. **06_intervention_timing.png** - Shows the timing insight

These charts are self-explanatory and work well as standalone visuals.
