# LinkedIn Posts: Proactive Churn Prevention System

---

## VERSION 1: The Realization Story (~350 words)

---

After 10+ years building ML systems, I have noticed a pattern.

We optimize classifiers, tune hyperparameters, celebrate AUC improvements. Then we hand over a ranked list of at-risk customers and the business team asks:

"Great, but when should we actually call them?"

Most churn models cannot answer that. They tell you WHO will leave, not WHEN to intervene.

For this portfolio project, I decided to solve both problems properly.

**The insight:**
I added survival analysis (Cox Proportional Hazards) to predict timing, not just probability. It revealed something fascinating: there is a specific window where intervention works best.

Day 45-95. Derived from the data, not assumed.

Before that? Customer has not hit their frustration point.
After that? They have already decided to leave.

**The validation:**
I did not just trust the model. I ran a proper A/B test.

5 variants. 900 customers each. Bonferroni correction for multiple comparisons.

Results:
- Call: 54.4% churn reduction (p < 0.0001)
- Email: 19% reduction, but 158.8x ROI

**The twist:**
Call had the best results. Email had 24x better ROI.

The answer is not "pick one." It is build a tiered system:
- Call for high-value customers (impact justifies cost)
- Email for everyone else (efficiency at scale)

**What I learned:**

1. Model metrics are gates, not goals. AUC 0.66 was enough to rank customers. The A/B test proved it worked.

2. The best predictor is not always the best intervention target. Tenure was my top feature, but you cannot change how long someone's been a customer.

3. Statistical rigor matters. Without Bonferroni correction, I would have declared Email "significant" when it was not.

Business impact (simulated): ~$264K protected, ~137 customers saved.

Sometimes the most valuable thing is not building a better model. It is asking a better question.

Full project with code: [link in comments]

#MachineLearning #DataScience #ChurnPrediction #ABTesting #Portfolio

---

## VERSION 2: The Contrarian Hook (~280 words)

---

Unpopular opinion: Your churn model's AUC does not matter.

After 10+ years in ML, I have seen teams celebrate 0.85 AUC scores that never moved the retention needle.

For this portfolio project, my AUC was 0.66. "Mediocre" by Kaggle standards.

But here is what experience has taught me:

The model's job is to RANK customers by risk. 0.66 does that fine.

The real questions are:
- WHEN should we intervene?
- WHICH channel should we use?
- Is it even WORTH the cost?

Classification models do not answer any of these.

So I built a system that does:

**For timing:** Added survival analysis. Discovered the optimal window is Day 45-95, derived from data, not industry benchmarks.

**For channels:** Ran an A/B test. Call reduced churn by 54.4%. But Email had 158.8x ROI (24x better than Call).

**For cost:** Built ROI analysis into the recommendation engine. High-value customers get calls. Everyone else gets emails.

The model metrics were "just okay":
- AUC: 0.6612
- C-Index: 0.6645

The business metrics were what mattered:
- $2.54M CLV at risk identified
- ~$264K protected
- 54.4% churn reduction validated (p < 0.0001)

Here is the thing about applied ML that competitions do not teach you:

A perfect model that answers the wrong question is worthless.
An adequate model that answers the right question is invaluable.

I stopped optimizing AUC when I realized the bottleneck was not prediction accuracy. It was knowing what to DO with the predictions.

That shift in thinking changed everything about how I approach problems now.

#DataScience #MachineLearning #AppliedML #ChurnPrediction

---

## VERSION 3: The Numbers That Matter (~200 words)

---

Portfolio project results that I am proud of:

❌ NOT this: AUC 0.66

✅ THIS:
- 54.4% churn reduction (validated by A/B test)
- p < 0.0001 (statistically significant)
- 158.8x ROI on email interventions
- ~$264K revenue protected
- Day 45-95 optimal window (derived from survival analysis)

Here is why the distinction matters:

AUC tells you if your model works in a vacuum.
Business metrics tell you if your SYSTEM works in reality.

For this project, I combined:
- Classification (who's at risk)
- Survival analysis (when to act)
- A/B testing (what works)
- ROI modeling (what is worth it)

No single model could answer all these questions.

The most interesting finding?

Call had the best results (54.4% lift).
Email had the best ROI (158.8x).

They are both "right" for different customers.

That is the kind of nuance that gets lost when we reduce everything to a single accuracy metric.

Applied ML is not about building the best model. It is about building the best system.

Full code and methodology in my portfolio.

#DataScience #MachineLearning #Portfolio

---

## VERSION 4: Carousel Script (Story-Driven)

---

**Slide 1:**
After 10+ years in ML, I keep seeing the same mistake.

We build churn models that answer the wrong question.

Here is what I mean...

**Slide 2:**
THE WRONG QUESTION:
"Will this customer churn?"

THE RIGHT QUESTION:
"When should we intervene?"

Knowing WHO leaves is useless without knowing WHEN to act.

**Slide 3:**
THE PROBLEM:

2,825 high-risk customers
$2.54M in lifetime value at risk

But when do you reach out?
Too early = feels random
Too late = they have already decided

**Slide 4:**
THE SOLUTION:

Added survival analysis to predict TIMING.

Discovered an optimal window: Day 45-95

Not assumed. Derived from the data.

![Intervention Window](viz/06_intervention_timing.png)

**Slide 5:**
BUT DOES IT WORK?

Ran a proper A/B test:
- 5 variants
- 900 customers each
- Bonferroni correction

Because predictions are hypotheses.
Experiments are proof.

**Slide 6:**
THE RESULTS:

![A/B Test Results](viz/08_ab_test_results.png)

Call: 54.4% churn reduction
(p < 0.0001)

It works. Statistically validated.

**Slide 7:**
THE TWIST:

Call had the best RESULTS.
Email had the best ROI (158.8x).

![Lift vs ROI](viz/10_lift_vs_roi.png)

24x more efficient.

**Slide 8:**
THE STRATEGY:

Do not pick one. Build a tiered system:

- High-value customers → Call
- Everyone else → Email

Match intervention to customer value.

**Slide 9:**
WHAT I LEARNED:

1. Model metrics are gates, not goals
2. The best predictor ≠ best intervention target
3. Statistical rigor prevents false confidence
4. Ask better questions, not just build better models

**Slide 10:**
THE TAKEAWAY:

My model's AUC was 0.66.
"Mediocre" by Kaggle standards.

But it protected ~$264K in revenue.
Validated by experiment. Not hoped for.

That is applied ML.

[Full project in comments]
