# Proactive Churn Prevention Multi-Agent System

End-to-end churn prevention system that ranks churn risk, estimates churn timing, routes interventions, and validates uplift with multi-variant A/B testing.

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## What this project does

1. **Predicts who** is likely to churn (classification, probability output).
2. **Estimates when** churn is likely to happen (survival-style timing; Cox PH when available, fallback estimator otherwise).
3. **Recommends what** to do (policy-based intervention routing with expected lift, cost, and ROI).
4. **Measures impact** with a multi-variant A/B test (control vs Email, Discount, Call, Combined).

This is designed as a business solution, not a leaderboard model.

---

## Key results (example run from the notebook)

### Data
- **Customers:** 6,000 (synthetic)
- **Baseline churn:** 21.0%

### Model quality (gating checks)
- **Churn AUC:** 0.6612
- **Operating threshold:** 0.5
- **Precision @ 0.5:** 29.3%
- **Recall @ 0.5:** 66.3%
- **F1 @ 0.5:** 0.406

Threshold trade-off (test set):

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.3 | 23.2% | 96.0% | 0.374 |
| 0.4 | 24.9% | 84.9% | 0.386 |
| 0.5 | 29.3% | 66.3% | 0.406 |
| 0.6 | 34.5% | 35.3% | 0.349 |
| 0.7 | 37.1% | 10.3% | 0.161 |

Confusion matrix @ 0.5 (test set):

```
              Predicted
              Retained  Churned
Actual
Retained         545      403
Churned           85      167
```

### Risk tiers (example run)
| Tier | Count | Share | Churn Probability |
|---|---:|---:|---|
| Low | 492 | 8.2% | <25% |
| Medium | 2,683 | 44.7% | 25–49% |
| High | 2,718 | 45.3% | 50–74% |
| Critical | 107 | 1.8% | ≥75% |

### Timing and intervention window (high-risk cohort)
- **High-risk cohort:** 2,825 customers (churn_probability ≥ 0.5)
- **Window:** day 31–74 (recommended start-to-end)
- **Optimal point:** around day 68 (center of window)

### A/B test uplift (n=900 per variant, Bonferroni α=0.0125)

| Variant | Churn rate | Rel. lift vs control | Abs. Δ (pp) | p-value | Significant |
|---|---:|---:|---:|---:|---|
| Control | 21.7% | - | - | - | - |
| Email | 17.6% | 19.0% | 4.1 | 0.0326 | No |
| Discount | 15.6% | 28.2% | 6.1 | 0.0011 | Yes |
| **Call** | **9.9%** | **54.4%** | **11.8** | **1.1e-11** | **Yes** |
| Combined | 15.0% | 30.8% | 6.7 | 0.00033 | Yes |

Interpretation:
- **Call** gives the largest churn reduction in this run (absolute and relative).
- **Email** often yields the highest ROI due to low cost, even if lift is smaller.

### Expected business impact (high-risk cohort)

These estimates use the notebook policy:
- Value at risk is **CLV × churn_probability**.
- Value saved is **value_at_risk × expected_lift** (lift comes from A/B results, adjusted by tier).

| Metric | Value |
|---|---:|
| Customers targeted | 2,825 |
| Total CLV (targeted) | $2,542,079 |
| Expected value at risk | $1,486,603 |
| Expected value saved | $690,827 |
| Estimated intervention cost | $60,030 |
| Expected ROI | 11.5x |

Channel mix for targeted customers (example run): Call 1,261, Discount 1,155, Email 317, Combined 92.

---

## Architecture (notebook)

The notebook is organized as a production-style workflow:

1. **Data generation / ingestion** (synthetic customer base)
2. **Feature engineering** (engagement, support, payment issues, inactivity, NPS, CLV)
3. **Churn model** (probability scoring + threshold analysis)
4. **Timing model** (Cox PH when available; fallback timing estimator)
5. **Intervention engine** (channel recommendation + ROI estimate)
6. **A/B testing** (multi-variant uplift measurement)
7. **Executive dashboard** (optional charts)

---

## Run on Kaggle

1. Upload `Proactive_Churn_Prevention.ipynb` to a Kaggle notebook.
2. If you want the Cox PH survival model, enable internet and install dependencies in the first code cell:

```bash
pip -q install lifelines
```

3. Run all cells top-to-bottom.

Notes:
- If `lifelines` is not available, the notebook will fall back to a deterministic timing estimator and still run end-to-end.
- The A/B test in this repo is simulated to demonstrate measurement and decision logic on synthetic data.

---

## Outputs

The notebook generates recruiter-friendly charts (examples):
- Risk distribution and cohort sizing
- Precision/recall vs threshold
- Intervention window plot
- A/B uplift comparison
- Executive dashboard (optional)

---

## Limitations and next steps

- Replace synthetic data with real product and CRM signals.
- Add calibration, monitoring, and drift detection.
- Learn intervention policies with uplift modeling (CATE) instead of static rules.
- Productionize as a service (API + scheduler + experiment tracker).

---

## License

MIT
