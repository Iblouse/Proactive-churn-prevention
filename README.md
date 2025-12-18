# 🚀 Proactive Churn Prevention Multi-Agent System

> Transform reactive customer retention into proactive churn prevention with ML-powered predictions and AI-orchestrated interventions.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Google ADK](https://img.shields.io/badge/Google%20ADK-v1.0.0+-green.svg)](https://cloud.google.com/vertex-ai/docs/generative-ai/agent-builder/agent-development-kit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Technical Details](#-technical-details)
- [Feature Importance & Actionability](#-feature-importance--actionability)
- [A/B Test Results](#-ab-test-results)
- [Skills Demonstrated](#-skills-demonstrated)
- [Future Enhancements](#-future-enhancements)

---

## 🎯 Overview

This project implements an end-to-end **Proactive Churn Prevention System** that:

1. **Predicts** which customers will churn using ML classification
2. **Forecasts** *when* they'll churn using survival analysis
3. **Identifies** actionable intervention targets (not just predictive features)
4. **Recommends** personalized interventions via multi-agent AI
5. **Validates** effectiveness through rigorous A/B testing

### The Core Insight

> **Timing is everything.** The optimal intervention window (Day 46-95) is derived directly from survival analysis—not hardcoded assumptions.

> **Note**: This project isn't about building the best ML model—it's about integrating traditional ML, survival analysis, GenAI, statistics, and experiment design to deliver measurable business impact.

---

## 📊 Key Results

### Dataset Overview

| Metric | Value |
|--------|-------|
| Total Customers | 3,000 |
| Churn Rate | 19.4% (581 churned) |
| Mean CLV | $1,924 |
| At-Risk Customers | 1,347 (High + Critical tiers) |
| **CLV at Risk** | **$1,181,624** |

### Model Performance

| Model | Metric | Value |
|-------|--------|-------|
| Churn Classifier | AUC-ROC | 0.6622 |
| Churn Classifier | Best Threshold | 0.4 (F1=0.378) |
| Churn Classifier | Recall @ 0.4 | 85.3% |
| Survival Model | C-Index | 0.6774 |
| Survival Model | Optimal Window | Day 46-95 |

### A/B Test Winner

| Variant | Churn Rate | Lift | Significant? | ROI |
|---------|------------|------|--------------|-----|
| Control | 19.2% | — | — | — |
| **Call** | **9.0%** | **53.2%** | **✅ Yes (p<0.0001)** | 4.5x |
| Email | 15.8% | 18.2% | ❌ No | **12.1x** |

**Strategic Insight**: Call delivers highest lift (53.2%) but Email offers highest ROI (12.1x). Use tiered strategy: Email for volume, Call for VIPs.

---

## ✨ Key Features

### 🤖 Multi-Agent Architecture
- **Orchestrator Agent**: Coordinates workflow and manages communication
- **Behavioral Monitoring Agent**: Analyzes real-time customer patterns
- **Predictive Analytics Agent**: Calculates churn probability and timing
- **Intervention Strategy Agent**: Recommends personalized actions based on A/B results
- **Evaluation Agent**: Tracks effectiveness and provides feedback

### 📊 ML Pipeline
- Logistic Regression for calibrated churn probabilities (AUC: 0.6622)
- Cox Proportional Hazards for survival analysis (C-Index: 0.6774)
- 120-day observation window for actionable predictions
- Risk-adjusted prediction thresholds

### ⚖️ Feature Actionability Framework
- Combines prediction power with business actionability
- Identifies features you can actually influence (not just predict with)
- Prioritizes `engagement_score` and `feature_usage_pct` over `tenure_months`

### 🧪 A/B Testing Framework
- Multi-variant testing (5 intervention channels)
- Bonferroni correction for multiple comparisons (α=0.0125)
- Chi-square significance testing with confidence intervals
- ROI-based channel recommendations

### 📈 Executive Dashboard
- Risk distribution visualization
- Optimal intervention window (Day 46-95)
- A/B test results with statistical significance
- CLV at risk by customer tier ($1.18M)
- Intervention ROI comparison

---

## 🏗 Architecture

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
│           │  │           │  │            │ │           │
│  Tools:   │  │  Tools:   │  │   Tools:   │ │  Tools:   │
│ •Behavior │  │ •Churn    │  │ •Recommend │ │ •Survival │
│  Monitor  │  │  Score    │  │  Interv.   │ │  Analysis │
│           │  │ •Survival │  │ •Behavior  │ │ •At-Risk  │
│           │  │  Analysis │  │  Monitor   │ │  List     │
└───────────┘  └───────────┘  └───────────┘  └───────────┘
```

### Data Flow

```
Customer Data → Feature Engineering → Churn Model → Survival Analysis → Agent Processing → Intervention
     │                │                   │               │                    │               │
     ▼                ▼                   ▼               ▼                    ▼               ▼
  3,000           15 features         Probability      Days Until         Risk-Based        A/B Test
 Customers       + Actionability      + Risk Tier       Churn             Routing          Validated
                   Scoring            AUC: 0.6622    Window: 46-95     Channel Select    Winner: Call
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.10+
python --version

# Required packages
pip install numpy pandas scikit-learn plotly lifelines
pip install google-genai google-adk  # For agent functionality
```

### Running the Notebook

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/churn-prevention-agents.git
cd churn-prevention-agents
```

2. **Set up environment** (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

3. **Configure credentials** (for Vertex AI agents)
```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

4. **Run the notebook**
```bash
jupyter notebook proactive-churn-prevention.ipynb
```

### Notebook Sections

| Section | Description | Key Output |
|---------|-------------|------------|
| 1-2. Setup & Data | Configuration, synthetic data | 3,000 customers, 19.4% churn |
| 3. Feature Engineering | 15 features + actionability | engagement_score, feature_usage_pct |
| 4. Churn Model | Logistic Regression | AUC: 0.6622, threshold: 0.4 |
| 5. Survival Analysis | Cox PH model | Window: Day 46-95 |
| 6. Feature Importance | Importance × Actionability | Priority intervention targets |
| 7-8. Agent Definitions | Multi-agent architecture | 5 agents + tools |
| 9. A/B Testing | Multi-variant experiments | Winner: Call (53.2% lift) |
| 10. Dashboard | Executive visualization | 6 key metrics |

---

## 🔬 Technical Details

### Risk Tier Distribution (Actual)

| Tier | Count | Percentage | Churn Probability |
|------|-------|------------|-------------------|
| Low | 286 | 9.5% | <25% |
| Medium | 1,367 | 45.6% | 25-49% |
| High | 1,275 | 42.5% | 50-74% |
| Critical | 72 | 2.4% | ≥75% |

### Churn Model Performance

**Threshold Analysis:**

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| 0.3 | 21.8% | 96.6% | 0.356 |
| **0.4** | **24.3%** | **85.3%** | **0.378** |
| 0.5 | 26.0% | 57.8% | 0.358 |
| 0.6 | 33.3% | 37.1% | 0.351 |
| 0.7 | 45.9% | 14.7% | 0.222 |

**Confusion Matrix (threshold=0.4):**
```
              Predicted
              Retained  Churned
Actual
Retained       175      309
Churned         17       99
```

### Survival Analysis Results

**Top Hazard Ratios (Cox PH):**

| Feature | Hazard Ratio | Interpretation |
|---------|--------------|----------------|
| support_tickets_90d | 1.208*** | 20.8% higher churn hazard |
| is_inactive | 1.117** | 11.7% higher churn hazard |
| has_payment_issues | 1.077 | 7.7% higher churn hazard |

**Optimal Intervention Window:**

| Phase | Day Range | Derivation |
|-------|-----------|------------|
| Too Early | 0-46 | Before 25th percentile |
| **Optimal** | **46-95** | Between q25 and median |
| Peak | ~93 | Maximum effectiveness |
| Too Late | 95+ | After median |

---

## ⚖️ Feature Importance & Actionability

### The Problem with Traditional Feature Importance

| Rank | By Coefficient | Actionability |
|------|----------------|---------------|
| #1 | tenure_months (0.50) | **Low** ❌ |
| #2 | support_tickets_90d (0.19) | Medium |
| #3 | engagement_score (0.18) | **High** ✅ |

**`tenure_months`** is our best predictor but we can't change how long a customer has been with us!

### Combined Scoring: Importance × Actionability

```
Combined Score = |Coefficient| × Actionability Multiplier
```

Where: High = 3.0, Medium = 2.0, Low = 1.0

| Rank | By Combined Score | Coefficient | Actionability |
|------|-------------------|-------------|---------------|
| #1 | **engagement_score** | 0.18 | High (×3) = **0.54** |
| #2 | tenure_months | 0.50 | Low (×1) = 0.50 |
| #3 | **feature_usage_pct** | 0.16 | High (×3) = **0.47** |
| #4 | support_tickets_90d | 0.19 | Medium (×2) = 0.38 |

### Actionability Categories

**High Actionability** (can directly influence):
- `engagement_score` — Product tours, onboarding
- `feature_usage_pct` — Feature education, tips
- `email_open_rate` — Subject line optimization
- `login_frequency_monthly` — Notifications, value reminders
- `has_payment_issues` — Payment flexibility programs

**Medium Actionability** (can influence indirectly):
- `support_tickets_90d` — Better documentation
- `is_inactive` — Re-engagement campaigns
- `last_activity_days` — Triggered workflows

**Low Actionability** (difficult to change):
- `tenure_months` — Cannot change tenure
- `nps_score` — Lagging indicator
- `monthly_charges` — Strategic pricing decision

### Business Impact

Without actionability lens:
- ❌ Build campaigns around "customer tenure" (can't change)
- ❌ Focus on NPS scores (lagging indicator)

With combined framework:
- ✅ Target engagement score improvements
- ✅ Drive feature adoption
- ✅ Address payment issues proactively

---

## 🧪 A/B Test Results

### Multi-Variant Experiment (Actual Data)

| Variant | n | Churned | Rate | Lift | P-value | Significant? |
|---------|---|---------|------|------|---------|--------------|
| Control | 400 | 77 | 19.2% | — | — | (baseline) |
| Email | 400 | 63 | 15.8% | +18.2% | 0.2264 | ❌ No |
| Discount | 400 | 53 | 13.2% | +31.2% | 0.0275 | ❌ No* |
| **Call** | **400** | **36** | **9.0%** | **+53.2%** | **0.00005** | **✅ Yes** |
| Combined | 400 | 54 | 13.5% | +29.9% | 0.0356 | ❌ No* |

*Not significant at Bonferroni-adjusted α=0.0125

### ROI Analysis

| Channel | Lift | Cost | ROI | Best For |
|---------|------|------|-----|----------|
| **Email** | 18.2% | $0.50 | **12.1x** | All customers (scalable) |
| Discount | 31.2% | $15 | 9.4x | Price-sensitive segments |
| Combined | 29.9% | $65 | 8.2x | Critical + high-value |
| Call | 53.2% | $50 | 4.5x | VIP customers only |

### Recommended Tiered Strategy

| Customer Segment | Channel | Rationale |
|-----------------|---------|-----------|
| Critical + High-Value | Combined | Maximum retention for VIPs |
| Critical + Standard | Call | High-touch for urgent cases |
| High + Payment Issues | Discount | Address price sensitivity |
| High + Low Engagement | Email | Feature education at scale |
| Medium | Email | Cost-effective coverage |
| Low | Monitor | ROI doesn't justify intervention |

---

## 💼 Skills Demonstrated

### Machine Learning
- Binary classification with calibrated probabilities
- Survival analysis for time-to-event prediction
- Feature engineering from behavioral data
- Threshold optimization for business objectives
- **Feature actionability assessment** (beyond just prediction)

### Statistical Analysis
- A/B test design with power analysis
- Bonferroni correction for multiple comparisons
- Chi-square significance testing
- Confidence interval estimation

### Software Engineering
- Multi-agent system architecture (Google ADK)
- Tool integration and API design
- Session and memory management
- Reproducibility through seed management

### Data Visualization
- Interactive Plotly dashboards
- Feature importance × actionability charts
- Risk distribution visualization
- Intervention timing analysis

### Business Acumen
- ROI-based decision making
- Tiered intervention strategies
- Actionable vs. predictive feature distinction
- Stakeholder-friendly documentation

---

## 🔮 Future Enhancements

1. **Real Data Integration**
   - Replace synthetic data with production customer data
   - Implement data pipeline with incremental updates

2. **Model Improvements**
   - Test gradient boosting for higher AUC
   - Add SHAP values for local explainability
   - Implement ensemble methods

3. **Real-Time Scoring**
   - Stream processing with Apache Kafka
   - Sub-second prediction latency
   - Continuous model monitoring

4. **Feedback Loops**
   - Automated model retraining
   - Intervention outcome tracking
   - Dynamic actionability scoring

5. **Causal Inference**
   - Propensity score matching
   - Heterogeneous treatment effects
   - Uplift modeling

---

## 📁 Project Structure

```
churn-prevention-agents/
├── proactive-churn-prevention.ipynb    # Main notebook
├── README.md                            # This file
├── blog_post_corrected.md              # Technical blog post
├── flowchart.mermaid                    # Architecture diagrams
├── requirements.txt                     # Dependencies
├── charts/
│   ├── 01_risk_distribution.png
│   ├── 02_intervention_window.png
│   ├── 03_ab_test_results.png
│   ├── 04_intervention_roi.png
│   ├── 07_threshold_analysis.png
│   ├── 08_survival_curves.png
│   ├── 09_feature_actionability.png
│   └── 10_feature_selection_matrix.png
└── outputs/
    └── executive_dashboard.png
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

*Built with ❤️ using Google ADK, Vertex AI, and modern ML best practices.*

---

## 📊 Quick Reference

| What | Value |
|------|-------|
| Customers | 3,000 |
| Churn Rate | 19.4% |
| CLV at Risk | $1,181,624 |
| Model AUC | 0.6622 |
| Survival C-Index | 0.6774 |
| Optimal Window | Day 46-95 |
| Best Lift | Call (53.2%) |
| Best ROI | Email (12.1x) |
| Top Actionable Feature | engagement_score |
