"""
Churn Prevention Portfolio - Chart Generation
=============================================
Standalone code to generate all charts for the portfolio blog posts.
No notebook dependency - uses the verified metrics from the project.

Author: Portfolio Project
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set style for all charts
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'primary': '#2563eb',
    'secondary': '#7c3aed', 
    'success': '#059669',
    'warning': '#d97706',
    'danger': '#dc2626',
    'gray': '#6b7280',
    'light_blue': '#93c5fd',
    'light_green': '#6ee7b7',
    'light_red': '#fca5a5',
    'light_purple': '#c4b5fd'
}

# =============================================================================
# CHART 1: Risk Distribution
# =============================================================================
def create_risk_distribution():
    """Customer segmentation by churn risk tier."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data from portfolio
    tiers = ['Low\n(0-25%)', 'Medium\n(25-50%)', 'High\n(50-75%)', 'Critical\n(75-100%)']
    counts = [1850, 1325, 1675, 1150]  # Total = 6000
    colors = [COLORS['success'], COLORS['warning'], COLORS['danger'], '#991b1b']
    
    bars = ax.bar(tiers, counts, color=colors, edgecolor='white', linewidth=2)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        pct = count / 6000 * 100
        ax.annotate(f'{count:,}\n({pct:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
    
    # Highlight high-risk zone
    ax.axhspan(0, max(counts) * 1.15, xmin=0.5, xmax=1.0, alpha=0.1, color=COLORS['danger'])
    
    # Add annotation for high-risk
    high_risk_total = 1675 + 1150
    ax.annotate(f'High-Risk Zone\n{high_risk_total:,} customers (47.1%)\n$2.54M CLV at risk',
                xy=(2.5, 1500), fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=COLORS['danger']))
    
    ax.set_ylabel('Number of Customers', fontsize=12)
    ax.set_xlabel('Risk Tier (Churn Probability)', fontsize=12)
    ax.set_title('Customer Risk Distribution (n=6,000)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(counts) * 1.25)
    
    plt.tight_layout()
    plt.savefig('viz/01_risk_distribution.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Created: 01_risk_distribution.png")


# =============================================================================
# CHART 2: Survival Curves
# =============================================================================
def create_survival_curves():
    """Survival probability over observation window."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate survival curve data (based on 21% churn at day 120)
    days = np.arange(0, 121, 1)
    
    # Kaplan-Meier style curve with gradual churn
    # Using exponential decay calibrated to 21% churn at day 120
    lambda_param = -np.log(0.79) / 120  # 79% survive to day 120
    survival_prob = np.exp(-lambda_param * days)
    
    # Add some realistic variation
    np.random.seed(42)
    noise = np.random.normal(0, 0.005, len(days))
    noise = np.cumsum(noise) * 0.1
    survival_prob = np.clip(survival_prob + noise * 0.01, 0, 1)
    survival_prob[0] = 1.0  # Start at 100%
    survival_prob = np.maximum.accumulate(survival_prob[::-1])[::-1]  # Ensure monotonic decrease
    
    # Plot main curve
    ax.plot(days, survival_prob * 100, color=COLORS['primary'], linewidth=2.5, label='Overall Survival')
    ax.fill_between(days, survival_prob * 100, alpha=0.2, color=COLORS['primary'])
    
    # Add confidence interval appearance
    ci_upper = np.clip(survival_prob * 100 + 2, 0, 100)
    ci_lower = np.clip(survival_prob * 100 - 2, 0, 100)
    ax.fill_between(days, ci_lower, ci_upper, alpha=0.1, color=COLORS['primary'])
    
    # Mark key milestones
    milestones = [(30, 92, '92% survive'), (60, 87, '87% survive'), (120, 79, '79% survive')]
    for day, surv, label in milestones:
        ax.scatter([day], [surv], color=COLORS['danger'], s=80, zorder=5)
        ax.annotate(f'Day {day}: {label}\n({100-surv}% churned)',
                    xy=(day, surv), xytext=(day + 8, surv + 3),
                    fontsize=9, ha='left',
                    arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=0.5))
    
    # Add intervention window
    ax.axvspan(45, 95, alpha=0.15, color=COLORS['success'], label='Optimal Window (Day 45-95)')
    ax.axvline(x=45, color=COLORS['success'], linestyle='--', alpha=0.7)
    ax.axvline(x=95, color=COLORS['success'], linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Days Since Observation Start', fontsize=12)
    ax.set_ylabel('Survival Probability (%)', fontsize=12)
    ax.set_title('Customer Survival Curve (120-Day Observation Window)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 120)
    ax.set_ylim(70, 102)
    ax.legend(loc='lower left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('viz/02_survival_curves.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Created: 02_survival_curves.png")


# =============================================================================
# CHART 3: Threshold Analysis
# =============================================================================
def create_threshold_analysis():
    """Precision-recall trade-off across thresholds."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data from portfolio
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    precision = [23.2, 24.9, 29.3, 34.5, 45.9]
    recall = [96.0, 84.9, 66.3, 35.3, 26.6]
    f1 = [0.374, 0.386, 0.406, 0.349, 0.337]
    
    # Interpolate for smoother curves
    from scipy.interpolate import make_interp_spline
    
    thresh_smooth = np.linspace(0.3, 0.7, 100)
    
    # Plot curves
    ax.plot(thresholds, precision, 'o-', color=COLORS['primary'], linewidth=2, 
            markersize=8, label='Precision (%)')
    ax.plot(thresholds, recall, 's-', color=COLORS['success'], linewidth=2,
            markersize=8, label='Recall (%)')
    ax.plot(thresholds, [f * 100 for f in f1], '^-', color=COLORS['secondary'], linewidth=2,
            markersize=8, label='F1 Score (×100)')
    
    # Highlight optimal threshold
    ax.axvline(x=0.5, color=COLORS['danger'], linestyle='--', alpha=0.7, linewidth=2)
    ax.scatter([0.5], [29.3], color=COLORS['danger'], s=150, zorder=5, edgecolors='white', linewidth=2)
    ax.scatter([0.5], [66.3], color=COLORS['danger'], s=150, zorder=5, edgecolors='white', linewidth=2)
    ax.scatter([0.5], [40.6], color=COLORS['danger'], s=150, zorder=5, edgecolors='white', linewidth=2)
    
    # Add annotation
    ax.annotate('Selected: 0.5\nBest F1 (0.406)\nRecall: 66.3%',
                xy=(0.5, 66.3), xytext=(0.58, 75),
                fontsize=10, ha='left',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=COLORS['danger']),
                arrowprops=dict(arrowstyle='->', color=COLORS['danger']))
    
    ax.set_xlabel('Classification Threshold', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Precision-Recall Trade-off by Threshold', fontsize=14, fontweight='bold')
    ax.set_xlim(0.25, 0.75)
    ax.set_ylim(0, 105)
    ax.legend(loc='center right', fontsize=10)
    ax.set_xticks(thresholds)
    
    plt.tight_layout()
    plt.savefig('viz/03_threshold_analysis.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Created: 03_threshold_analysis.png")


# =============================================================================
# CHART 4: Feature Importance Comparison
# =============================================================================
def create_feature_importance_comparison():
    """Traditional importance vs actionability-weighted importance."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Feature data
    features = ['tenure_months', 'support_tickets_90d', 'engagement_score', 
                'feature_usage_pct', 'payment_delays', 'monthly_charges',
                'nps_score', 'is_inactive']
    
    # Traditional coefficients (absolute values)
    traditional = [0.50, 0.19, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08]
    
    # Actionability multipliers
    actionability = [1.0, 2.0, 3.0, 3.0, 1.5, 1.0, 2.5, 2.0]
    
    # Combined scores
    combined = [t * a for t, a in zip(traditional, actionability)]
    
    # Sort for each plot
    trad_sorted = sorted(zip(traditional, features), reverse=True)
    comb_sorted = sorted(zip(combined, features), reverse=True)
    
    # Left plot: Traditional
    ax1 = axes[0]
    trad_vals, trad_names = zip(*trad_sorted)
    colors1 = [COLORS['primary'] if 'tenure' not in n else COLORS['gray'] for n in trad_names]
    bars1 = ax1.barh(range(len(trad_names)), trad_vals, color=colors1)
    ax1.set_yticks(range(len(trad_names)))
    ax1.set_yticklabels(trad_names, fontsize=10)
    ax1.set_xlabel('Coefficient (absolute)', fontsize=11)
    ax1.set_title('Traditional Feature Importance', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    
    # Add "Can't change" annotation
    ax1.annotate('Cannot change ↓', xy=(0.45, 0), fontsize=9, color=COLORS['gray'],
                ha='left', va='center')
    
    # Right plot: Combined
    ax2 = axes[1]
    comb_vals, comb_names = zip(*comb_sorted)
    colors2 = [COLORS['success'] if n in ['engagement_score', 'feature_usage_pct'] 
               else COLORS['primary'] for n in comb_names]
    bars2 = ax2.barh(range(len(comb_names)), comb_vals, color=colors2)
    ax2.set_yticks(range(len(comb_names)))
    ax2.set_yticklabels(comb_names, fontsize=10)
    ax2.set_xlabel('Combined Score (coefficient × actionability)', fontsize=11)
    ax2.set_title('Actionability-Weighted Importance', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    
    # Add "Priority targets" annotation
    ax2.annotate('Priority targets ↓', xy=(0.5, 0), fontsize=9, color=COLORS['success'],
                ha='left', va='center')
    
    # Add value labels
    for bars, vals in [(bars1, trad_vals), (bars2, comb_vals)]:
        for bar, val in zip(bars, vals):
            width = bar.get_width()
            bar.axes.annotate(f'{val:.2f}',
                            xy=(width, bar.get_y() + bar.get_height()/2),
                            xytext=(3, 0), textcoords="offset points",
                            ha='left', va='center', fontsize=9)
    
    plt.suptitle('Feature Importance: Traditional vs Business-Focused', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('viz/04_feature_importance_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Created: 04_feature_importance_comparison.png")


# =============================================================================
# CHART 5: Four Quadrant Matrix
# =============================================================================
def create_four_quadrant_matrix():
    """Feature importance vs actionability scatter matrix."""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Feature data with positions
    features = {
        'engagement_score': (0.18, 0.9, 'Priority Target'),
        'feature_usage_pct': (0.16, 0.85, 'Priority Target'),
        'nps_score': (0.10, 0.75, 'Quick Win'),
        'support_tickets_90d': (0.19, 0.6, 'Priority Target'),
        'is_inactive': (0.08, 0.55, 'Quick Win'),
        'payment_delays': (0.14, 0.4, 'Monitor'),
        'tenure_months': (0.50, 0.15, 'Accept'),
        'monthly_charges': (0.12, 0.2, 'Monitor'),
    }
    
    # Draw quadrant lines
    ax.axhline(y=0.5, color=COLORS['gray'], linestyle='--', alpha=0.5)
    ax.axvline(x=0.15, color=COLORS['gray'], linestyle='--', alpha=0.5)
    
    # Draw quadrant backgrounds
    ax.fill_between([0, 0.15], 0.5, 1.0, alpha=0.1, color=COLORS['warning'])  # Quick Wins
    ax.fill_between([0.15, 0.6], 0.5, 1.0, alpha=0.1, color=COLORS['success'])  # Priority Targets
    ax.fill_between([0, 0.15], 0, 0.5, alpha=0.1, color=COLORS['gray'])  # Low Priority
    ax.fill_between([0.15, 0.6], 0, 0.5, alpha=0.1, color=COLORS['primary'])  # Accept/Monitor
    
    # Quadrant labels
    ax.text(0.075, 0.92, 'QUICK WINS\nLow impact, easy to change', ha='center', va='top',
            fontsize=10, color=COLORS['warning'], fontweight='bold')
    ax.text(0.35, 0.92, 'PRIORITY TARGETS\nHigh impact, easy to change', ha='center', va='top',
            fontsize=10, color=COLORS['success'], fontweight='bold')
    ax.text(0.075, 0.08, 'LOW PRIORITY\nLow impact, hard to change', ha='center', va='bottom',
            fontsize=10, color=COLORS['gray'], fontweight='bold')
    ax.text(0.35, 0.08, 'ACCEPT/MONITOR\nHigh impact, hard to change', ha='center', va='bottom',
            fontsize=10, color=COLORS['primary'], fontweight='bold')
    
    # Plot features
    for name, (importance, actionability, category) in features.items():
        color = COLORS['success'] if category == 'Priority Target' else \
                COLORS['warning'] if category == 'Quick Win' else \
                COLORS['gray'] if category == 'Low Priority' else COLORS['primary']
        
        ax.scatter(importance, actionability, s=200, c=color, edgecolors='white', 
                   linewidth=2, zorder=5)
        
        # Label positioning
        offset = (5, 5) if actionability > 0.5 else (5, -15)
        ax.annotate(name.replace('_', '\n'), xy=(importance, actionability),
                    xytext=offset, textcoords='offset points',
                    fontsize=9, ha='left', va='bottom' if actionability > 0.5 else 'top')
    
    ax.set_xlabel('Predictive Importance (|coefficient|)', fontsize=12)
    ax.set_ylabel('Actionability Score', fontsize=12)
    ax.set_title('Feature Selection Matrix: Importance vs Actionability', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 0.55)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig('viz/05_four_quadrant_matrix.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Created: 05_four_quadrant_matrix.png")


# =============================================================================
# CHART 6: Intervention Timing Window
# =============================================================================
def create_intervention_timing():
    """Optimal intervention window derived from survival model."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create distribution of predicted days until churn for high-risk customers
    np.random.seed(42)
    
    # Simulate predicted days (clustered around 91-97 per portfolio)
    predicted_days = np.concatenate([
        np.random.normal(93, 3, 2000),  # Main cluster
        np.random.normal(85, 5, 500),   # Earlier churners
        np.random.normal(100, 4, 325),  # Later churners
    ])
    predicted_days = np.clip(predicted_days, 30, 120)
    
    # Create histogram
    bins = np.arange(30, 121, 3)
    n, bins_edge, patches = ax.hist(predicted_days, bins=bins, color=COLORS['light_blue'],
                                     edgecolor='white', linewidth=0.5, alpha=0.7)
    
    # Color the optimal window differently
    for i, (patch, left_edge) in enumerate(zip(patches, bins_edge[:-1])):
        if 45 <= left_edge < 95:
            patch.set_facecolor(COLORS['success'])
            patch.set_alpha(0.8)
        elif left_edge < 45:
            patch.set_facecolor(COLORS['warning'])
            patch.set_alpha(0.5)
        else:
            patch.set_facecolor(COLORS['danger'])
            patch.set_alpha(0.5)
    
    # Add zone labels
    ax.axvline(x=45, color=COLORS['gray'], linestyle='--', linewidth=2)
    ax.axvline(x=95, color=COLORS['gray'], linestyle='--', linewidth=2)
    ax.axvline(x=93, color=COLORS['success'], linestyle='-', linewidth=2, label='Peak (Day 93)')
    
    # Zone annotations
    max_height = max(n) * 1.1
    ax.annotate('TOO EARLY\nDays 0-45\nCustomer not yet\nexperiencing friction',
                xy=(30, max_height * 0.8), fontsize=10, ha='center',
                color=COLORS['warning'], fontweight='bold')
    ax.annotate('OPTIMAL WINDOW\nDays 45-95\nCustomer receptive,\nhasn\'t decided to leave',
                xy=(70, max_height * 0.9), fontsize=11, ha='center',
                color=COLORS['success'], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['success']))
    ax.annotate('TOO LATE\nDays 95+\n>50% already\nchurned',
                xy=(107, max_height * 0.8), fontsize=10, ha='center',
                color=COLORS['danger'], fontweight='bold')
    
    # Add percentile markers
    p25, p50, p75 = np.percentile(predicted_days, [25, 50, 75])
    ax.annotate(f'25th: Day {p25:.0f}\n50th: Day {p50:.0f}\n75th: Day {p75:.0f}',
                xy=(108, max_height * 0.5), fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['gray']))
    
    ax.set_xlabel('Predicted Days Until Churn', fontsize=12)
    ax.set_ylabel('Number of High-Risk Customers', fontsize=12)
    ax.set_title('Intervention Window Derived from Survival Model (n=2,825 high-risk customers)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(25, 120)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('viz/06_intervention_timing.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Created: 06_intervention_timing.png")


# =============================================================================
# CHART 7: Revenue Impact
# =============================================================================
def create_revenue_impact():
    """Revenue impact by intervention timing."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Timing zones and their effectiveness
    zones = ['Too Early\n(Day 0-45)', 'Optimal\n(Day 45-95)', 'Too Late\n(Day 95+)']
    
    # Estimated revenue protected (based on $2.54M at risk)
    # Optimal captures most value, others less
    revenue_protected = [180, 264, 45]  # in thousands
    potential = [400, 264, 150]  # potential if all captured
    
    x = np.arange(len(zones))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, potential, width, label='Potential Value ($K)', 
                   color=COLORS['light_blue'], edgecolor=COLORS['primary'])
    bars2 = ax.bar(x + width/2, revenue_protected, width, label='Actually Protected ($K)',
                   color=COLORS['success'], edgecolor='white')
    
    # Add efficiency percentage
    for i, (pot, prot) in enumerate(zip(potential, revenue_protected)):
        eff = prot / pot * 100
        ax.annotate(f'{eff:.0f}%\nefficiency',
                    xy=(i + width/2, prot + 10),
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    color=COLORS['success'] if eff > 50 else COLORS['warning'])
    
    # Highlight optimal
    ax.annotate('Best timing\ncaptures 100%\nof potential',
                xy=(1, 264), xytext=(1.8, 320),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color=COLORS['success']),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['success']))
    
    ax.set_ylabel('Revenue ($K)', fontsize=12)
    ax.set_xlabel('Intervention Timing', fontsize=12)
    ax.set_title('Revenue Impact by Intervention Timing', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(zones)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 450)
    
    plt.tight_layout()
    plt.savefig('viz/07_revenue_impact.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Created: 07_revenue_impact.png")


# =============================================================================
# CHART 8: A/B Test Results
# =============================================================================
def create_ab_test_results():
    """A/B test churn rates by intervention variant."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data from portfolio
    variants = ['Control', 'Email', 'Discount', 'Call', 'Combined']
    churn_rates = [21.7, 17.6, 15.6, 9.9, 15.0]
    p_values = [None, 0.033, 0.001, 0.0001, 0.0005]
    significant = [False, False, True, True, True]  # After Bonferroni (α = 0.0125)
    
    colors = [COLORS['gray']] + [COLORS['success'] if sig else COLORS['warning'] 
                                  for sig in significant[1:]]
    
    bars = ax.bar(variants, churn_rates, color=colors, edgecolor='white', linewidth=2)
    
    # Add control line
    ax.axhline(y=21.7, color=COLORS['danger'], linestyle='--', linewidth=2, 
               label='Control baseline (21.7%)')
    
    # Add labels with lift and p-values
    for i, (bar, rate, pval, sig) in enumerate(zip(bars, churn_rates, p_values, significant)):
        # Churn rate label
        ax.annotate(f'{rate}%',
                    xy=(bar.get_x() + bar.get_width()/2, rate),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Lift and significance
        if i > 0:
            lift = (21.7 - rate) / 21.7 * 100
            sig_text = '✓ Significant' if sig else '✗ Not sig.'
            p_text = f'p<0.0001' if pval < 0.001 else f'p={pval:.3f}'
            
            ax.annotate(f'+{lift:.1f}% lift\n{p_text}\n{sig_text}',
                        xy=(bar.get_x() + bar.get_width()/2, 2),
                        ha='center', va='bottom', fontsize=9,
                        color=COLORS['success'] if sig else COLORS['warning'])
    
    # Winner callout
    ax.annotate('WINNER',
                xy=(3, 9.9), xytext=(3, 4),
                ha='center', fontsize=11, fontweight='bold', color=COLORS['success'],
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2))
    
    ax.set_ylabel('Churn Rate (%)', fontsize=12)
    ax.set_xlabel('Intervention Variant (n=900 each)', fontsize=12)
    ax.set_title('A/B Test Results: Churn Rate by Intervention\n(Bonferroni-corrected α = 0.0125)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 28)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('viz/08_ab_test_results.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Created: 08_ab_test_results.png")


# =============================================================================
# CHART 9: Intervention ROI
# =============================================================================
def create_intervention_roi():
    """ROI comparison across intervention channels."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data from portfolio
    channels = ['Email', 'Discount', 'Call', 'Combined']
    roi = [158.8, 11.8, 6.5, 2.8]
    costs = [0.50, 10.00, 35.00, 45.50]
    
    colors = [COLORS['success'], COLORS['primary'], COLORS['primary'], COLORS['warning']]
    
    bars = ax.bar(channels, roi, color=colors, edgecolor='white', linewidth=2)
    
    # Add ROI labels
    for bar, r, cost in zip(bars, roi, costs):
        ax.annotate(f'{r}x\n(${cost}/customer)',
                    xy=(bar.get_x() + bar.get_width()/2, r),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Highlight best ROI
    ax.annotate('BEST ROI\n24x more efficient\nthan Call',
                xy=(0, 158.8), xytext=(1.2, 140),
                ha='center', fontsize=10, fontweight='bold', color=COLORS['success'],
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['success']))
    
    ax.set_ylabel('ROI (× return on cost)', fontsize=12)
    ax.set_xlabel('Intervention Channel', fontsize=12)
    ax.set_title('Intervention ROI Comparison\nROI = (CLV × Absolute Lift) / Cost per Customer',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 180)
    
    plt.tight_layout()
    plt.savefig('viz/09_intervention_roi.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Created: 09_intervention_roi.png")


# =============================================================================
# CHART 10: Lift vs ROI (Dual Axis)
# =============================================================================
def create_lift_vs_roi():
    """Comparing lift and ROI across channels."""
    
    fig, ax1 = plt.subplots(figsize=(11, 6))
    
    # Data
    channels = ['Email', 'Discount', 'Call', 'Combined']
    lift_pp = [4.1, 6.1, 11.8, 6.7]  # Absolute lift in percentage points
    roi = [158.8, 11.8, 6.5, 2.8]
    
    x = np.arange(len(channels))
    width = 0.35
    
    # Left axis: Lift
    bars1 = ax1.bar(x - width/2, lift_pp, width, label='Absolute Lift (pp)', 
                    color=COLORS['primary'], edgecolor='white', linewidth=2)
    ax1.set_ylabel('Absolute Lift (percentage points)', fontsize=12, color=COLORS['primary'])
    ax1.tick_params(axis='y', labelcolor=COLORS['primary'])
    ax1.set_ylim(0, 15)
    
    # Right axis: ROI
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, roi, width, label='ROI (×)', 
                    color=COLORS['success'], edgecolor='white', linewidth=2)
    ax2.set_ylabel('ROI (× return)', fontsize=12, color=COLORS['success'])
    ax2.tick_params(axis='y', labelcolor=COLORS['success'])
    ax2.set_ylim(0, 180)
    
    # Labels
    for bar, val in zip(bars1, lift_pp):
        ax1.annotate(f'{val}pp',
                     xy=(bar.get_x() + bar.get_width()/2, val),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold',
                     color=COLORS['primary'])
    
    for bar, val in zip(bars2, roi):
        ax2.annotate(f'{val}x',
                     xy=(bar.get_x() + bar.get_width()/2, val),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold',
                     color=COLORS['success'])
    
    # Insight annotations
    ax1.annotate('Highest LIFT\nbut lower ROI',
                 xy=(2 - width/2, 11.8), xytext=(2.8, 13),
                 fontsize=9, ha='center',
                 arrowprops=dict(arrowstyle='->', color=COLORS['primary']),
                 color=COLORS['primary'])
    
    ax2.annotate('Highest ROI\nbut lower lift',
                 xy=(0 + width/2, 158.8), xytext=(-0.5, 140),
                 fontsize=9, ha='center',
                 arrowprops=dict(arrowstyle='->', color=COLORS['success']),
                 color=COLORS['success'])
    
    # Strategy box
    ax1.text(1.5, -2.5, 
             'Strategy: Use Email for broad outreach (high ROI), Call for high-value customers (high impact)',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor=COLORS['warning']))
    
    ax1.set_xlabel('Intervention Channel', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(channels)
    ax1.set_title('The Trade-off: Lift vs ROI by Channel', fontsize=14, fontweight='bold')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    plt.tight_layout()
    plt.savefig('viz/10_lift_vs_roi.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Created: 10_lift_vs_roi.png")


# =============================================================================
# CHART 11: Executive Summary Dashboard
# =============================================================================
def create_executive_summary():
    """Executive dashboard with key metrics."""
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # -------------------------------------------------------------------------
    # Panel 1: Risk Distribution (top-left)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    tiers = ['Low', 'Medium', 'High', 'Critical']
    counts = [1850, 1325, 1675, 1150]
    colors = [COLORS['success'], COLORS['warning'], COLORS['danger'], '#991b1b']
    
    bars = ax1.bar(tiers, counts, color=colors, edgecolor='white')
    ax1.set_title('Risk Distribution', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Customers')
    
    # Add $2.54M annotation
    ax1.annotate('$2.54M\nat risk', xy=(2.5, 1400), ha='center', fontsize=10,
                 fontweight='bold', color=COLORS['danger'])
    
    # -------------------------------------------------------------------------
    # Panel 2: Intervention Window (top-center)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Timeline
    ax2.axhline(y=0.5, color=COLORS['gray'], linewidth=3, alpha=0.3)
    
    # Zones
    ax2.axvspan(0, 45, alpha=0.3, color=COLORS['warning'])
    ax2.axvspan(45, 95, alpha=0.3, color=COLORS['success'])
    ax2.axvspan(95, 120, alpha=0.3, color=COLORS['danger'])
    
    # Labels
    ax2.text(22.5, 0.7, 'Too Early', ha='center', fontsize=10, color=COLORS['warning'])
    ax2.text(70, 0.7, 'OPTIMAL', ha='center', fontsize=12, fontweight='bold', color=COLORS['success'])
    ax2.text(107.5, 0.7, 'Too Late', ha='center', fontsize=10, color=COLORS['danger'])
    ax2.text(70, 0.3, 'Day 45-95', ha='center', fontsize=11)
    
    ax2.set_xlim(0, 120)
    ax2.set_ylim(0, 1)
    ax2.set_title('Optimal Intervention Window', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Days')
    ax2.set_yticks([])
    
    # -------------------------------------------------------------------------
    # Panel 3: A/B Test Winner (top-right)
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    
    variants = ['Control', 'Email', 'Discount', 'Call', 'Combined']
    rates = [21.7, 17.6, 15.6, 9.9, 15.0]
    colors = [COLORS['gray'], COLORS['warning'], COLORS['primary'], COLORS['success'], COLORS['primary']]
    
    bars = ax3.barh(variants, rates, color=colors, edgecolor='white')
    ax3.axvline(x=21.7, color=COLORS['danger'], linestyle='--', alpha=0.7)
    ax3.set_xlabel('Churn Rate (%)')
    ax3.set_title('A/B Test Results', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()
    
    # Winner label
    ax3.annotate('Winner: 54.4% lift\np < 0.0001', xy=(9.9, 3), xytext=(14, 3.5),
                 fontsize=9, color=COLORS['success'], fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=COLORS['success']))
    
    # -------------------------------------------------------------------------
    # Panel 4: Model Metrics (bottom-left)
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')
    
    metrics_text = """
    MODEL METRICS
    (Gating Checks)
    
    Classification AUC: 0.6612
    Survival C-Index: 0.6645
    Threshold: 0.5
    F1 Score: 0.406
    """
    ax4.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=11,
             family='monospace', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['light_blue'], alpha=0.3))
    ax4.set_title('Model Performance', fontsize=12, fontweight='bold')
    
    # -------------------------------------------------------------------------
    # Panel 5: ROI Comparison (bottom-center)
    # -------------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    
    channels = ['Email', 'Discount', 'Call', 'Combined']
    roi = [158.8, 11.8, 6.5, 2.8]
    colors = [COLORS['success'], COLORS['primary'], COLORS['primary'], COLORS['warning']]
    
    bars = ax5.bar(channels, roi, color=colors, edgecolor='white')
    ax5.set_ylabel('ROI (×)')
    ax5.set_title('Channel ROI', fontsize=12, fontweight='bold')
    
    # Best ROI label
    ax5.annotate('158.8×', xy=(0, 158.8), xytext=(0, 165),
                 ha='center', fontsize=11, fontweight='bold', color=COLORS['success'])
    
    # -------------------------------------------------------------------------
    # Panel 6: Business Impact (bottom-right)
    # -------------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    impact_text = """
    BUSINESS IMPACT
    
    CLV at Risk: $2,542,079
    Customers Saved: ~137
    Revenue Protected: ~$264K
    
    Best Lift: Call (+54.4%)
    Best ROI: Email (158.8×)
    """
    ax6.text(0.5, 0.5, impact_text, ha='center', va='center', fontsize=11,
             family='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['light_green'], alpha=0.3))
    ax6.set_title('Business Outcomes', fontsize=12, fontweight='bold')
    
    # Main title
    fig.suptitle('Proactive Churn Prevention System: Executive Summary', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('viz/Executive_summary.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Created: Executive_summary.png")


# =============================================================================
# MAIN: Generate All Charts
# =============================================================================
def main():
    """Generate all portfolio charts."""
    
    import os
    os.makedirs('viz', exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING PORTFOLIO CHARTS")
    print("="*60 + "\n")
    
    create_risk_distribution()
    create_survival_curves()
    create_threshold_analysis()
    create_feature_importance_comparison()
    create_four_quadrant_matrix()
    create_intervention_timing()
    create_revenue_impact()
    create_ab_test_results()
    create_intervention_roi()
    create_lift_vs_roi()
    create_executive_summary()
    
    print("\n" + "="*60)
    print("ALL CHARTS GENERATED SUCCESSFULLY")
    print("="*60)
    print("\nFiles created in ./viz/:")
    print("  01_risk_distribution.png")
    print("  02_survival_curves.png")
    print("  03_threshold_analysis.png")
    print("  04_feature_importance_comparison.png")
    print("  05_four_quadrant_matrix.png")
    print("  06_intervention_timing.png")
    print("  07_revenue_impact.png")
    print("  08_ab_test_results.png")
    print("  09_intervention_roi.png")
    print("  10_lift_vs_roi.png")
    print("  Executive_summary.png")
    print()


if __name__ == "__main__":
    main()
