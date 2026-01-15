"""
Comprehensive Visualization Code for Flight Delay Analysis
Add this to your Jupyter notebook after the EDA sections
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# ============================================================================
# VISUALIZATION 1: Delay Distribution
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

df = pd.read_csv("/Users/nikitha/Documents/flight-delay-prediction-ice/data/Flight Delay Dataset.csv")
# Histogram of delay minutes
axes[0].hist(df['ARR_DELAY'], bins=100, edgecolor='black', alpha=0.7)
axes[0].axvline(x=15, color='red', linestyle='--', linewidth=2, label='15 min threshold')
axes[0].set_xlabel('Arrival Delay (minutes)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Distribution of Flight Delays', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].set_xlim(-60, 200)

# Box plot by delay category
delay_cats = pd.cut(df['ARR_DELAY'], 
                    bins=[-np.inf, 0, 15, 60, np.inf],
                    labels=['Early', 'On-time', 'Minor Delay', 'Major Delay'])
delay_summary = delay_cats.value_counts()
axes[1].bar(range(len(delay_summary)), delay_summary.values, 
            color=['green', 'blue', 'orange', 'red'], alpha=0.7, edgecolor='black')
axes[1].set_xticks(range(len(delay_summary)))
axes[1].set_xticklabels(delay_summary.index, rotation=0)
axes[1].set_ylabel('Number of Flights', fontsize=12)
axes[1].set_title('Flights by Delay Category', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================================
# VISUALIZATION 2: Temporal Patterns
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Day of week pattern
dow_data = df.groupby('DAY_OF_WEEK')['IS_DELAYED'].mean() * 100
dow_names = {1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat', 7: 'Sun'}
dow_data.index = dow_data.index.map(dow_names)
axes[0, 0].bar(dow_data.index, dow_data.values, 
               color=['#3498db' if x < 25 else '#e74c3c' for x in dow_data.values],
               edgecolor='black', alpha=0.8)
axes[0, 0].axhline(y=df['IS_DELAYED'].mean()*100, color='green', 
                    linestyle='--', label='Average', linewidth=2)
axes[0, 0].set_ylabel('Delay Rate (%)', fontsize=12)
axes[0, 0].set_title('Delay Rate by Day of Week', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# Month pattern
month_data = df.groupby('MONTH')['IS_DELAYED'].mean() * 100
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
axes[0, 1].plot(range(1, 13), month_data.values, marker='o', 
                linewidth=3, markersize=10, color='#e74c3c')
axes[0, 1].fill_between(range(1, 13), month_data.values, alpha=0.3, color='#e74c3c')
axes[0, 1].set_xticks(range(1, 13))
axes[0, 1].set_xticklabels(month_names, rotation=45)
axes[0, 1].set_ylabel('Delay Rate (%)', fontsize=12)
axes[0, 1].set_title('Seasonal Delay Patterns', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Hour of day pattern
hour_data = df.groupby('HOUR')['IS_DELAYED'].mean() * 100
colors = ['#2ecc71' if h < 12 else '#f39c12' if h < 18 else '#e74c3c' 
          for h in hour_data.index]
axes[1, 0].bar(hour_data.index, hour_data.values, color=colors, 
               edgecolor='black', alpha=0.8)
axes[1, 0].set_xlabel('Hour of Day', fontsize=12)
axes[1, 0].set_ylabel('Delay Rate (%)', fontsize=12)
axes[1, 0].set_title('Delay Rate by Hour of Day', fontsize=14, fontweight='bold')
axes[1, 0].axhline(y=df['IS_DELAYED'].mean()*100, color='black', 
                    linestyle='--', label='Average', linewidth=2)
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# Heatmap: Day of week vs Hour
pivot_data = df.pivot_table(values='IS_DELAYED', 
                             index='DAY_OF_WEEK', 
                             columns='HOUR', 
                             aggfunc='mean') * 100
sns.heatmap(pivot_data, annot=False, cmap='RdYlGn_r', cbar_kws={'label': 'Delay Rate (%)'},
            ax=axes[1, 1], vmin=10, vmax=40)
axes[1, 1].set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
axes[1, 1].set_xlabel('Hour of Day', fontsize=12)
axes[1, 1].set_ylabel('Day of Week', fontsize=12)
axes[1, 1].set_title('Delay Patterns: Day vs Hour Heatmap', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================================
# VISUALIZATION 3: Geographic Analysis
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Top airports by delay rate (origin)
origin_delays = df.groupby('ORIGIN').agg({
    'IS_DELAYED': 'mean',
    'ORIGIN_CITY_NAME': 'first'
}).reset_index()
origin_delays = origin_delays[df.groupby('ORIGIN').size() >= 500]  # Min 500 flights
origin_delays['IS_DELAYED'] *= 100
origin_delays = origin_delays.sort_values('IS_DELAYED', ascending=False).head(15)

axes[0].barh(range(len(origin_delays)), origin_delays['IS_DELAYED'].values,
             color=['#e74c3c' if x > 30 else '#f39c12' if x > 20 else '#3498db' 
                    for x in origin_delays['IS_DELAYED'].values],
             edgecolor='black', alpha=0.8)
axes[0].set_yticks(range(len(origin_delays)))
axes[0].set_yticklabels([f"{row['ORIGIN']} ({row['ORIGIN_CITY_NAME'].split(',')[0]})" 
                         for _, row in origin_delays.iterrows()])
axes[0].set_xlabel('Delay Rate (%)', fontsize=12)
axes[0].set_title('Top 15 Airports by Delay Rate (Origin)', fontsize=14, fontweight='bold')
axes[0].axvline(x=df['IS_DELAYED'].mean()*100, color='green', 
                linestyle='--', label='Average', linewidth=2)
axes[0].legend()
axes[0].grid(axis='x', alpha=0.3)

# State-level analysis
state_delays = df.groupby('ORIGIN_STATE_ABR').agg({
    'IS_DELAYED': 'mean',
    'ORIGIN': 'count'
}).reset_index()
state_delays.columns = ['State', 'Delay_Rate', 'Flight_Count']
state_delays = state_delays[state_delays['Flight_Count'] >= 200]
state_delays['Delay_Rate'] *= 100
state_delays = state_delays.sort_values('Delay_Rate', ascending=False).head(15)

axes[1].barh(range(len(state_delays)), state_delays['Delay_Rate'].values,
             color='#3498db', edgecolor='black', alpha=0.8)
axes[1].set_yticks(range(len(state_delays)))
axes[1].set_yticklabels(state_delays['State'].values)
axes[1].set_xlabel('Delay Rate (%)', fontsize=12)
axes[1].set_title('Top 15 States by Delay Rate', fontsize=14, fontweight='bold')
axes[1].axvline(x=df['IS_DELAYED'].mean()*100, color='green', 
                linestyle='--', label='Average', linewidth=2)
axes[1].legend()
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# VISUALIZATION 4: Carrier Performance
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Carrier delay rates
carrier_perf = df.groupby('UNIQUE_CARRIER').agg({
    'IS_DELAYED': 'mean',
    'ORIGIN': 'count',
    'ARR_DELAY': 'mean'
}).reset_index()
carrier_perf.columns = ['Carrier', 'Delay_Rate', 'Flight_Count', 'Avg_Delay']
carrier_perf = carrier_perf[carrier_perf['Flight_Count'] >= 100]
carrier_perf['Delay_Rate'] *= 100
carrier_perf = carrier_perf.sort_values('Delay_Rate', ascending=False)

colors_carrier = ['#e74c3c' if x > 30 else '#f39c12' if x > 25 else '#2ecc71' 
                  for x in carrier_perf['Delay_Rate'].values]
axes[0].bar(range(len(carrier_perf)), carrier_perf['Delay_Rate'].values,
            color=colors_carrier, edgecolor='black', alpha=0.8)
axes[0].set_xticks(range(len(carrier_perf)))
axes[0].set_xticklabels(carrier_perf['Carrier'].values, rotation=45)
axes[0].set_ylabel('Delay Rate (%)', fontsize=12)
axes[0].set_title('Carrier Performance Comparison', fontsize=14, fontweight='bold')
axes[0].axhline(y=df['IS_DELAYED'].mean()*100, color='blue', 
                linestyle='--', label='Average', linewidth=2)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Scatter: Delay rate vs avg delay minutes
axes[1].scatter(carrier_perf['Delay_Rate'], carrier_perf['Avg_Delay'], 
                s=carrier_perf['Flight_Count']/10, alpha=0.6, 
                c=carrier_perf['Delay_Rate'], cmap='RdYlGn_r', edgecolor='black')
for _, row in carrier_perf.iterrows():
    axes[1].annotate(row['Carrier'], 
                     (row['Delay_Rate'], row['Avg_Delay']),
                     fontsize=10, fontweight='bold')
axes[1].set_xlabel('Delay Rate (%)', fontsize=12)
axes[1].set_ylabel('Average Delay (minutes)', fontsize=12)
axes[1].set_title('Carrier Performance: Rate vs Duration', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# VISUALIZATION 5: Distance Analysis
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Distance groups
distance_data = df.groupby('DISTANCE_GROUP').agg({
    'IS_DELAYED': 'mean',
    'ORIGIN': 'count'
}).reset_index()
distance_data.columns = ['Distance_Group', 'Delay_Rate', 'Flight_Count']
distance_data['Delay_Rate'] *= 100
distance_data['Distance_Range'] = distance_data['Distance_Group'] * 250

axes[0].plot(distance_data['Distance_Range'], distance_data['Delay_Rate'], 
             marker='o', linewidth=3, markersize=10, color='#e74c3c')
axes[0].fill_between(distance_data['Distance_Range'], distance_data['Delay_Rate'], 
                      alpha=0.3, color='#e74c3c')
axes[0].set_xlabel('Distance (miles)', fontsize=12)
axes[0].set_ylabel('Delay Rate (%)', fontsize=12)
axes[0].set_title('Delay Rate by Flight Distance', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Scatter: Distance vs delay
sample_data = df.sample(n=min(5000, len(df)), random_state=42)
scatter = axes[1].scatter(sample_data['DISTANCE'], sample_data['ARR_DELAY'], 
                          c=sample_data['IS_DELAYED'], cmap='RdYlGn_r', 
                          alpha=0.3, s=20)
axes[1].set_xlabel('Flight Distance (miles)', fontsize=12)
axes[1].set_ylabel('Arrival Delay (minutes)', fontsize=12)
axes[1].set_title('Distance vs Delay (5K sample)', fontsize=14, fontweight='bold')
axes[1].axhline(y=15, color='red', linestyle='--', label='15 min threshold', linewidth=2)
axes[1].legend()
axes[1].set_ylim(-50, 150)
plt.colorbar(scatter, ax=axes[1], label='Delayed (1) or On-time (0)')

plt.tight_layout()
plt.show()

# ============================================================================
# VISUALIZATION 6: Feature Importance (After Model Training)
# ============================================================================

# This goes after you train the XGBoost model

fig, ax = plt.subplots(figsize=(12, 8))

# Get top 20 features
top_features = feature_importance.head(20)

colors_importance = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
ax.barh(range(len(top_features)), top_features['importance'].values,
        color=colors_importance, edgecolor='black', alpha=0.8)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'].values)
ax.set_xlabel('Feature Importance Score', fontsize=12)
ax.set_title('Top 20 Features - XGBoost Model', fontsize=16, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# VISUALIZATION 7: Model Comparison
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Performance comparison bar chart
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
models = ['Logistic Regression', 'Random Forest', 'XGBoost']

x = np.arange(len(metrics))
width = 0.25

axes[0].bar(x - width, comparison_df.loc['Logistic Regression'].values, 
            width, label='Logistic Regression', color='#3498db', alpha=0.8, edgecolor='black')
axes[0].bar(x, comparison_df.loc['Random Forest'].values, 
            width, label='Random Forest', color='#2ecc71', alpha=0.8, edgecolor='black')
axes[0].bar(x + width, comparison_df.loc['XGBoost'].values, 
            width, label='XGBoost', color='#e74c3c', alpha=0.8, edgecolor='black')

axes[0].set_ylabel('Score', fontsize=12)
axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics, rotation=45, ha='right')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_ylim(0.7, 1.0)

# ROC Curves
from sklearn.metrics import roc_curve

fpr_lr, tpr_lr, _ = roc_curve(y_val, y_pred_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_val, y_pred_proba_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_val, y_pred_proba_xgb)

axes[1].plot(fpr_lr, tpr_lr, label=f'Logistic Reg (AUC={lr_metrics["roc_auc"]:.3f})', 
             linewidth=2, color='#3498db')
axes[1].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={rf_metrics["roc_auc"]:.3f})', 
             linewidth=2, color='#2ecc71')
axes[1].plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC={xgb_metrics["roc_auc"]:.3f})', 
             linewidth=3, color='#e74c3c')
axes[1].plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)

axes[1].set_xlabel('False Positive Rate', fontsize=12)
axes[1].set_ylabel('True Positive Rate', fontsize=12)
axes[1].set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
axes[1].legend(loc='lower right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# VISUALIZATION 8: Confusion Matrix (Test Set)
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 6))

cm = confusion_matrix(y_test, y_pred_test)
cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm_display, annot=cm, fmt='d', cmap='Blues', 
            xticklabels=['On-Time', 'Delayed'],
            yticklabels=['On-Time', 'Delayed'],
            cbar_kws={'label': 'Proportion'}, ax=ax)
ax.set_ylabel('True Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_title('Confusion Matrix - XGBoost (Test Set)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n✅ All visualizations generated successfully!")
print("Add these to your notebook for a comprehensive visual analysis.")