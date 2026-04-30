#!/usr/bin/env python3
"""
生成论文图表 - 整合所有实验结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import json

plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# ============================================================================
# 加载数据
# ============================================================================
print("加载实验数据...")

DATA_PATH = "data/raw/Steel_industry_data.csv"
RESULTS_DIR = "results"

# 读取训练数据和测试数据
df = pd.read_csv(DATA_PATH)
features = ['Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh', 'CO2(tCO2)', 'Load_Type']
data = df[features].copy()

from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_features = train_data[features].copy()

# 加载合成数据
synthetic_tvae = pd.read_csv("results/synthetic_tvae.csv")
synthetic_ctgan = pd.read_csv("results/synthetic_ctgan.csv")
synthetic_conditional = pd.read_csv("results/synthetic_conditional.csv")

# ============================================================================
# Figure 1: KDE分布对比图 (合并为一个图)
# ============================================================================
print("生成 Figure 1: KDE分布对比...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

variables = ['Usage_kWh', 'CO2(tCO2)', 'Lagging_Current_Reactive.Power_kVarh']

for idx, var in enumerate(variables):
    # TVAE对比
    ax = axes[0, idx]
    sns.kdeplot(train_features[var], ax=ax, label='Real', color='#2E86AB', linewidth=2.5, linestyle='-')
    sns.kdeplot(synthetic_tvae[var], ax=ax, label='TVAE', color='#F18F01', linewidth=2.5, linestyle='--')
    ax.set_title(f'{var}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_xlabel('')

    # CTGAN对比
    ax = axes[1, idx]
    sns.kdeplot(train_features[var], ax=ax, label='Real', color='#2E86AB', linewidth=2.5, linestyle='-')
    sns.kdeplot(synthetic_ctgan[var], ax=ax, label='CTGAN', color='#C73E1D', linewidth=2.5, linestyle='--')
    ax.set_title(f'{var}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_xlabel('')

# 添加行标签
axes[0, 0].set_ylabel('Density')
axes[1, 0].set_ylabel('Density')
axes[0, 0].text(-0.25, 0.5, 'TVAE', transform=axes[0, 0].transAxes, fontsize=14, fontweight='bold',
                rotation=90, va='center', ha='center')
axes[1, 0].text(-0.25, 0.5, 'CTGAN', transform=axes[1, 0].transAxes, fontsize=14, fontweight='bold',
                rotation=90, va='center', ha='center')

plt.suptitle('Figure 1: Variable Distribution Comparison - Real vs Synthetic Data', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/figure1_kde_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/figure1_kde_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1 saved!")

# ============================================================================
# Figure 2: 相关性热力图
# ============================================================================
print("生成 Figure 2: 相关性热力图...")

corr_vars = ['Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh', 'CO2(tCO2)']
display_labels = ['Usage', 'Reactive', 'CO2']
base_matrices = {
    'Real': train_features[corr_vars].corr(),
    'TVAE': synthetic_tvae[corr_vars].corr(),
    'CTGAN': synthetic_ctgan[corr_vars].corr(),
}
diff_matrices = {
    'Real': base_matrices['Real'] - base_matrices['Real'],
    'TVAE': base_matrices['TVAE'] - base_matrices['Real'],
    'CTGAN': base_matrices['CTGAN'] - base_matrices['Real'],
}

fig = plt.figure(figsize=(11.4, 6.8))
gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 0.06], hspace=0.28, wspace=0.22)
axes_top = [fig.add_subplot(gs[0, i]) for i in range(3)]
axes_bottom = [fig.add_subplot(gs[1, i]) for i in range(3)]
cbar_top_ax = fig.add_subplot(gs[0, 3])
cbar_bottom_ax = fig.add_subplot(gs[1, 3])

mask = np.triu(np.ones((len(corr_vars), len(corr_vars)), dtype=bool), k=1)
heatmap_style = dict(
    mask=mask,
    annot=True,
    fmt='.3f',
    cmap='RdBu_r',
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.6,
    square=True,
    annot_kws={'fontsize': 10},
    cbar=False,
    xticklabels=display_labels,
    yticklabels=display_labels
)

diff_style = dict(
    mask=mask,
    annot=True,
    fmt='.3f',
    cmap='PuOr',
    center=0,
    vmin=-0.12,
    vmax=0.12,
    linewidths=0.6,
    square=True,
    annot_kws={'fontsize': 10},
    cbar=False,
    xticklabels=display_labels,
    yticklabels=display_labels
)

for idx, panel_label in enumerate(['Real', 'TVAE', 'CTGAN']):
    ax = axes_top[idx]
    sns.heatmap(base_matrices[panel_label], ax=ax, **heatmap_style)
    ax.set_xlabel(panel_label, fontsize=11, labelpad=8)
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelrotation=0, labelsize=10)
    ax.tick_params(axis='y', labelrotation=0, labelsize=10)
    if idx > 0:
        ax.set_yticklabels([])

for idx, panel_label in enumerate(['Real', 'TVAE', 'CTGAN']):
    ax = axes_bottom[idx]
    sns.heatmap(diff_matrices[panel_label], ax=ax, **diff_style)
    ax.set_xlabel(panel_label, fontsize=11, labelpad=8)
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelrotation=0, labelsize=10)
    ax.tick_params(axis='y', labelrotation=0, labelsize=10)
    if idx > 0:
        ax.set_yticklabels([])

axes_top[0].text(-0.32, 0.5, 'Correlation', transform=axes_top[0].transAxes,
                 fontsize=11, fontweight='bold', rotation=90, va='center', ha='center')
axes_bottom[0].text(-0.32, 0.5, 'Delta vs Real', transform=axes_bottom[0].transAxes,
                    fontsize=11, fontweight='bold', rotation=90, va='center', ha='center')

fig.colorbar(axes_top[-1].collections[0], cax=cbar_top_ax)
fig.colorbar(axes_bottom[-1].collections[0], cax=cbar_bottom_ax)
cbar_top_ax.tick_params(labelsize=10)
cbar_bottom_ax.tick_params(labelsize=10)

plt.savefig('results/figure2_correlation_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/figure2_correlation_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()
print("Figure 2 saved!")

# ============================================================================
# Figure 3: TSTR效用对比柱状图
# ============================================================================
print("生成 Figure 3: TSTR效用对比...")

fig, ax = plt.subplots(figsize=(6.6, 4.2))

models = ['Real', 'TVAE', 'CTGAN']
accs = [0.7205, 0.6665, 0.6553]
f1s = [0.7175, 0.6655, 0.6586]

x = np.arange(len(models))
width = 0.24

bars1 = ax.bar(x - width/2, accs, width, label='Accuracy', color='#2E86AB', edgecolor='black', linewidth=1)
bars2 = ax.bar(x + width/2, f1s, width, label='F1-score', color='#F18F01', edgecolor='black', linewidth=1)

ax.set_ylabel('Score', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.tick_params(axis='y', labelsize=11)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False, fontsize=10)
ax.set_ylim(0, 1.0)
ax.set_xlim(-0.45, 2.45)
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, linewidth=1.2)
ax.text(2.43, 0.905, '0.90', color='red', fontsize=9, va='bottom', ha='right')

for bar in list(bars1) + list(bars2):
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.tight_layout(pad=0.7)
plt.savefig('results/figure3_tstr_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/figure3_tstr_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print("Figure 3 saved!")

# ============================================================================
# Figure 4: DCR隐私分布图
# ============================================================================
print("生成 Figure 4: DCR隐私分布...")

from scipy.spatial.distance import cdist

features_for_dcr = ['Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh', 'CO2(tCO2)']
real_features = train_features[features_for_dcr].values

# 计算TVAE DCR
tvae_features = synthetic_tvae[features_for_dcr].values
distances_tvae = cdist(tvae_features, real_features, metric='euclidean')
dcr_tvae = distances_tvae.min(axis=1)

# 计算CTGAN DCR
ctgan_features = synthetic_ctgan[features_for_dcr].values
distances_ctgan = cdist(ctgan_features, real_features, metric='euclidean')
dcr_ctgan = distances_ctgan.min(axis=1)

fig, axes = plt.subplots(2, 1, figsize=(6.8, 8.0), sharex=True)

# TVAE DCR
ax = axes[0]
ax.hist(dcr_tvae, bins=50, alpha=0.75, color='#F18F01', edgecolor='black')
ax.axvline(x=0, color='red', linestyle='--', linewidth=1.4)
ax.axvline(x=dcr_tvae.mean(), color='blue', linestyle='-', linewidth=1.6)
ax.set_ylabel('Frequency', fontsize=12)
ax.tick_params(axis='both', labelsize=10)
ax.text(-0.16, 0.5, 'TVAE', transform=ax.transAxes, fontsize=11, fontweight='bold',
        rotation=90, va='center', ha='center')
ax.text(0.97, 0.90, f'Mean={dcr_tvae.mean():.3f}\nDCR>0: {(dcr_tvae>0).mean()*100:.1f}%', transform=ax.transAxes,
        fontsize=10, va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='0.7'))

# CTGAN DCR
ax = axes[1]
ax.hist(dcr_ctgan, bins=50, alpha=0.75, color='#C73E1D', edgecolor='black')
ax.axvline(x=0, color='red', linestyle='--', linewidth=1.4)
ax.axvline(x=dcr_ctgan.mean(), color='blue', linestyle='-', linewidth=1.6)
ax.set_xlabel('Distance to Closest Record (DCR)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.tick_params(axis='both', labelsize=10)
ax.text(-0.16, 0.5, 'CTGAN', transform=ax.transAxes, fontsize=11, fontweight='bold',
        rotation=90, va='center', ha='center')
ax.text(0.97, 0.90, f'Mean={dcr_ctgan.mean():.3f}\nDCR>0: {(dcr_ctgan>0).mean()*100:.1f}%', transform=ax.transAxes,
        fontsize=10, va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='0.7'))

plt.tight_layout(pad=0.9)
plt.savefig('results/figure4_dcr_distribution.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/figure4_dcr_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print("Figure 4 saved!")

# ============================================================================
# Figure 5: 条件生成 - Load_Type分布对比
# ============================================================================
print("生成 Figure 5: 条件生成Load_Type分布...")

fig, axes = plt.subplots(3, 1, figsize=(6.6, 9.8), sharex=True)

distribution_panels = [
    ('Real', train_features['Load_Type'].value_counts().sort_index()),
    ('Conditional', synthetic_conditional['Load_Type'].value_counts().sort_index()),
    ('Random TVAE', synthetic_tvae['Load_Type'].value_counts().sort_index()),
]
colors = ['#2E86AB', '#F18F01', '#C73E1D']
max_count = max(counts.max() for _, counts in distribution_panels) * 1.15

for idx, (panel_label, counts) in enumerate(distribution_panels):
    ax = axes[idx]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='black', linewidth=1)
    ax.set_ylabel('Count', fontsize=12)
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='x', labelsize=11, rotation=0)
    ax.set_ylim(0, max_count)
    ax.text(-0.16, 0.5, panel_label, transform=ax.transAxes, fontsize=12, fontweight='bold',
            rotation=90, va='center', ha='center')

    for bar, val in zip(bars, counts.values):
        ax.annotate(f'{val:,}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=10)

axes[-1].set_xlabel('Load Type', fontsize=12)
plt.tight_layout()
plt.savefig('results/figure5_load_type_distribution.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/figure5_load_type_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print("Figure 5 saved!")

# ============================================================================
# Figure 6: 条件生成摘要表
# ============================================================================
print("生成 Figure 6: 条件生成摘要表...")

summary_rows = []
for load_type in ['Light_Load', 'Medium_Load', 'Maximum_Load']:
    real_subset = train_features[train_features['Load_Type'] == load_type]
    synth_subset = synthetic_conditional[synthetic_conditional['Load_Type'] == load_type]
    summary_rows.append({
        'Load Type': load_type.replace('_', ' '),
        'Real Count': len(real_subset),
        'Synthetic Count': len(synth_subset),
        'Real Corr': real_subset[['Usage_kWh', 'CO2(tCO2)']].corr().iloc[0, 1],
        'Synthetic Corr': synth_subset[['Usage_kWh', 'CO2(tCO2)']].corr().iloc[0, 1],
    })

summary_df = pd.DataFrame(summary_rows)
summary_df['Real Corr'] = summary_df['Real Corr'].map(lambda x: f'{x:.4f}')
summary_df['Synthetic Corr'] = summary_df['Synthetic Corr'].map(lambda x: f'{x:.4f}')
summary_df.to_csv('results/figure6_conditional_summary.csv', index=False)
print("Figure 6 summary saved!")

# ============================================================================
# Figure 7: 综合结果表格图
# ============================================================================
print("跳过 Figure 7: 论文最终版本改为使用 LaTeX 表格，不再生成图片表格。")

# ==========================================================================
# Figure 8: 条件生成各类型相关性保持
# ==========================================================================
print("生成 Figure 8: 各Load_Type相关性保持...")

fig, ax = plt.subplots(figsize=(10, 6))

load_types = ['Light_Load', 'Medium_Load', 'Maximum_Load']
real_corrs = []
synth_corrs = []

for lt in load_types:
    real_lt = train_features[train_features['Load_Type'] == lt]
    synth_lt = synthetic_conditional[synthetic_conditional['Load_Type'] == lt]
    real_corrs.append(real_lt[['Usage_kWh', 'CO2(tCO2)']].corr().iloc[0, 1])
    synth_corrs.append(synth_lt[['Usage_kWh', 'CO2(tCO2)']].corr().iloc[0, 1])

x = np.arange(len(load_types))
width = 0.35

bars1 = ax.bar(x - width/2, real_corrs, width, label='Real Data', color='#2E86AB', edgecolor='black')
bars2 = ax.bar(x + width/2, synth_corrs, width, label='Conditional Generated', color='#F18F01', edgecolor='black')

ax.set_ylabel('Usage-CO2 Correlation', fontsize=12)
ax.set_title('Figure 8: Physics Coupling Preservation by Load_Type', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(load_types)
ax.legend()
ax.set_ylim(0.9, 1.01)
ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='95% threshold')

for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('results/figure8_correlation_by_load_type.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/figure8_correlation_by_load_type.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 8 saved!")

# ============================================================================
# 保存所有图表清单
# ============================================================================
print("\n" + "="*60)
print("所有论文图表已生成!")
print("="*60)
print("\n图表清单:")
figures = [
    ("figure1_kde_comparison.pdf/png", "变量分布KDE对比 (TVAE vs CTGAN)"),
    ("figure2_correlation_heatmap.pdf/png", "相关性热力图对比"),
    ("figure3_tstr_comparison.pdf/png", "TSTR机器学习效用对比"),
    ("figure4_dcr_distribution.pdf/png", "DCR隐私分布图"),
    ("figure5_load_type_distribution.pdf/png", "Load_Type分布对比"),
    ("figure6_max_load_kde.pdf/png", "Maximum_Load条件生成KDE"),
    ("figure8_correlation_by_load_type.pdf/png", "各Load_Type相关性保持"),
]
for fname, desc in figures:
    print(f"  - {fname}: {desc}")

print("\n所有文件保存在: results/")