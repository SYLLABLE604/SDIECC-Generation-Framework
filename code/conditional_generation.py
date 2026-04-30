#!/usr/bin/env python3
"""
条件生成实验：针对Maximum_Load等特定工况的按需生成
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import json

from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer
from sdv.sampling import Condition

# Configuration
DATA_PATH = "data/raw/Steel_industry_data.csv"
OUTPUT_DIR = "results"
RANDOM_STATE = 42

# ============================================================================
# 1. 数据预处理
# ============================================================================
print("=" * 60)
print("1. 数据预处理")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
features = ['Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh', 'CO2(tCO2)', 'Load_Type']
data = df[features].copy()

# 数据划分
train_data, test_data = train_test_split(data, test_size=0.2, random_state=RANDOM_STATE)
print(f"训练集: {len(train_data)}, 测试集: {len(test_data)}")

# ============================================================================
# 2. 训练TVAE模型
# ============================================================================
print("\n" + "=" * 60)
print("2. 训练TVAE模型")
print("=" * 60)

train_features = train_data[features].copy()

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(train_features)

tvae_model = TVAESynthesizer(metadata, epochs=100, batch_size=200, verbose=True)
tvae_model.fit(train_features)
print("TVAE模型训练完成!")

# ============================================================================
# 3. 条件生成实验
# ============================================================================
print("\n" + "=" * 60)
print("3. 条件生成实验")
print("=" * 60)

# 3.1 统计真实数据各Load_Type分布
print("\n真实数据Load_Type分布:")
real_counts = train_features['Load_Type'].value_counts()
print(real_counts)

# 分别生成各Load_Type的合成数据
load_types = ['Light_Load', 'Medium_Load', 'Maximum_Load']
synthetic_by_type = {}

for lt in load_types:
    real_count = len(train_features[train_features['Load_Type'] == lt])
    print(f"\n--- 条件生成 {lt} (目标: {real_count} 条) ---")

    # 创建条件
    condition = Condition(
        column_values={'Load_Type': lt},
        num_rows=real_count  # 生成与真实数据相同数量
    )

    # 条件生成
    synthetic_lt = tvae_model.sample_from_conditions(conditions=[condition])

    # 验证生成的Load_Type分布
    generated_lt_counts = synthetic_lt['Load_Type'].value_counts()
    print(f"生成数据Load_Type分布: {dict(generated_lt_counts)}")

    # 统计对比
    real_lt = train_features[train_features['Load_Type'] == lt]
    print(f"  Usage_kWh - Real Mean: {real_lt['Usage_kWh'].mean():.2f}, Generated Mean: {synthetic_lt['Usage_kWh'].mean():.2f}")
    print(f"  CO2 - Real Mean: {real_lt['CO2(tCO2)'].mean():.4f}, Generated Mean: {synthetic_lt['CO2(tCO2)'].mean():.4f}")

    # 计算相关性
    real_corr = real_lt[['Usage_kWh', 'CO2(tCO2)']].corr().iloc[0, 1]
    synth_corr = synthetic_lt[['Usage_kWh', 'CO2(tCO2)']].corr().iloc[0, 1]
    print(f"  Usage-CO2相关性 - Real: {real_corr:.4f}, Generated: {synth_corr:.4f}")

    synthetic_by_type[lt] = synthetic_lt

# ============================================================================
# 4. 合并条件生成数据并评估
# ============================================================================
print("\n" + "=" * 60)
print("4. 合并条件生成数据并评估")
print("=" * 60)

# 合并所有条件生成的数据
synthetic_conditional = pd.concat([synthetic_by_type[lt] for lt in load_types], ignore_index=True)
print(f"\n条件生成总数据量: {len(synthetic_conditional)}")
print(f"条件生成Load_Type分布:\n{synthetic_conditional['Load_Type'].value_counts()}")

# 保存条件生成数据
synthetic_conditional.to_csv(f"{OUTPUT_DIR}/synthetic_conditional.csv", index=False)
print(f"条件生成数据已保存: {OUTPUT_DIR}/synthetic_conditional.csv")

# 对比：非条件生成（随机采样）
print("\n对比：非条件生成（随机采样）...")
synthetic_random = tvae_model.sample(len(train_features))
print(f"非条件生成Load_Type分布:\n{synthetic_random['Load_Type'].value_counts()}")

# ============================================================================
# 5. 分布对比可视化
# ============================================================================
print("\n" + "=" * 60)
print("5. 分布对比可视化")
print("=" * 60)

# 5.1 Load_Type分布对比
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 真实 vs 条件生成 vs 非条件生成
distributions = {
    'Real': train_features['Load_Type'].value_counts().sort_index(),
    'Conditional': synthetic_conditional['Load_Type'].value_counts().sort_index(),
    'Random': synthetic_random['Load_Type'].value_counts().sort_index()
}

for idx, (name, dist) in enumerate(distributions.items()):
    ax = axes[idx]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax.bar(dist.index, dist.values, color=colors[idx], alpha=0.7)
    ax.set_title(f'{name} Distribution', fontsize=12)
    ax.set_xlabel('Load_Type')
    ax.set_ylabel('Count')
    for bar, val in zip(bars, dist.values):
        ax.annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/conditional_load_type_distribution.png", dpi=150)
plt.close()
print("Load_Type分布对比图已保存!")

# 5.2 Maximum_Load条件生成的KDE对比
print("\n生成Maximum_Load条件生成KDE对比图...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

real_max = train_features[train_features['Load_Type'] == 'Maximum_Load']
synth_max = synthetic_by_type['Maximum_Load']

# Usage_kWh
ax = axes[0]
sns.kdeplot(real_max['Usage_kWh'], ax=ax, label='Real', color='blue', linewidth=2)
sns.kdeplot(synth_max['Usage_kWh'], ax=ax, label='Conditional Generated', color='orange', linewidth=2, linestyle='--')
ax.set_title('Maximum_Load: Usage_kWh Distribution', fontsize=12)
ax.legend()

# CO2
ax = axes[1]
sns.kdeplot(real_max['CO2(tCO2)'], ax=ax, label='Real', color='blue', linewidth=2)
sns.kdeplot(synth_max['CO2(tCO2)'], ax=ax, label='Conditional Generated', color='orange', linewidth=2, linestyle='--')
ax.set_title('Maximum_Load: CO2 Distribution', fontsize=12)
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/conditional_max_load_kde.png", dpi=150)
plt.close()
print("Maximum_Load KDE对比图已保存!")

# ============================================================================
# 6. TSTR效用评估：条件生成 vs 非条件生成
# ============================================================================
print("\n" + "=" * 60)
print("6. TSTR效用评估")
print("=" * 60)

# 准备测试数据
X_test = test_data[['Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh', 'CO2(tCO2)']].values
y_test = test_data['Load_Type'].values

# 6.1 真实数据基线
clf_real = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
clf_real.fit(train_features[['Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh', 'CO2(tCO2)']].values, train_features['Load_Type'].values)
acc_real = accuracy_score(y_test, clf_real.predict(X_test))
f1_real = f1_score(y_test, clf_real.predict(X_test), average='weighted')
print(f"真实数据训练 - Accuracy: {acc_real:.4f}, F1: {f1_real:.4f}")

# 6.2 条件生成数据训练的分类器
clf_cond = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
clf_cond.fit(synthetic_conditional[['Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh', 'CO2(tCO2)']].values,
             synthetic_conditional['Load_Type'].values)
acc_cond = accuracy_score(y_test, clf_cond.predict(X_test))
f1_cond = f1_score(y_test, clf_cond.predict(X_test), average='weighted')
print(f"条件生成数据训练 - Accuracy: {acc_cond:.4f}, F1: {f1_cond:.4f}")

# 6.3 非条件生成数据训练的分类器
clf_rand = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
clf_rand.fit(synthetic_random[['Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh', 'CO2(tCO2)']].values,
             synthetic_random['Load_Type'].values)
acc_rand = accuracy_score(y_test, clf_rand.predict(X_test))
f1_rand = f1_score(y_test, clf_rand.predict(X_test), average='weighted')
print(f"非条件生成数据训练 - Accuracy: {acc_rand:.4f}, F1: {f1_rand:.4f}")

# Utility Ratio
print(f"\nTSTR效用比值 (相对于真实数据):")
print(f"  条件生成: {acc_cond/acc_real:.2%}")
print(f"  非条件生成: {acc_rand/acc_real:.2%}")

# 6.4 效用对比柱状图
fig, ax = plt.subplots(figsize=(8, 5))
models = ['Real', 'Conditional\nGenerated', 'Random\nGenerated']
accs = [acc_real, acc_cond, acc_rand]
f1s = [f1_real, f1_cond, f1_rand]

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, accs, width, label='Accuracy', color=['#3498db', '#e74c3c', '#2ecc71'])
bars2 = ax.bar(x + width/2, f1s, width, label='F1-Score', color=['#85c1e9', '#f5b7b1', '#a9dfbf'])

ax.set_ylabel('Score')
ax.set_title('TSTR: Conditional vs Random Generation')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(0, 1.1)

for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/conditional_tstr_comparison.png", dpi=150)
plt.close()
print("TSTR效用对比图已保存!")

# ============================================================================
# 7. 物理耦合保持度评估
# ============================================================================
print("\n" + "=" * 60)
print("7. 物理耦合保持度评估")
print("=" * 60)

print("\n各Load_Type的Usage-CO2相关性:")
print(f"{'Load_Type':<15} {'Real':>10} {'Conditional':>12} {'Error':>10}")
print("-" * 50)

for lt in load_types:
    real_lt = train_features[train_features['Load_Type'] == lt]
    synth_lt = synthetic_by_type[lt]

    real_corr = real_lt[['Usage_kWh', 'CO2(tCO2)']].corr().iloc[0, 1]
    synth_corr = synth_lt[['Usage_kWh', 'CO2(tCO2)']].corr().iloc[0, 1]
    error = abs(synth_corr - real_corr)

    print(f"{lt:<15} {real_corr:>10.4f} {synth_corr:>12.4f} {error:>10.4f}")

# ============================================================================
# 8. 保存结果
# ============================================================================
print("\n" + "=" * 60)
print("8. 保存结果")
print("=" * 60)

results = {
    "load_type_distribution": {
        lt: {
            "real_count": int(len(train_features[train_features['Load_Type'] == lt])),
            "generated_count": int(len(synthetic_by_type[lt]))
        }
        for lt in load_types
    },
    "tstr_utility": {
        "real_accuracy": float(acc_real),
        "conditional_accuracy": float(acc_cond),
        "random_accuracy": float(acc_rand),
        "conditional_ratio": float(acc_cond/acc_real),
        "random_ratio": float(acc_rand/acc_real)
    },
    "correlation_by_load_type": {
        lt: {
            "real": float(train_features[train_features['Load_Type'] == lt][['Usage_kWh', 'CO2(tCO2)']].corr().iloc[0, 1]),
            "generated": float(synthetic_by_type[lt][['Usage_kWh', 'CO2(tCO2)']].corr().iloc[0, 1])
        }
        for lt in load_types
    }
}

with open(f"{OUTPUT_DIR}/conditional_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n条件生成结果已保存: {OUTPUT_DIR}/conditional_results.json")

print("\n" + "=" * 60)
print("条件生成实验完成!")
print("=" * 60)