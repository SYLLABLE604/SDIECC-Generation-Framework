#!/usr/bin/env python3
"""
综合实验脚本 - 解决评审者提出的所有问题

新增内容：
1. Membership Inference Attack 隐私审计
2. 与SDV内置方法的公平对比
3. 改进的评估指标
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.spatial.distance import cdist
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer, CTGANSynthesizer

DATA_PATH = "data/raw/Steel_industry_data.csv"
OUTPUT_DIR = "results"
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 数据分割策略
# ============================================================================
def split_data(df, features, test_size=0.2, val_size=0.1):
    """三分法分割：训练/验证/测试"""
    train_val_data, test_data = train_test_split(
        df[features], test_size=test_size, random_state=RANDOM_STATE
    )
    train_data, val_data = train_test_split(
        train_val_data, test_size=val_size/(1-test_size), random_state=RANDOM_STATE
    )
    return train_data, val_data, test_data

def compute_physics_bounds(val_data):
    """从验证集计算物理边界"""
    usage = val_data['Usage_kWh'].values
    co2 = val_data['CO2(tCO2)'].values
    ratio = co2 / usage
    ratio = ratio[np.isfinite(ratio)]
    ratio_5th, ratio_95th = np.percentile(ratio, 5), np.percentile(ratio, 95)
    return ratio_5th, ratio_95th

def physics_filter(usage, co2, ratio_lower, ratio_upper):
    """物理边界过滤"""
    expected_lower = usage * ratio_lower
    expected_upper = usage * ratio_upper
    violations = (co2 < expected_lower) | (co2 > expected_upper)
    return violations

# ============================================================================
# Membership Inference Attack (MIA)
# ============================================================================
def membership_inference_attack(real_data, synthetic_data, model_class=RandomForestClassifier, n_estimators=100):
    """
    Membership Inference Attack 隐私审计

    原理：训练一个攻击模型区分成员（训练数据）和非成员（合成数据）
    如果攻击模型准确率显著高于随机猜测，说明存在隐私泄露
    """
    print("\n--- Membership Inference Attack ---")

    # 准备数据
    # 成员数据：来自真实训练集
    # 非成员数据：来自合成数据

    # 对于每个真实记录，判断它是否接近训练集或合成集
    # 使用最近邻距离比率作为信号

    # 合并用于最近邻搜索
    real_array = real_data.values
    synth_array = synthetic_data.values

    # 标准化
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_array)
    synth_scaled = scaler.transform(synth_array)

    # 计算每个合成记录到最近真实训练记录的距离
    # 如果距离很小，说明可能是成员

    # 方法：使用DCR作为成员信号
    from scipy.spatial.distance import cdist
    distances = cdist(synth_scaled, real_scaled, 'euclidean')
    min_distances = distances.min(axis=1)

    # 攻击模型：基于距离判断是否为成员
    # 真实成员的距离应该小于合成数据的距离
    attack_features = min_distances.reshape(-1, 1)

    # 真实标签：0表示合成数据
    y_attack = np.zeros(len(synthetic_data))

    # 也需要对真实数据进行攻击测试
    distances_real = cdist(real_scaled, real_scaled, 'euclidean')
    np.fill_diagonal(distances_real, np.inf)
    min_distances_real = distances_real.min(axis=1)

    # 合并特征
    X_attack = np.vstack([attack_features, min_distances_real.reshape(-1, 1)])
    y_attack = np.hstack([y_attack, np.ones(len(real_data))])

    # 训练攻击模型
    attack_model = model_class(n_estimators=n_estimators, random_state=RANDOM_STATE)
    attack_model.fit(X_attack, y_attack)

    # 评估攻击效果
    attack_pred = attack_model.predict(X_attack)
    attack_acc = accuracy_score(y_attack, attack_pred)

    print(f"  MIA Attack Accuracy: {attack_acc:.4f}")
    print(f"  Random Guess Baseline: 0.5000")

    # 如果准确率显著高于0.5，说明存在隐私风险
    privacy_risk = "HIGH" if attack_acc > 0.55 else "LOW"
    print(f"  Privacy Risk Level: {privacy_risk}")

    return attack_acc, privacy_risk

def enhanced_privacy_audit(real_train, synthetic_data, n_samples=1000):
    """
    增强版隐私审计：多个指标
    """
    print("\n--- Enhanced Privacy Audit ---")

    # 选择数值列
    numeric_cols = ['Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh', 'CO2(tCO2)']
    real_train = real_train[numeric_cols]
    synthetic_data = synthetic_data[numeric_cols]

    # 1. DCR (Distance to Closest Record)
    real_array = real_train.values
    synth_array = synthetic_data.values

    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_array)
    synth_scaled = scaler.transform(synth_array)

    distances = cdist(synth_scaled, real_scaled, 'euclidean')
    dcr_values = distances.min(axis=1)

    dcr_mean = dcr_values.mean()
    dcr_median = np.median(dcr_values)
    dcr_min = dcr_values.min()
    dcr_positive_ratio = (dcr_values > 0).mean()

    print(f"  DCR Mean: {dcr_mean:.4f}")
    print(f"  DCR Median: {dcr_median:.4f}")
    print(f"  DCR Min: {dcr_min:.4f}")
    print(f"  DCR>0 Ratio: {dcr_positive_ratio:.4f}")

    # 2. Nearest Neighbor Distance Ratio (NNDR)
    # 计算每个合成记录到最近和次近真实记录的距离比率
    sorted_distances = np.sort(distances, axis=1)
    nndr = sorted_distances[:, 0] / (sorted_distances[:, 1] + 1e-10)
    nndr_mean = nndr.mean()
    nndr_median = np.median(nndr)

    print(f"  NNDR Mean: {nndr_mean:.4f}")
    print(f"  NNDR Median: {nndr_median:.4f}")

    # 3. Membership Inference Attack
    mia_acc, risk = membership_inference_attack(real_train, synthetic_data)

    return {
        'dcr_mean': dcr_mean,
        'dcr_median': dcr_median,
        'dcr_min': dcr_min,
        'dcr_positive_ratio': dcr_positive_ratio,
        'nndr_mean': nndr_mean,
        'nndr_median': nndr_median,
        'mia_accuracy': mia_acc,
        'privacy_risk': risk
    }

# ============================================================================
# 主实验
# ============================================================================
print("=" * 70)
print("综合实验 - 评审优化版本")
print("=" * 70)

# 1. 数据加载
df = pd.read_csv(DATA_PATH)
features = ['Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh', 'CO2(tCO2)', 'Load_Type']
data = df[features].copy()

le = LabelEncoder()
data['Load_Type_encoded'] = le.fit_transform(data['Load_Type'])

# 数据分割
train_data, val_data, test_data = split_data(df, features)

print(f"\n数据分割:")
print(f"  训练集: {len(train_data)}")
print(f"  验证集: {len(val_data)}")
print(f"  测试集: {len(test_data)}")

# 物理边界
ratio_lower, ratio_upper = compute_physics_bounds(val_data)
print(f"  物理边界: [{ratio_lower:.6f}, {ratio_upper:.6f}]")

# 真实数据违规率
real_violations = physics_filter(
    train_data['Usage_kWh'].values,
    train_data['CO2(tCO2)'].values,
    ratio_lower, ratio_upper
)
print(f"  真实数据违规率: {real_violations.mean()*100:.2f}%")

# ============================================================================
# 模型训练
# ============================================================================
print("\n" + "=" * 70)
print("模型训练")
print("=" * 70)

train_features = train_data.copy()

# TVAE
print("\nTraining TVAE...")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(train_features)

tvae = TVAESynthesizer(metadata, epochs=100, batch_size=200, verbose=False)
tvae.fit(train_features)
synthetic_tvae = tvae.sample(len(train_features))

# CTGAN
print("Training CTGAN...")
ctgan = CTGANSynthesizer(metadata, epochs=50, batch_size=200, verbose=False)
ctgan.fit(train_features)
synthetic_ctgan = ctgan.sample(len(train_features))

# 应用物理过滤
print("\nApplying physics filter...")
tvae_violations = physics_filter(
    synthetic_tvae['Usage_kWh'].values,
    synthetic_tvae['CO2(tCO2)'].values,
    ratio_lower, ratio_upper
)
synthetic_tvae_filtered = synthetic_tvae[~tvae_violations]

ctgan_violations = physics_filter(
    synthetic_ctgan['Usage_kWh'].values,
    synthetic_ctgan['CO2(tCO2)'].values,
    ratio_lower, ratio_upper
)
synthetic_ctgan_filtered = synthetic_ctgan[~ctgan_violations]

print(f"  TVAE过滤后: {len(synthetic_tvae_filtered)}/{len(synthetic_tvae)} ({len(synthetic_tvae_filtered)/len(synthetic_tvae)*100:.1f}%)")
print(f"  CTGAN过滤后: {len(synthetic_ctgan_filtered)}/{len(synthetic_ctgan)} ({len(synthetic_ctgan_filtered)/len(synthetic_ctgan)*100:.1f}%)")

# 平衡样本数量
min_samples = min(len(synthetic_tvae_filtered), len(synthetic_ctgan_filtered), len(train_data))
synthetic_tvae_balanced = synthetic_tvae_filtered.sample(min_samples, random_state=RANDOM_STATE).copy()
synthetic_ctgan_balanced = synthetic_ctgan_filtered.sample(min_samples, random_state=RANDOM_STATE).copy()
real_balanced = train_data.sample(min_samples, random_state=RANDOM_STATE).copy()

# 添加编码列到合成数据
synthetic_tvae_balanced['Load_Type_encoded'] = le.transform(synthetic_tvae_balanced['Load_Type'])
synthetic_ctgan_balanced['Load_Type_encoded'] = le.transform(synthetic_ctgan_balanced['Load_Type'])
real_balanced['Load_Type_encoded'] = le.transform(real_balanced['Load_Type'])

print(f"  平衡后每组样本数: {min_samples}")

# ============================================================================
# 相关性评估
# ============================================================================
print("\n" + "=" * 70)
print("Fidelity: 相关性评估")
print("=" * 70)

def compute_correlation(data, col1, col2):
    return data[col1].corr(data[col2])

real_corr = compute_correlation(real_balanced, 'Usage_kWh', 'CO2(tCO2)')
tvae_corr = compute_correlation(synthetic_tvae_balanced, 'Usage_kWh', 'CO2(tCO2)')
ctgan_corr = compute_correlation(synthetic_ctgan_balanced, 'Usage_kWh', 'CO2(tCO2)')

print(f"\nUsage-CO2 Correlation:")
print(f"  Real:     {real_corr:.4f}")
print(f"  TVAE:     {tvae_corr:.4f} (偏差: {(tvae_corr-real_corr)/real_corr*100:.1f}%)")
print(f"  CTGAN:    {ctgan_corr:.4f} (偏差: {(ctgan_corr-real_corr)/real_corr*100:.1f}%)")

# ============================================================================
# TSTR 评估
# ============================================================================
print("\n" + "=" * 70)
print("Utility: TSTR评估")
print("=" * 70)

X_train = real_balanced[['Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh', 'CO2(tCO2)']].values
y_train = real_balanced['Load_Type_encoded'].values
# Add encoded column to test data
test_data_copy = test_data.copy()
test_data_copy['Load_Type_encoded'] = le.transform(test_data_copy['Load_Type'])

X_test = test_data_copy[['Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh', 'CO2(tCO2)']].values
y_test = test_data_copy['Load_Type_encoded'].values

# Real baseline
clf_real = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
clf_real.fit(X_train, y_train)
real_acc = accuracy_score(y_test, clf_real.predict(X_test))
print(f"\nReal Baseline: {real_acc:.4f}")

# TVAE+Filter
X_train_tvae = synthetic_tvae_balanced[['Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh', 'CO2(tCO2)']].values
y_train_tvae = synthetic_tvae_balanced['Load_Type_encoded'].values
clf_tvae = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
clf_tvae.fit(X_train_tvae, y_train_tvae)
tvae_acc = accuracy_score(y_test, clf_tvae.predict(X_test))
print(f"TVAE+Filter: {tvae_acc:.4f} (Ratio: {tvae_acc/real_acc*100:.1f}%)")

# CTGAN+Filter
X_train_ctgan = synthetic_ctgan_balanced[['Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh', 'CO2(tCO2)']].values
y_train_ctgan = synthetic_ctgan_balanced['Load_Type_encoded'].values
clf_ctgan = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
clf_ctgan.fit(X_train_ctgan, y_train_ctgan)
ctgan_acc = accuracy_score(y_test, clf_ctgan.predict(X_test))
print(f"CTGAN+Filter: {ctgan_acc:.4f} (Ratio: {ctgan_acc/real_acc*100:.1f}%)")

# ============================================================================
# 隐私审计
# ============================================================================
print("\n" + "=" * 70)
print("Privacy: 增强隐私审计")
print("=" * 70)

print("\n--- TVAE+Filter Privacy ---")
tvae_privacy = enhanced_privacy_audit(train_data[['Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh', 'CO2(tCO2)']], synthetic_tvae_balanced)

print("\n--- CTGAN+Filter Privacy ---")
ctgan_privacy = enhanced_privacy_audit(train_data[['Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh', 'CO2(tCO2)']], synthetic_ctgan_balanced)

# ============================================================================
# 保存结果
# ============================================================================
results = {
    'fidelity': {
        'real_correlation': real_corr,
        'tvae_correlation': tvae_corr,
        'ctgan_correlation': ctgan_corr,
        'tvae_deviation_pct': (tvae_corr-real_corr)/real_corr*100,
        'ctgan_deviation_pct': (ctgan_corr-real_corr)/real_corr*100
    },
    'utility': {
        'real_tstr_accuracy': real_acc,
        'tvae_tstr_accuracy': tvae_acc,
        'ctgan_tstr_accuracy': ctgan_acc,
        'tvae_tstr_ratio': tvae_acc/real_acc*100,
        'ctgan_tstr_ratio': ctgan_acc/real_acc*100
    },
    'privacy': {
        'tvae': tvae_privacy,
        'ctgan': ctgan_privacy
    }
}

with open(f'{OUTPUT_DIR}/comprehensive_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("实验完成！结果已保存到 results/comprehensive_results.json")
print("=" * 70)

# 打印最终总结
print("\n" + "=" * 70)
print("最终结果总结")
print("=" * 70)
print(f"\n【保真度 - 相关性】")
print(f"  Real:     {real_corr:.4f}")
print(f"  TVAE:     {tvae_corr:.4f} ({(tvae_corr-real_corr)/real_corr*100:+.1f}%)")
print(f"  CTGAN:    {ctgan_corr:.4f} ({(ctgan_corr-real_corr)/real_corr*100:+.1f}%)")

print(f"\n【效用 - TSTR准确率】")
print(f"  Real:     {real_acc:.4f} (baseline)")
print(f"  TVAE:     {tvae_acc:.4f} ({tvae_acc/real_acc*100:.1f}%)")
print(f"  CTGAN:    {ctgan_acc:.4f} ({ctgan_acc/real_acc*100:.1f}%)")

print(f"\n【隐私 - DCR指标】")
print(f"  TVAE:  DCR Mean={tvae_privacy['dcr_mean']:.4f}, DCR>0={tvae_privacy['dcr_positive_ratio']*100:.1f}%, MIA Acc={tvae_privacy['mia_accuracy']:.4f}")
print(f"  CTGAN: DCR Mean={ctgan_privacy['dcr_mean']:.4f}, DCR>0={ctgan_privacy['dcr_positive_ratio']*100:.1f}%, MIA Acc={ctgan_privacy['mia_accuracy']:.4f}")