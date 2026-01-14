---
title: "PICU患者死亡率预测：多模型机器学习全流程实现"
collection: portfolio
type: "Machine Learning"
permalink: /portfolio/picu-mortality-prediction/
date: 2026-01-14
excerpt: "基于PICU临床数据构建Logistic回归、随机森林与SVM模型，实现住院死亡率预测；覆盖缺失值处理、异常值截断、交叉验证调参、ROC/PR/校准评估与特征贡献解释。"
header:
  teaser: /images/portfolio/picu-mortality-prediction/roc_curves_models.png
tags:
- 临床机器学习
- 死亡率预测
- 模型可解释性
- 端到端Pipeline
tech_stack:
- name: Python
- name: Scikit-learn
- name: Pandas
- name: Matplotlib
- name: Seaborn
classes: wide
---

## 项目背景
儿科重症监护室（PICU）患者的院内死亡率预测对风险分层与资源分配具有重要意义。本项目基于PICU临床表格数据，搭建端到端的机器学习流程：**数据读取 → 缺失值处理 → 异常值截断（winsorization）→ 特征标准化 → 多模型训练与调参 → 综合评估（ROC/PR/校准/混淆矩阵）→ 特征贡献解释**。

---

## 数据概览与缺失值情况

**数据规模与结局比例：** 总样本约 8,952，死亡率约 7.5%（类别不平衡明显）。

{% comment %} 如果你生成了 outcome_distribution.png / missing_values_bar.png 就直接展示 {% endcomment %}

<figure>
  <img src="/images/portfolio/picu-mortality-prediction/outcome_distribution.png" alt="Outcome distribution">
  <figcaption>结局分布（Survival vs Death）：数据存在明显类别不平衡。</figcaption>
</figure>

<figure>
  <img src="/images/portfolio/picu-mortality-prediction/missing_values_bar.png" alt="Missing value rate">
  <figcaption>变量缺失率分布：用于支撑后续的中位数填补策略。</figcaption>
</figure>

---

## 核心实现：端到端预处理 Pipeline

本项目采用统一 Pipeline 实现“**训练集拟合 + 测试集应用**”，避免数据泄漏：

- **缺失值填补**：连续变量用 **Median Imputation**（鲁棒）
- **异常值处理**：Winsorization（1%–99%分位截断）
- **标准化**：Z-score 标准化（对 LR / SVM 更友好）
- **类不平衡**：所有模型均使用 `class_weight="balanced"` 或 `balanced_subsample`

```python
# 数值特征：中位数填补 + 异常值截断 + 标准化
numeric_preprocess = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("winsor", Winsorizer(lower_q=0.01, upper_q=0.99)),
    ("scaler", StandardScaler()),
])

# 合并预处理流程（本数据集实际无分类变量）
preprocess = ColumnTransformer(
    transformers=[("num", numeric_preprocess, numeric_cols)],
    remainder="drop"
)
