---
title: "PICU患者死亡率预测：多模型机器学习全流程实现"
collection: portfolio
type: "Machine Learning"
permalink: /portfolio/
date: 2026-01-14
excerpt: "基于PICU临床数据构建Logistic回归、随机森林和SVM模型，实现患者住院死亡率精准预测，包含完整的数据预处理、模型调优与可解释性分析"
header:
  teaser: /images/portfolio/picu-mortality-prediction/roc_curves_models.png
tags:
- 临床机器学习
- 死亡率预测
- 模型可解释性
- 特征工程
tech_stack:
- name: Python
- name: Scikit-learn
- name: Pandas
- name: Matplotlib
- name: Seaborn
---

## 项目背景
本项目针对儿科重症监护室（PICU）患者的住院死亡率预测问题，基于临床数据集构建多模型机器学习预测系统。通过完整的数据预处理流程（缺失值填充、异常值截断、特征标准化），对比Logistic回归、随机森林和SVM三种模型的预测性能，并通过特征重要性分析揭示关键临床影响因素，为临床决策提供数据支持。

## 核心实现

### 1. 数据预处理Pipeline
```python
# 数值特征处理：中位数填充+异常值截断+标准化
numeric_preprocess = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("winsor", Winsorizer(lower_q=0.01, upper_q=0.99)),
    ("scaler", StandardScaler()),
])

# 分类特征处理：众数填充+独热编码
categorical_preprocess = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

# 合并预处理流程
preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_preprocess, numeric_cols),
        ("cat", categorical_preprocess, categorical_cols),
    ],
    remainder="drop"
)
# 模型与参数网格定义
models = {
    "Logistic(L2)": (Pipeline(steps=[("prep", preprocess), ("clf", LogisticRegression(class_weight="balanced"))]), 
                     {"clf__C": [0.01, 0.1, 1, 5, 10]}),
    "RandomForest": (Pipeline(steps=[("prep", preprocess), ("clf", RandomForestClassifier(class_weight="balanced_subsample", n_estimators=500))]),
                     {"clf__max_depth": [None, 5, 10, 20], "clf__min_samples_leaf": [1, 3, 5, 10]}),
    "SVM(RBF)": (Pipeline(steps=[("prep", preprocess), ("clf", SVC(probability=True, class_weight="balanced"))]),
                 {"clf__C": [0.5, 1, 2, 5, 10], "clf__gamma": ["scale", 0.01, 0.05, 0.1]})
}

# 5折分层交叉验证+网格搜索
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_estimators = {}
for name, (pipe, grid) in models.items():
    gs = GridSearchCV(estimator=pipe, param_grid=grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    gs.fit(X_train, y_train)
    best_estimators[name] = gs.best_estimator_
# 置换重要性分析（最优模型）
r = permutation_importance(
    best_est, X_test, y_test,
    scoring="roc_auc", n_repeats=10, random_state=42, n_jobs=-1
)
imp = pd.Series(r.importances_mean, index=feature_names).sort_values(ascending=False)
