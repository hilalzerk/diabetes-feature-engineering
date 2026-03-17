# 🩺 Diabetes Feature Engineering & ML Project

## 📌 Business Problem
The goal is to develop a machine learning model that can predict whether a person has diabetes based on given features.

## 📊 Dataset
- **Source:** National Institute of Diabetes and Digestive and Kidney Diseases, USA
- **Observations:** 768 | **Variables:** 9
- **Target:** Outcome (1: Diabetes, 0: Healthy)

## 🔧 Project Steps

### Task 1 — Exploratory Data Analysis (EDA)
- General overview
- Categorical & numeric variable analysis
- Target variable analysis
- Outlier analysis
- Missing value analysis (hidden 0 values)
- Correlation analysis

### Task 2 — Feature Engineering
- Replacing 0 values with NaN
- Filling missing values by Outcome group median
- Outlier suppression (IQR)
- Deriving 15+ new variables
- Label & One-Hot Encoding
- Standardization with RobustScaler

## 📈 Model Results

| Metric    | Random Forest | XGBoost |
|-----------|:------------:|:-------:|
| Accuracy  | **0.900**    | 0.887   |
| Recall    | **0.892**    | 0.802   |
| Precision | 0.810        | **0.867** |
| F1        | **0.850**    | 0.833   |
| AUC       | **0.900**    | 0.868   |

## 🚀 Improvement from Start to Finish

| Stage | Changes | Accuracy |
|-------|---------|----------|
| Raw data (no processing) | No feature engineering, no scaling | ~74% |
| Initial Feature Engineering | StandardScaler + basic new variables | 78% |
| + New Variables | Added interaction & log features | 78% |
| + Outcome-based filling + RobustScaler | Filled missing values by Outcome group median, switched to RobustScaler | **90%** |

## 🔑 Key Insight
The most critical improvement came from a single change in the **missing value filling strategy**:

| Method | Description | Impact |
|--------|-------------|--------|
| ❌ Global Median | Same median for everyone | Baseline |
| ✅ Outcome-based Median | Separate median for diabetic vs healthy group | **+12% accuracy** |

> Diabetic patients have significantly different insulin and glucose values than healthy individuals.
> Filling missing values **within each group** provides the model with much more realistic data.

## 🛠️ Technologies Used
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib, Seaborn

## 📁 Project Structure
```
diabetes-feature-engineering/
├── diabets.py          # Main project code
├── diabetes.csv        # Dataset
├── README.md           # Project description
├── Plots/              # Output plots
│   ├── correlation_matrix_before_fe.png
│   ├── correlation_matrix_after_fe.png
│   ├── feature_importance_rf.png
│   ├── feature_importance_xgb.png
│   ├── confusion_matrix_rf.png
│   ├── confusion_matrix_xgb.png
│   ├── model_comparison_rf_vs_xgb.png
│   └── target_vs_*.png
└── .gitignore
```

## 👩‍💻 Author
Hilal Zerk Demirkan — Miuul Data Science & AI Bootcamp
