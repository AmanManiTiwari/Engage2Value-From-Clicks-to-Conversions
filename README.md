<div align="center">

# 🛒 Engage2Value: From Clicks to Conversions

### *Predicting purchase value from user session behavior using ensemble regression*

<br>

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Regression-006400?style=for-the-badge)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-Regression-9ACD32?style=for-the-badge)](https://lightgbm.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipelines-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)](https://github.com/AmanManiTiwari/Engage2Value-From-Clicks-to-Conversions)

<br>

> **Every click tells a story. Can we read it well enough to predict what a user will spend?**
> This project turns raw e-commerce clickstream data into revenue predictions using production-grade ML.

</div>

---

## 📌 Overview

**Engage2Value** is an end-to-end regression project that models **purchase value** from digital user session data - the sequence of clicks, page visits, and interactions a shopper makes before (or without) buying. By learning from session-level behavioral signals, the model predicts how much a user is likely to spend, enabling smarter personalization, dynamic pricing, and marketing spend decisions.

The project showcases a complete ML workflow applied to a real-world e-commerce problem that directly impacts revenue.

---

## 🎯 Problem Statement

| | Details |
|---|---|
| **Task** | Regression - predict the purchase value of a user session |
| **Domain** | E-Commerce / Digital Marketing Analytics |
| **Input** | User clickstream & session-level behavioral data |
| **Target Variable** | Purchase value (continuous) |
| **Key Challenge** | High sparsity, zero-inflated target (most sessions don't convert), mixed feature types |
| **Business Impact** | Better conversion optimization, personalized offers, smarter ad spend |

---

## 🔬 ML Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                  RAW CLICKSTREAM DATA                            │
│         (train_data.csv  /  test_data.csv from Kaggle)           │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                  EXPLORATORY DATA ANALYSIS                       │
│   • Distribution of purchase values (zero-inflated check)        │
│   • Session feature correlations & outlier profiling             │
│   • Conversion funnel analysis: clicks → add-to-cart → purchase  │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                  DATA PREPROCESSING                              │
│   • Missing values  →  SimpleImputer                             │
│   • Categorical vars  →  OneHotEncoder + CountEncoder            │
│   • All steps wrapped in ColumnTransformer + Pipeline            │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING                             │
│   • PCA for dimensionality reduction                             │
│   • SelectKBest (chi2 + f_regression) for relevance filtering    │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                  ENSEMBLE MODEL TRAINING                         │
│   ┌──────────────┐ ┌──────────────┐ ┌────────────┐ ┌─────────┐  │
│   │  XGBRegressor│ │  LGBMReg.    │ │ RF Reg.    │ │ Extra   │  │
│   │  (XGBoost)   │ │  (LightGBM)  │ │ (RF)       │ │ Trees   │  │
│   └──────────────┘ └──────────────┘ └────────────┘ └─────────┘  │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│               HYPERPARAMETER TUNING                              │
│              GridSearchCV + Cross-Validation                     │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                   EVALUATION                                     │
│          MAE  ·  MSE  ·  RMSE  ·  R²  ·  Residual Plots         │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Models Compared

| Model | Type | Why It's Here |
|-------|------|---------------|
| **XGBoost Regressor** | Gradient Boosting | Industry standard for tabular regression; handles mixed features and non-linearity exceptionally well |
| **LightGBM Regressor** | Gradient Boosting | Faster training on large datasets with leaf-wise tree growth; great for sparse click data |
| **Random Forest Regressor** | Bagging Ensemble | Strong baseline; interpretable feature importances; resistant to overfitting |
| **Extra Trees Regressor** | Extreme Randomization | High variance reduction; effective complement to RF in ensemble comparisons |

All models run inside a **scikit-learn Pipeline** - ensuring zero data leakage between training and validation sets.

---

## 📐 Feature Engineering Highlights

```python
# Dual feature selection strategy
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(score_func=f_regression, k=20))
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ('count_enc', CountEncoder())         # frequency-based encoding for high-cardinality features
])

# Dimensionality reduction after encoding
pca = PCA(n_components=0.95)             # retain 95% of variance
```

---

## 📊 Evaluation Metrics

| Metric | What It Measures | Why It Matters Here |
|--------|-----------------|---------------------|
| **MAE** | Average absolute error in predicted spend | Directly interpretable in currency units |
| **RMSE** | Penalizes large prediction errors heavily | Critical when high-value orders matter most |
| **R²** | Variance explained by the model | Overall model quality benchmark |
| **Residual Plot** | Distribution of prediction errors | Detects systematic bias in predictions |

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| **Language** | Python 3.8+ |
| **Notebook** | Jupyter Notebook |
| **Data Handling** | `pandas`, `numpy` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Preprocessing** | `SimpleImputer`, `OneHotEncoder`, `CountEncoder`, `StandardScaler` |
| **Pipeline** | `Pipeline`, `ColumnTransformer` |
| **Feature Selection** | `SelectKBest` (`chi2`, `f_regression`), `PCA` |
| **Models** | `XGBRegressor`, `LGBMRegressor`, `RandomForestRegressor`, `ExtraTreesRegressor` |
| **Optimization** | `GridSearchCV`, `train_test_split` |
| **Dataset** | Kaggle - Engage2Value Clickstream Competition |

---

## 📁 Repository Structure

```
Engage2Value-From-Clicks-to-Conversions/
│
├── 📓 22f3003055-notebook-t22025.ipynb   ← Full pipeline: EDA → modeling → evaluation
├── 📄 README.md                          ← You are here
│
└── 📦 Data (Kaggle input - not committed)
    ├── train_data.csv                    ← Session features + purchase value labels
    └── test_data.csv                     ← Holdout set for final predictions
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/AmanManiTiwari/Engage2Value-From-Clicks-to-Conversions.git
cd Engage2Value-From-Clicks-to-Conversions
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm category_encoders
```

### 3. Set Up Kaggle Dataset
```bash
# Ensure your Kaggle API credentials are configured
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Download the dataset
kaggle competitions download -c engage-2-value-from-clicks-to-conversions
unzip engage-2-value-from-clicks-to-conversions.zip -d data/
```

### 4. Run the Notebook
```bash
jupyter notebook 22f3003055-notebook-t22025.ipynb
```

---

## 💡 Key Learning Outcomes

- Applying **regression ML** to predict continuous business outcomes (revenue) from behavioral data
- Handling **high-cardinality categorical features** using CountEncoder alongside OneHotEncoder
- Designing **leak-free ML pipelines** with `ColumnTransformer` + `Pipeline`
- Using **dual feature selection** strategies - statistical (SelectKBest) and variance-based (PCA)
- Comparing **four ensemble regressors** systematically with cross-validated evaluation
- Interpreting model performance in **business terms** (predicted spend vs. actual)

---

## 💼 Real-World Relevance

Predicting purchase value from sessions is a core capability at:

- **E-commerce platforms** (Amazon, Flipkart) - for personalized discounting and upselling
- **Ad-tech companies** (Google Ads, Meta) - for optimizing cost-per-conversion bidding
- **Retail analytics** - for identifying high-intent users before they bounce
- **CRM & lifecycle tools** - for scoring leads by predicted lifetime value

---

## 🙋 About the Author

**Aman Mani Tiwari** - Data Science practitioner building real-world ML portfolio projects across e-commerce, cybersecurity, and business analytics.

[![GitHub](https://img.shields.io/badge/GitHub-AmanManiTiwari-181717?style=flat-square&logo=github)](https://github.com/AmanManiTiwari)

---

<div align="center">

*If this project helped or inspired you, a ⭐ goes a long way - thank you!*

</div>
