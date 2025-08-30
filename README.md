# Engage2Value: From Clicks to Conversions

This project focuses on analyzing user session-level behavior and predicting **purchase value** from digital commerce clickstream data. The goal is to transform user engagement (clicks) into actionable insights that help businesses optimize conversions and revenue.

---

## 📂 Project Files

- **Notebook**: `22f3003055-notebook-t22025.ipynb` — End-to-end analysis, feature engineering, modeling, and evaluation.
- **Data**:
  - Training data → `/kaggle/input/engage-2-value-from-clicks-to-conversions/train_data.csv`
  - Test data → `/kaggle/input/engage-2-value-from-clicks-to-conversions/test_data.csv`

---

## 🛠️ Tech Stack

- **Languages**: Python  
- **Data Handling**: `pandas`, `numpy`  
- **Visualization**: `matplotlib`, `seaborn`  
- **Preprocessing & Encoding**:  
  - `SimpleImputer` (missing value handling)  
  - `OneHotEncoder`, `CountEncoder` (categorical encoding)  
- **Pipelines & Transformation**:  
  - `Pipeline`, `ColumnTransformer`  
- **Dimensionality Reduction**: `PCA`  
- **Feature Selection**: `SelectKBest` with `chi2` and `f_regression`  
- **Models**:
  - `XGBRegressor` (XGBoost)
  - `RandomForestRegressor`
  - `LGBMRegressor` (LightGBM)
  - `ExtraTreesRegressor`
- **Model Selection & Evaluation**:  
  - `train_test_split`, `GridSearchCV`
  - Metrics: `MAE`, `MSE`, `RMSE`, `R²`

---

## ⚙️ Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/AmanManiTiwari/Engage2Value-From-Clicks-to-Conversions.git
   cd Engage2Value-From-Clicks-to-Conversions
