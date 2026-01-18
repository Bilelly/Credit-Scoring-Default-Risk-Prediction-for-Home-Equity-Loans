# Credit Scoring & Default Risk Prediction for Home Equity Loans

<p align="center">
  <strong>Advanced Machine Learning Model for Credit Risk Assessment</strong>
</p>

---
## 💼 Business Context

The model addresses critical challenges in credit risk management:
- **High Default Rates**: Home equity loans require accurate risk assessment
- **Portfolio Optimization**: Balance approval rates with loss minimization
- **Regulatory Compliance**: Provide transparent, explainable decision-making processes
- **Operational Efficiency**: Automate risk assessment at scale
---

## 📊 Dataset

### Data Source
Home equity loan dataset containing 5,960 borrower records with 13 features

### Feature Categories

| Category | Features | Description |
|----------|----------|-------------|
| **Professional** | JOB, REASON | Employment sector and loan purpose |
| **Financial** | DELINQ, DEROG, NINO, NINQ | Credit history and inquiries |
| **Debt Metrics** | DEBTINC, MORTDUE, CLAGE, CLNO | Debt ratios and credit line activity |
| **Employment** | YOJ | Years on job (employment stability) |

### Target Variable
- **BAD**: 0 = Good credit (No default), 1 = Bad credit (Default risk)

### Data Quality
- **Total Records**: 5,960
- **Features**: 13 (after preprocessing)
- **Missing Values**: Handled through imputation strategies
- **Class Distribution**: Addressed with appropriate handling techniques

---

## 🔄 Methodology

### 1. **Data Exploration & Cleaning** (`01_data_exploration.ipynb`)
- Descriptive statistics and distribution analysis
- Missing value imputation
- Outlier detection and treatment
- Feature correlation analysis

### 2. **Data Preprocessing** (`02_data_preprocessing.ipynb`)
- Categorical encoding (One-Hot Encoding)
- Numerical feature scaling (StandardScaler)
- Pipeline construction for reproducibility
- Train/test split: 80/20

### 3. **Model Development** (`03_data_modeling.ipynb`)
- **Algorithm**: XGBoost Classifier
- **Baseline Model**: Initial hyperparameter configuration
- **Cross-Validation**: K-Fold cross-validation for robustness
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### 4. **Hyperparameter Optimization** (`04_model_optimisation.ipynb`)
- **Method**: Grid Search & Optuna
- **Key Parameters Tuned**: 
  - Learning rate (eta)
  - Max depth
  - Min child weight
  - Subsample ratio
  - Colsample bytree

### 5. **Model Explainability** (`05_model_explainability.ipynb`)
- Feature importance analysis (XGBoost native)
- SHAP (SHapley Additive exPlanations) values for local and global interpretability
- Risk driver identification

---

## 🧠 Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA INPUT                               │
│  (Borrower Profile: Job, Credit History, Debt Metrics)     │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              PREPROCESSING PIPELINE                         │
│  - Categorical Encoding                                     │
│  - Numerical Scaling                                        │
│  - Feature Engineering                                      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              XGBOOST CLASSIFIER                             │
│  - Ensemble of Decision Trees                               │
│  - Optimized Hyperparameters                                │
│  - Gradient Boosting Framework                              │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│            RISK PREDICTION OUTPUT                           │
│  - Probability of Default                                   │
│  - Risk Classification (Good/Bad)                           │
│  - SHAP Explainability                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📈 Key Results

### Model Performance
- **Accuracy**: Evaluated on held-out test set (check `04_model_optimisation.ipynb`)
- **Precision**: Balance between false positives and true positives
- **Recall**: Ability to identify true defaults
- **F1-Score**: Harmonic mean for imbalanced data
- **ROC-AUC**: Discrimination ability across thresholds

### Top 5 Risk Drivers (Feature Importance)
1. **cat_JOB_Sales** (12.2%) - Professional sector is critical
2. **num_DELINQ** (11.8%) - Payment delinquency history
3. **cat_JOB_Office** (8.9%) - Office professionals risk profile
4. **cat_JOB_Other** (7.0%) - Other professional categories
5. **cat_JOB_ProfExe** (6.2%) - Executive professionals profile

---

## 📁 Project Structure

```
risk-credit/
│
├── README.md                           # Project documentation
│
├── data/
│   ├── Data.csv                       # Original dataset
│   ├── cleaned_data.csv               # Cleaned data
│   └── preprocessed_data.csv          # Pipeline-ready data
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # EDA and initial analysis
│   ├── 02_data_preprocessing.ipynb    # Data preparation
│   ├── 03_data_modeling.ipynb         # Model training
│   ├── 04_model_optimisation.ipynb    # Hyperparameter tuning
│   └── 05_model_explainability.ipynb  # SHAP and interpretability
│
├── src/
│   ├── PreprocessingFunction.py       # Pipeline builder
│   ├── EvaluationFunction.py          # Model evaluation metrics
│   ├── ValidationCross.py             # Cross-validation utilities
│   ├── RemoveOutliers.py              # Outlier detection/removal
│   └── helperfunction.py              # Utility functions
│
├── models/
│   └── best_model.pkl                 # Trained XGBoost pipeline
│
└── report/
    └── eda_report.html                # EDA visualization report
```

---

## 🚀 Installation & Usage

### Prerequisites
```bash
Python >= 3.8
conda or pip
```

---

## 🔍 Model Explainability

### Feature Importance (XGBoost)
- **Global perspective**: Which features reduce prediction error most
- **Non-directional**: Shows magnitude only
- **Fast computation**: Integrated in gradient boosting

### SHAP Values
- **Local + Global explanations**: How each instance contributes to predictions
- **Directional impact**: Shows if feature increases or decreases risk
- **Theoretically grounded**: Based on Shapley values from game theory

### Key Insights
- **DELINQ (Payment History)**: High delinquency strongly increases default probability
- **Professional Sector**: Sales and Office sectors have distinct risk profiles
- **Debt-to-Income**: Higher ratios correlate with increased default risk




