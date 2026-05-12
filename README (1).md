# 💊 PRIS — Pharmaceutical Regulatory Intelligence System 

> *Predicting whether a drug will be classified as Regulated or Non-Regulated using clinical, safety, and pharmacological features — with full model explainability via SHAP.*

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Live Demo](#-live-demo)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Pipeline](#-project-pipeline)
- [Feature Engineering](#-feature-engineering)
- [Models Trained](#-models-trained)
- [Results](#-results)
- [SHAP Explainability](#-shap-explainability)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation--local-setup)
- [Author](#-author)

---

## 🔬 Project Overview

Drug regulation is one of the most critical decisions in pharmaceutical governance. Misclassifying a **high-risk drug as non-regulated** can lead to severe public health consequences — addiction, misuse, overdose, or unreported adverse events.

This project builds a **binary classification system** that predicts whether a drug should be classified as:

| Class | Label | Meaning |
|---|---|---|
| Non-Regulated Drug | `0` | Lower risk, generally available |
| Regulated Drug | `1` | Higher risk / controlled substance |

The goal is not just accuracy — but **high recall on Regulated Drugs**, minimizing dangerous false negatives.

---
## Live Demo




## 🎯 Problem Statement

Given a drug's clinical, safety, pharmacological, and market-related features, predict its **regulatory classification**.

This is a **supervised binary classification** problem with:
- Class imbalance (~74% Non-Regulated vs ~26% Regulated)
- High recall requirement for the minority class (Regulated Drugs)
- Need for model interpretability (regulatory decisions must be explainable)

---

## 📁 Dataset

**File:** `drug_regulatory_classification_dataset.csv`

| Property | Value |
|---|---|
| Total Records | 60,000 |
| Features | 30 raw + 11 engineered = **41 total** |
| Records after cleaning | 57,000 |
| Training samples | 45,600 |
| Test samples | 11,400 |
| Train / Test Split | 80 / 20 |
| Target Classes | `Regulated Drug` / `Non-Regulated Drug` |


---

**Key Feature Categories:**

| Category | Features |
|---|---|
| 🧪 Safety | `Side_Effect_Severity_Score`, `Abuse_Potential_Score`, `Adverse_Event_Reports` |
| 📋 Regulatory | `Regulatory_Risk_Score`, `Clinical_Trial_Phase`, `Recall_History_Count` |
| 💰 Financial | `Price_Per_Unit`, `Production_Cost`, `R&D_Investment_Million`, `Marketing_Spend` |
| 🏥 Distribution | `Prescription_Rate`, `Hospital_Distribution_Percentage`, `Doctor_Recommendation_Rate` |
| 🏭 Product | `Dosage_mg`, `Drug_Form`, `Therapeutic_Class`, `Manufacturing_Region` |


---

## 🔄 Project Pipeline

```
Raw CSV Data
    │
    ▼
Data Cleaning & EDA
    │  - Drop missing target rows (3000)
    │  - Analyze class distribution
    │  - Correlation & redundancy check
    ▼
Feature Engineering
    │  - Profit_Margin = Price_Per_Unit - Production_Cost
    │  - Drug_Safety_Risk = Side_Effect_Severity_Score + Abuse_Potential_Score
    │  - Risk_Safety_Interaction = Regulatory_Risk_Score × Drug_Safety_Risk
    │  - Log_Risk_Safety = log1p(Risk_Safety_Interaction)
    │  - Doctor_Influence_Index = Doctor_Recommendation_Rate × Prescription_Rate
    │  - Marketing_Efficiency = Annual_Sales_Volume / (Marketing_Spend + 1)
    ▼
Train-Test Split (80/20, random_state=42)
    ▼
Class Imbalance Handling
    │  - SMOTE on training data only (prevent data leakage)
    │  - scale_pos_weight for XGBoost
    ▼
Model Training & Evaluation
    │  - Logistic Regression (baseline)
    │  - Random Forest
    │  - Logistic Regression + SMOTE
    │  - Threshold Tuning
    │  - XGBoost (best model)
    │  - Final XGBoost (GridSearchCV tuned)
    ▼
Hyperparameter Tuning
    │  - RandomizedSearchCV (initial search)
    │  - GridSearchCV (fine-tuned)
    ▼
Model Explainability
    │  - SHAP TreeExplainer
    │  - Feature importance summary plot
    ▼
Model Deployment (Planned)
```

---

## 🛠️ Feature Engineering
  Domain knowledge was used to create 11 new features that capture complex interactions the raw features can't express alone.

### Risk-Safety Interaction
```
Risk_Safety_Interaction = Regulatory_Risk_Score × Drug_Safety_Risk
```
> Captures the **combined amplified effect** of regulatory and safety risk — a drug high on both is exponentially more likely to be regulated.

### Drug Safety Risk
```
Drug_Safety_Risk = Side_Effect_Severity_Score + Abuse_Potential_Score
```

### Doctor Influence Index
```
Doctor_Influence_Index = Doctor_Recommendation_Rate × Prescription_Rate
```

### Marketing Efficiency
```
Marketing_Efficiency = Annual_Sales_Volume / Marketing_Spend
```

### Log Risk Safety
```
Log_Risk_Safety = log(1 + Risk_Safety_Interaction)
```
> Log transformation reduces skew in the interaction term.

---


## 🤖 Models Trained

| Model | Technique | Notes |
|---|---|---|
| Logistic Regression | Baseline | StandardScaler applied |
| Random Forest | Ensemble | 200 estimators, no scaling |
| Logistic Regression + SMOTE | Imbalance handling | SMOTE on train only |
| Threshold Tuned (LR + SMOTE) | Probability thresholding | Optimal threshold at 0.64 |
| XGBoost | Gradient boosting | `scale_pos_weight=2.84` |
| Final XGBoost (Tuned) | GridSearchCV | Best hyperparameters applied |

**Best Hyperparameters (Final XGBoost):**

```python
XGBClassifier(
    colsample_bytree=0.7,
    gamma=0.2,
    learning_rate=0.03,
    max_depth=4,
    n_estimators=250,
    subsample=0.7,
    scale_pos_weight=2.84,
    eval_metric="logloss",
    random_state=42
)
```

---

## 📊 Results

| Model | Accuracy | Recall | Precision | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.779 | 0.374 | 0.639 | 0.472 |
| Random Forest | 0.776 | 0.370 | 0.628 | 0.466 |
| Logistic Regression + SMOTE | 0.623 | 0.771 | 0.391 | 0.519 |
| Threshold Tuned Model | 0.673 | 0.692 | 0.426 | 0.528 |
| XGBoost | 0.717 | 0.694 | 0.476 | 0.564 |
| **Final Tuned XGBoost ✅** | **0.709** | **0.725** | **0.467** | **0.568** |

**Key Insight:**
- Logistic Regression baseline recall = 37% → unacceptable (misses regulated drugs)
- XGBoost with `scale_pos_weight` raised recall to 69%
- Threshold tuning on SMOTE model further improved F1

**Primary evaluation metric: F1 Score** (balances precision and recall under class imbalance)

---

## 🔍 SHAP Explainability

SHAP (SHapley Additive exPlanations) was used to interpret the final XGBoost model.

**Top influential features (from SHAP summary plot):**

```
1. Risk_Safety_Interaction      ████████████████████  Most important
2. Regulatory_Risk_Score        ████████████████
3. Drug_Safety_Risk             ██████████████
4. Doctor_Influence_Index       ████████████
5. Abuse_Potential_Score        ██████████
```

**Insight:** Regulatory decisions are driven by safety and risk factors, not market or economic variables. This aligns with real-world pharmaceutical regulation principles.



---

## 🧰 Tech Stack

```
Language     : Python 3.x
Data         : pandas, numpy
Visualization: matplotlib, seaborn
ML Framework : scikit-learn
Boosting     : XGBoost
Imbalance    : imbalanced-learn (SMOTE)
Explainability: SHAP
Web app deployment : Streamlit
Model serialization : Joblib
```

**Install dependencies:**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap
```

---

## 📂 Project Structure

```
PharmaSentinel/
│
├── data/
│   └── drug_regulatory_classification_dataset.csv
│
├── notebooks/
│   └── Notebook.ipynb          # Full analysis and modeling
│
├── outputs/
│   ├── model_comparison.png    # Bar chart: F1 Score comparison
│   └── shap_summary.png        # SHAP feature importance plot
│
├── requirements.txt
└── README.md
```

---

## ▶️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/pharma-sentinel.git
cd pharma-sentinel
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---
### `requirements.txt`

```
streamlit
xgboost
scikit-learn
shap
pandas
numpy
matplotlib
joblib
```

## 👤 Author


**Yugal Rade**
B.Tech Computer Science — Dbatu University



*Built with ❤️ for the intersection of Data Science and Public Health.*



---

