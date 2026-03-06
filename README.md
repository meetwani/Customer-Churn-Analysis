# 📡 Telco Customer Churn — Segmentation & Retention Analysis

> Predict which telecom customers are likely to churn — and help the business retain them.

---

## 🔗 Live Demo

👉 **[Try the App Here](https://your-app-link.streamlit.app)** ← *(update after Streamlit deploy)*

---

## 📌 Project Overview

Every telecom company loses revenue when customers leave. This project builds a **Machine Learning model** that predicts customer churn based on their behavior, account info, and services — enabling the business to take proactive retention action.

**Key Questions Answered:**
- Which customers are likely to churn?
- What factors drive churn the most?
- Who should get a retention offer?

---

## 📊 Dataset

- **Source:** [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Records:** 7,043 customers
- **Features:** 19 (demographics, services, account info)
- **Target:** `Churn` → Yes / No

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | ML models & preprocessing |
| XGBoost | Boosting model |
| SMOTE (imbalanced-learn) | Handle class imbalance |
| Streamlit | Web app deployment |
| Pickle | Model saving & loading |

---

## 🔍 Project Workflow

```
1. Data Loading & Cleaning
        ↓
2. Exploratory Data Analysis (EDA)
        ↓
3. Label Encoding (Categorical Features)
        ↓
4. SMOTE (Handle Class Imbalance)
        ↓
5. Model Training & Cross Validation
   → Decision Tree
   → Random Forest ✅ (Best)
   → XGBoost
        ↓
6. Model Evaluation
        ↓
7. Streamlit Web App Deployment
```

---

## 🤖 Model Performance

| Model | CV Accuracy |
|-------|------------|
| Decision Tree | ~79% |
| **Random Forest** | **~86% ✅** |
| XGBoost | ~84% |

> **Random Forest** selected as final model — highest accuracy with stable cross-validation scores.

---

## 🌐 Web App Features

- 📋 Enter all 19 customer attributes via form
- 🔍 Predict churn with one click
- 📊 See churn probability %
- 💡 Get business insight & retention recommendation

---

## 📁 Project Structure

```
📁 Customer-Churn-Analysis/
   ├── app.py                      # Streamlit web app
   ├── Teleco.ipynb                # Main analysis notebook
   ├── Teleco.csv                  # Dataset
   ├── customer churn model.pkl    # Trained Random Forest model
   ├── encoders.pikle              # Label encoders
   └── requirements.txt            # Dependencies
```

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/meetwani/Customer-Churn-Analysis.git

# Go to project folder
cd Customer-Churn-Analysis

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 💡 Business Insights

- **Month-to-month** contract customers churn the most
- **New customers** (low tenure) are at highest risk
- **High monthly charges** without added services → churn trigger
- Customers without **Tech Support or Online Security** leave more often

---

## 👤 Author

**Meet Wani**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/meetwani/)
[![GitHub](https://img.shields.io/badge/GitHub-meetwani-black?logo=github)](https://github.com/meetwani)

---

*Built as part of Data Science portfolio — Customer Segmentation & Retention Analysis*