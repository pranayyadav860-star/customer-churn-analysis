# 📊 Customer Churn Intelligence

Machine Learning-based telecom customer churn prediction and analytics dashboard.

## 📌 Project Overview

Customer Churn Intelligence is an end-to-end Machine Learning project that analyzes telecom customer data, identifies churn patterns, and predicts customers who are likely to leave.

The project combines data analytics, machine learning, and an interactive Streamlit dashboard to support customer retention strategies.

## 🎯 Objectives

- Analyze customer churn patterns.
- Identify high-risk customer segments.
- Perform feature engineering.
- Build and compare ML classification models.
- Predict individual and bulk customer churn.
- Generate customer retention recommendations.

## 📊 Dataset

**Telco Customer Churn Dataset**

- Total Customers: 7,043
- Original Features: 21
- Target Variable: Churn
- Problem Type: Binary Classification

## 🛠️ Technologies Used

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- Google Colab
- Streamlit
- Joblib
- Git & GitHub

## ⚙️ Project Workflow

1. Data Loading and Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Data Preprocessing
5. Machine Learning Model Training
6. Model Evaluation
7. Customer Churn Prediction
8. Business Recommendations

## 🤖 Machine Learning Models

Two classification algorithms were implemented:

- Logistic Regression
- Random Forest Classifier

### Model Performance

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 73.95% | 77.79% |
| Precision | 50.60% | 60.41% |
| Recall | 79.14% | 47.33% |
| F1-score | 61.73% | 53.07% |
| ROC-AUC | 84.68% | 81.93% |

Logistic Regression was selected for the prediction dashboard because of its higher recall and F1-score.

## 💻 Streamlit Dashboard

The interactive dashboard includes:

- **Overview:** Customer statistics and churn insights.
- **Individual Prediction:** Predict customer churn probability.
- **Churn Analytics:** Visualize churn patterns.
- **High-Risk Customers:** Identify and download high-risk customer records.
- **Model Performance:** Evaluate model metrics and explainability.

Additional features include retention recommendations and downloadable CSV/PDF reports.

## 📁 Project Structure

```text
customer-churn-analysis/
│
├── notebooks/
│   └── Customer_Churn_Analysis.ipynb
├── models/
│   └── customer_churn_model.pkl
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ▶️ Run Locally

Clone the repository:

```bash
git clone https://github.com/pranayyadav860-star/customer-churn-analysis.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
python -m streamlit run app.py
```

## 💡 Business Impact

The project helps businesses:

- Identify customers at risk of leaving.
- Understand factors associated with churn.
- Prioritize customer retention activities.
- Develop targeted retention strategies.
- Make data-driven business decisions.

## 👨‍💻 Author

**Pranay Yadav**

B.Tech – Computer Science and Engineering

GitHub: [pranayyadav860-star](https://github.com/pranayyadav860-star)

Project: [Customer Churn Analysis](https://github.com/pranayyadav860-star/customer-churn-analysis)