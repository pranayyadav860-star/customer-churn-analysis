CUSTOMER CHURN INTELLIGENCE — PROFESSIONAL DASHBOARD

1. Extract this ZIP.
2. Copy your existing customer_churn_model.pkl into this folder, next to app.py.
3. Open this folder in VS Code.
4. In the terminal, run:
   python -m pip install -r requirements.txt
5. Start the app:
   python -m streamlit run app.py

Upload the original Telco Customer Churn CSV using the sidebar for Overview,
Churn Analytics and High-Risk Customers.

FEATURES:
- Professional overview and churn analytics.
- Individual churn prediction and rule-based retention suggestions.
- Bulk risk scoring, CSV export and PDF business report.
- Model Performance page with held-out metrics recorded in the notebook.
- Logistic Regression coefficient inspection when supported by the saved pipeline.
- Optional held-out test CSV upload to calculate confusion matrix, ROC curve and metrics.

OPTIONAL TEST CSV:
In Colab, after the train/test split, create and download a held-out test CSV:
    test_export = X_test.copy()
    test_export["Churn"] = y_test
    test_export.to_csv("churn_test_holdout.csv", index=False)
    from google.colab import files
    files.download("churn_test_holdout.csv")
Upload that file on the Model Performance page to calculate a real confusion matrix
and ROC curve from the held-out test set.

IMPORTANT:
- Keep customer_churn_model.pkl in the same folder as app.py.
- The model is trained for the Telco Customer Churn dataset schema.
- scikit-learn is pinned to 1.6.1 to match the training environment.
- Metrics shown by default are the values recorded in the training notebook.
- Do not use the original full dataset as a substitute for held-out test data.
- Retention suggestions are rule-based, not causal explanations.
