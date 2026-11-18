import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt

MODEL_PATH = Path(__file__).parent / "customer_churn_model.pkl"

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Professional visual theme ----------
st.markdown("""
<style>
:root { --ink:#17243a; --muted:#718096; --line:#e7edf5; --panel:#ffffff; }
.stApp { background: #f5f7fb; color: var(--ink); }
[data-testid="stSidebar"] { background: #101d35; }
[data-testid="stSidebar"] * { color: #eef4ff !important; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] section { background:#1c2b47; border:1px dashed #6b7c99; }
.block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1450px; }
.hero {
    background: linear-gradient(115deg,#142642 0%,#244b78 58%,#287d91 100%);
    padding: 1.55rem 1.8rem; border-radius: 18px; color: white; margin-bottom: 1.2rem;
    box-shadow: 0 10px 28px rgba(22,42,70,.12);
}
.hero h1 { color:white; font-size:2rem; margin:0 0 .25rem 0; font-weight:750; }
.hero p { color:#dbeafe; margin:0; font-size:.98rem; }
.eyebrow { color:#9ed9e6; font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; font-weight:700; margin-bottom:.45rem; }
.section-title { font-size:1.25rem; font-weight:750; color:#17243a; margin:.45rem 0 .15rem 0; }
.section-sub { color:#718096; font-size:.9rem; margin-bottom:1rem; }
.kpi {
    background:white; border:1px solid #e7edf5; border-radius:14px; padding:1rem 1.1rem;
    box-shadow:0 4px 14px rgba(31,55,88,.04); min-height:112px;
}
.kpi-label { color:#718096; font-size:.82rem; font-weight:600; margin-bottom:.45rem; }
.kpi-value { color:#17243a; font-size:1.75rem; font-weight:800; line-height:1.15; }
.kpi-note { color:#8a97a9; font-size:.76rem; margin-top:.35rem; }
.result-card { background:white; border:1px solid #e7edf5; border-radius:16px; padding:1.2rem 1.3rem; }
div[data-testid="stForm"] { background:white; padding:1rem 1.2rem; border:1px solid #e7edf5; border-radius:16px; }
.stButton button, .stDownloadButton button, div[data-testid="stFormSubmitButton"] button {
    border-radius:9px; font-weight:700; min-height:2.6rem;
}
[data-testid="stMetric"] { background:white; padding:1rem; border:1px solid #e7edf5; border-radius:13px; }
hr { border-color:#e7edf5; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

def hero():
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">Predictive customer analytics</div>
      <h1>Customer Churn Intelligence</h1>
      <p>Telecom customer churn prediction, risk monitoring and retention insights</p>
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("Model file not found. Place customer_churn_model.pkl in the same folder as app.py.")
    st.stop()
try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

SERVICE_COLUMNS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                   "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
RAW_COLUMNS = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
               "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
               "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
               "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
               "MonthlyCharges", "TotalCharges"]
MODEL_COLUMNS = RAW_COLUMNS + ["Tenure_Band", "Contract_Bucket",
    "Monthly_Charges_Band", "Total_Charges_Band", "Service_Count"]

def engineer_features(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Tenure_Band"] = pd.cut(df["tenure"], [-1, 3, 6, 12, 24, 48, 72],
        labels=["0-3 Months", "4-6 Months", "7-12 Months", "13-24 Months",
                "25-48 Months", "49-72 Months"])
    df["Contract_Bucket"] = df["Contract"].map({
        "Month-to-month": "Short_Term", "One year": "Long_Term", "Two year": "Long_Term"})
    df["Monthly_Charges_Band"] = pd.cut(df["MonthlyCharges"], [0, 30, 60, 90, np.inf],
        labels=["Low", "Medium", "High", "Very_High"])
    df["Total_Charges_Band"] = pd.cut(df["TotalCharges"], [-1, 1000, 3000, 6000, 10000, np.inf],
        labels=["Very_Low", "Low", "Medium", "High", "Very_High"])
    service = df[SERVICE_COLUMNS].copy()
    for col in SERVICE_COLUMNS:
        service[col] = service[col].map({"Yes": 1, "No": 0,
            "No phone service": 0, "No internet service": 0}).fillna(0)
    df["Service_Count"] = service.sum(axis=1)
    return df

def model_input(df):
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    return engineer_features(df)[MODEL_COLUMNS]

def churn_values(series):
    return pd.to_numeric(series.map({"Yes": 1, "No": 0}).fillna(series), errors="coerce")

def kpi(label, value, note=""):
    st.markdown(f"""<div class="kpi"><div class="kpi-label">{label}</div>
    <div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>""",
    unsafe_allow_html=True)



# Held-out test metrics recorded from the model evaluation notebook.
# These are reported results, not recomputed from the full uploaded customer CSV.
HELDOUT_METRICS = {
    "Accuracy": 0.7395,
    "Precision": 0.5060,
    "Recall": 0.7914,
    "F1-score": 0.6173,
    "ROC-AUC": 0.8468,
}

def find_classifier_and_transformer(pipeline):
    """Find fitted classifier and feature transformer inside a sklearn Pipeline."""
    classifier = None
    transformer = None
    if hasattr(pipeline, "named_steps"):
        for step in pipeline.named_steps.values():
            if hasattr(step, "coef_"):
                classifier = step
            if hasattr(step, "get_feature_names_out"):
                transformer = step
    return classifier, transformer

def get_logistic_coefficients(pipeline):
    """Return encoded feature coefficients when available."""
    classifier, transformer = find_classifier_and_transformer(pipeline)
    if classifier is None or transformer is None:
        return None
    try:
        names = transformer.get_feature_names_out()
        coefs = np.asarray(classifier.coef_).reshape(-1)
        if len(names) != len(coefs):
            return None
        result = pd.DataFrame({"Encoded feature": names, "Coefficient": coefs})
        result["Direction"] = np.where(result["Coefficient"] > 0,
                                       "Higher churn odds", "Lower churn odds")
        result["Absolute coefficient"] = result["Coefficient"].abs()
        return result.sort_values("Absolute coefficient", ascending=False)
    except Exception:
        return None

def prepare_test_features(test_df):
    """Accept a held-out test CSV with raw customer fields and a Churn target."""
    if "Churn" not in test_df.columns:
        raise ValueError("The held-out test CSV must include the Churn column.")
    y = churn_values(test_df["Churn"])
    valid = y.notna()
    clean = test_df.loc[valid].copy()
    y = y.loc[valid].astype(int)
    X = model_input(clean)
    return X, y

def retention_recommendations(row, probability):
    """Rule-based retention ideas using customer attributes, not model explanations."""
    recs = []
    if row.get("Contract") == "Month-to-month":
        recs.append("Offer a suitable 1-year contract option with a clear price comparison.")
    if row.get("tenure", 0) <= 6:
        recs.append("Prioritize an early-life check-in and onboarding support.")
    if row.get("TechSupport") == "No" and row.get("InternetService") != "No":
        recs.append("Offer a technical support check-in or a relevant support plan.")
    if row.get("OnlineSecurity") == "No" and row.get("InternetService") != "No":
        recs.append("Explain available online security options and any applicable bundle pricing.")
    if row.get("PaymentMethod") == "Electronic check":
        recs.append("Offer convenient automatic payment options, without applying pressure.")
    if row.get("MonthlyCharges", 0) >= 80:
        recs.append("Review the customer's plan and explain lower-cost options where available.")
    if not recs:
        recs.append("Use a routine satisfaction check-in and review recent service interactions.")
    if probability >= .70:
        recs.insert(0, "High priority: arrange a timely, human-led retention conversation.")
    elif probability >= .40:
        recs.insert(0, "Medium priority: monitor the account and schedule a proactive check-in.")
    else:
        recs.insert(0, "Low priority: continue standard customer care and periodic monitoring.")
    return recs

def make_pdf_report(scored_df, threshold_pct):
    """Create a concise PDF summary and top-risk customer list."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4),
                            rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Customer Churn Intelligence", styles["Title"]),
        Paragraph("Customer risk scoring report", styles["Heading2"]),
        Spacer(1, 10),
    ]
    total = len(scored_df)
    churned_pred = int((scored_df["Predicted_Churn"] == 1).sum())
    high = int((scored_df["Churn_Probability"] >= .70).sum())
    threshold_count = int((scored_df["Churn_Probability"] >= threshold_pct / 100).sum())
    avg_prob = float(scored_df["Churn_Probability"].mean() * 100) if total else 0
    summary = [
        ["Metric", "Result"],
        ["Customers scored", f"{total:,}"],
        ["Predicted churn (model class)", f"{churned_pred:,}"],
        ["High risk (estimated probability ≥70%)", f"{high:,}"],
        [f"Customers at or above selected threshold ({threshold_pct}%)", f"{threshold_count:,}"],
        ["Average estimated churn probability", f"{avg_prob:.2f}%"],
    ]
    table = Table(summary, colWidths=[330, 180])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#18314f")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), .4, colors.HexColor("#d9e2ef")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f2f6fb")]),
        ("PADDING", (0,0), (-1,-1), 7),
    ]))
    story.extend([table, Spacer(1, 18), Paragraph("Top 20 highest-risk customers", styles["Heading2"])])
    top = scored_df.sort_values("Churn_Probability", ascending=False).head(20)
    fields = [c for c in ["customerID", "Contract", "tenure", "MonthlyCharges", "Churn_Probability", "Risk_Level"] if c in top.columns]
    rows = [["Customer ID" if c == "customerID" else c.replace("_", " ") for c in fields]]
    for _, record in top[fields].iterrows():
        vals = []
        for c in fields:
            v = record[c]
            if c == "Churn_Probability":
                vals.append(f"{float(v)*100:.1f}%")
            else:
                vals.append(str(v))
        rows.append(vals)
    widths = [130 if c == "customerID" else 100 for c in fields]
    top_table = Table(rows, colWidths=widths, repeatRows=1)
    top_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#287d91")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#d9e2ef")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f2f6fb")]),
        ("PADDING", (0,0), (-1,-1), 5),
    ]))
    story.extend([top_table, Spacer(1, 12),
        Paragraph("Important: probabilities are model estimates, not confirmed future cancellations. Risk bands are operational display conventions. Validate interventions against business outcomes.", styles["Italic"])])
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

with st.sidebar:
    st.markdown("## 📊 CCI")
    st.caption("CUSTOMER INTELLIGENCE")
    page = st.radio("WORKSPACE", ["Overview", "Predict a Customer",
        "Churn Analytics", "High-Risk Customers", "Model Performance"], label_visibility="visible")
    st.markdown("---")
    uploaded = st.file_uploader("Upload telecom customer CSV", type=["csv"],
        help="Use the original Telco Customer Churn CSV for analytics and bulk scoring.")
    st.markdown("---")
    st.caption("MODEL · Logistic Regression")
    st.caption("DATA · Telco Customer Churn")

data = None
if uploaded:
    try:
        data = pd.read_csv(uploaded)
    except Exception as exc:
        st.sidebar.error(f"CSV could not be read: {exc}")

hero()

if page == "Overview":
    st.markdown('<div class="section-title">Business overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">A snapshot of customer retention and churn performance.</div>', unsafe_allow_html=True)
    if data is None:
        st.info("Upload the original telecom customer CSV from the sidebar to populate the overview.")
    elif "Churn" not in data.columns:
        st.warning("The uploaded CSV must contain a Churn column.")
    else:
        churn = churn_values(data["Churn"])
        total, lost = len(data), int((churn == 1).sum())
        stayed = int((churn == 0).sum())
        rate = lost / total * 100 if total else 0
        cols = st.columns(4)
        with cols[0]: kpi("Total customers", f"{total:,}", "Customers in uploaded dataset")
        with cols[1]: kpi("Churned customers", f"{lost:,}", "Customers marked as churned")
        with cols[2]: kpi("Retained customers", f"{stayed:,}", "Customers who stayed")
        with cols[3]: kpi("Churn rate", f"{rate:.2f}%", "Share of customers churned")
        st.markdown("---")
        left, right = st.columns([1, 1.4])
        with left:
            st.markdown("**Customer status distribution**")
            st.bar_chart(pd.Series({"Retained": stayed, "Churned": lost}), color="#287d91")
        with right:
            st.markdown("**Churn rate by contract**")
            if "Contract" in data:
                temp = data.assign(_churn=churn)
                st.bar_chart(temp.groupby("Contract")["_churn"].mean().mul(100), color="#287d91")
        st.caption("Descriptive statistics from the uploaded dataset; not a forecast.")

elif page == "Predict a Customer":
    st.markdown('<div class="section-title">Individual churn assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Enter customer attributes to estimate churn probability. Engineered features are generated automatically.</div>', unsafe_allow_html=True)
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Customer profile**")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        with c2:
            st.markdown("**Services**")
            phone = st.selectbox("Phone service", ["Yes", "No"])
            lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
            backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
            protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
        with c3:
            st.markdown("**Support & billing**")
            support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
            tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
            payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly = st.number_input("Monthly charges", min_value=0.0, max_value=200.0, value=70.0)
            total_charges = st.number_input("Total charges", min_value=0.0, max_value=20000.0,
                                            value=float(70 * tenure))
        submit = st.form_submit_button("Run churn assessment", use_container_width=True)
    if submit:
        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone, "MultipleLines": lines,
            "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
            "DeviceProtection": protection, "TechSupport": support, "StreamingTV": tv,
            "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total_charges
        }])
        try:
            features = model_input(row)
            pred = int(model.predict(features)[0])
            prob = float(model.predict_proba(features)[0][1])
            risk = "High" if prob >= .70 else ("Medium" if prob >= .40 else "Low")
            st.markdown("---")
            st.markdown('<div class="section-title">Assessment result</div>', unsafe_allow_html=True)
            a, b, c = st.columns(3)
            with a: kpi("Estimated churn probability", f"{prob*100:.2f}%", "Model-estimated likelihood")
            with b: kpi("Predicted outcome", "Churn" if pred else "Stay", "Based on the model's classification")
            with c: kpi("Risk band", risk, "Low <40% · Medium 40–69.9% · High ≥70%")
            st.progress(prob)
            st.markdown("**Suggested retention actions**")
            for recommendation in retention_recommendations(row.iloc[0].to_dict(), prob):
                st.markdown(f"- {recommendation}")
            st.caption("Recommendations are rule-based suggestions from visible customer attributes; they are not explanations of the model prediction. This is a statistical estimate, not a guarantee.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

elif page == "Churn Analytics":
    st.markdown('<div class="section-title">Churn analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Explore how churn varies across customer contracts, services and tenure.</div>', unsafe_allow_html=True)
    if data is None:
        st.info("Upload the original Telco CSV from the sidebar to view analytics.")
    elif "Churn" not in data.columns:
        st.warning("The uploaded CSV must contain a Churn column.")
    else:
        data = data.copy()
        data["Churn"] = churn_values(data["Churn"])
        dims = [("Contract", "Contract type"), ("InternetService", "Internet service"),
                ("PaymentMethod", "Payment method"), ("TechSupport", "Tech support")]
        available = [(c, label) for c, label in dims if c in data.columns]
        for start in range(0, len(available), 2):
            pair = available[start:start+2]
            chart_cols = st.columns(len(pair))
            for col, (field, label) in zip(chart_cols, pair):
                with col:
                    st.markdown(f"**Churn rate by {label.lower()}**")
                    st.bar_chart(data.groupby(field)["Churn"].mean().mul(100), color="#287d91")
        if "tenure" in data:
            data["Tenure_Band"] = pd.cut(data["tenure"], [-1, 3, 6, 12, 24, 48, 72],
                labels=["0-3 Months", "4-6 Months", "7-12 Months", "13-24 Months",
                        "25-48 Months", "49-72 Months"])
            st.markdown("**Churn rate by customer tenure**")
            st.bar_chart(data.groupby("Tenure_Band", observed=True)["Churn"].mean().mul(100), color="#287d91")


elif page == "Model Performance":
    st.markdown('<div class="section-title">Model performance & explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Review the Logistic Regression model results recorded during held-out test evaluation, and inspect its learned feature coefficients.</div>', unsafe_allow_html=True)

    st.markdown("**Held-out test results from the training notebook**")
    metric_cols = st.columns(5)
    for col, (label, value) in zip(metric_cols, HELDOUT_METRICS.items()):
        with col:
            kpi(label, f"{value*100:.2f}%", "20% stratified test split")

    st.markdown("---")
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**What these metrics mean**")
        st.markdown("""
        - **Accuracy:** share of test customers classified correctly.
        - **Precision:** among customers predicted to churn, the share who actually churned.
        - **Recall:** among actual churners, the share the model successfully identified.
        - **F1-score:** balance between precision and recall.
        - **ROC-AUC:** how well the model ranks churners above non-churners across thresholds.
        """)
        st.info("Recall is important in churn work because a false negative is a customer who actually churns but was not flagged. Higher recall can also create more false alarms, so retention costs matter.")
    with right:
        st.markdown("**Evaluation context**")
        st.markdown("""
        - Model: Logistic Regression with class balancing.
        - Split: 80% training / 20% testing.
        - Split strategy: stratified by churn label.
        - Test metrics above are copied from the model evaluation notebook.
        """)
        st.warning("The original test predictions and labels are not stored inside the model file. Therefore, this page does not invent a confusion matrix or ROC curve from the full dataset.")

    st.markdown("---")
    st.markdown("**Optional: calculate confusion matrix and ROC curve from a held-out test CSV**")
    st.caption("In Colab, export X_test together with y_test to a CSV, then upload it below. The file must contain the original customer input columns plus a Churn column.")
    test_upload = st.file_uploader("Upload held-out test dataset (CSV with Churn)", type=["csv"], key="heldout_test_upload")
    if test_upload is not None:
        try:
            test_df = pd.read_csv(test_upload)
            X_holdout, y_holdout = prepare_test_features(test_df)
            pred_holdout = model.predict(X_holdout)
            prob_holdout = model.predict_proba(X_holdout)[:, 1]
            actual_metrics = {
                "Accuracy": accuracy_score(y_holdout, pred_holdout),
                "Precision": precision_score(y_holdout, pred_holdout, zero_division=0),
                "Recall": recall_score(y_holdout, pred_holdout, zero_division=0),
                "F1-score": f1_score(y_holdout, pred_holdout, zero_division=0),
                "ROC-AUC": roc_auc_score(y_holdout, prob_holdout) if y_holdout.nunique() == 2 else np.nan,
            }
            st.markdown("**Metrics calculated from the uploaded held-out file**")
            mcols = st.columns(5)
            for col, (name, value) in zip(mcols, actual_metrics.items()):
                with col:
                    kpi(name, f"{value*100:.2f}%" if pd.notna(value) else "N/A", "Uploaded hold-out data")
            cm = confusion_matrix(y_holdout, pred_holdout, labels=[0, 1])
            fig, ax = plt.subplots()
            image = ax.imshow(cm)
            ax.set_title("Confusion matrix")
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("Actual label")
            ax.set_xticks([0, 1], labels=["Stay (0)", "Churn (1)"])
            ax.set_yticks([0, 1], labels=["Stay (0)", "Churn (1)"])
            for (i, j), value in np.ndenumerate(cm):
                ax.text(j, i, str(value), ha="center", va="center")
            fig.colorbar(image, ax=ax)
            st.pyplot(fig)
            plt.close(fig)

            if y_holdout.nunique() == 2:
                fpr, tpr, _ = roc_curve(y_holdout, prob_holdout)
                fig2, ax2 = plt.subplots()
                ax2.plot(fpr, tpr, label=f"ROC-AUC = {actual_metrics['ROC-AUC']:.3f}")
                ax2.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
                ax2.set_xlabel("False positive rate")
                ax2.set_ylabel("True positive rate")
                ax2.set_title("ROC curve")
                ax2.legend()
                st.pyplot(fig2)
                plt.close(fig2)
            st.download_button("Download test-set predictions",
                pd.DataFrame({"Actual_Churn": y_holdout, "Predicted_Churn": pred_holdout,
                              "Churn_Probability": prob_holdout}).to_csv(index=False).encode("utf-8"),
                file_name="churn_test_predictions.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"Could not evaluate the uploaded test file: {exc}")
            st.caption("Check that it contains the raw Telco customer columns used during training and a Churn target.")

    st.markdown("---")
    st.markdown("**Learned Logistic Regression coefficients**")
    st.caption("Positive coefficients push the model score toward churn; negative coefficients push it toward stay, holding other encoded inputs constant. These are model associations, not causal effects.")
    coefficient_df = get_logistic_coefficients(model)
    if coefficient_df is not None:
        top_n = st.slider("Number of encoded features to display", 5, min(25, len(coefficient_df)), min(15, len(coefficient_df)))
        display = coefficient_df.head(top_n).sort_values("Coefficient")
        st.bar_chart(display.set_index("Encoded feature")["Coefficient"])
        st.dataframe(coefficient_df.head(top_n)[["Encoded feature", "Coefficient", "Direction"]],
                     use_container_width=True, hide_index=True)
    else:
        st.info("The loaded model does not expose a compatible fitted Logistic Regression coefficient table. The held-out evaluation metrics above are still available.")

elif page == "High-Risk Customers":
    st.markdown('<div class="section-title">High-risk customer analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Score the uploaded customer file and prioritize accounts with the highest estimated churn probability.</div>', unsafe_allow_html=True)
    if data is None:
        st.info("Upload the original Telco CSV from the sidebar to score all customers.")
    else:
        try:
            features = model_input(data)
            scored = data.copy()
            scored["Churn_Probability"] = model.predict_proba(features)[:, 1]
            scored["Predicted_Churn"] = model.predict(features)
            scored["Risk_Level"] = pd.cut(scored["Churn_Probability"],
                bins=[-0.001, .4, .7, 1.001], labels=["Low", "Medium", "High"])
            scored["Suggested_Action"] = [
                " | ".join(retention_recommendations(record, float(prob))[:3])
                for record, prob in zip(data.to_dict(orient="records"), scored["Churn_Probability"])
            ]
            scored = scored.sort_values("Churn_Probability", ascending=False)
            total = len(scored)
            high = int((scored["Churn_Probability"] >= .7).sum())
            medium = int(((scored["Churn_Probability"] >= .4) & (scored["Churn_Probability"] < .7)).sum())
            low = total - high - medium
            a, b, c, d = st.columns(4)
            with a: kpi("Customers scored", f"{total:,}", "Rows successfully evaluated")
            with b: kpi("High risk", f"{high:,}", "Estimated probability ≥70%")
            with c: kpi("Medium risk", f"{medium:,}", "Estimated probability 40–69.9%")
            with d: kpi("Low risk", f"{low:,}", "Estimated probability <40%")
            st.markdown("---")
            threshold = st.slider("Minimum churn probability to display", 0, 100, 50, 5)
            filtered = scored[scored["Churn_Probability"] >= threshold/100]
            cols = [c for c in ["customerID", "Contract", "tenure", "MonthlyCharges",
                "Churn_Probability", "Risk_Level", "Predicted_Churn"] if c in filtered.columns]
            view = filtered[cols].head(100).copy()
            if "Churn_Probability" in view:
                view["Churn_Probability"] = (view["Churn_Probability"] * 100).round(2)
            st.markdown(f"**{len(filtered):,} customers meet the selected threshold** · displaying up to 100 highest-risk records")
            st.dataframe(view, use_container_width=True, hide_index=True)
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button("Download all scored customers (CSV)",
                    scored.to_csv(index=False).encode("utf-8"),
                    file_name="customer_churn_risk_scores.csv", mime="text/csv",
                    use_container_width=True)
            with dl2:
                pdf_bytes = make_pdf_report(scored, threshold)
                st.download_button("Download business summary (PDF)",
                    data=pdf_bytes, file_name="customer_churn_business_report.pdf",
                    mime="application/pdf", use_container_width=True)
            st.caption("Risk bands and threshold are operational display conventions. Validate them against business costs before using them for interventions.")
        except Exception as exc:
            st.error(f"Bulk scoring failed: {exc}")
            st.caption("Confirm that the CSV has the same Telco columns used to train this model.")

st.markdown("---")
st.caption("Customer Churn Intelligence · Predictive analytics prototype · Model outputs should support, not replace, business judgment.")
