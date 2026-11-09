import streamlit as st
import pandas as pd
from src.model import train_model, score_applicant
from src.retrieval import PolicyRetriever

st.set_page_config(page_title="Credit Risk Intelligence Copilot", page_icon="📊", layout="wide")
st.title("📊 Credit Risk Intelligence Copilot")
st.caption("Explainable ML + policy-document retrieval | Educational demo with synthetic data")

@st.cache_resource
def load_model():
    return train_model()

@st.cache_resource
def load_retriever():
    return PolicyRetriever()

model = load_model()
retriever = load_retriever()

tab1, tab2, tab3 = st.tabs(["Applicant risk demo", "Policy Q&A retrieval", "Model evaluation"])

with tab1:
    st.subheader("Illustrative applicant assessment")
    c1, c2 = st.columns(2)
    with c1:
        bureau = st.slider("Bureau score", 450, 850, 700)
        income = st.number_input("Monthly income (₹)", 12000, 300000, 65000, step=1000)
        loan = st.number_input("Loan amount (₹)", 25000, 900000, 250000, step=5000)
        employment = st.slider("Employment history (years)", 0.0, 35.0, 4.0, 0.5)
    with c2:
        emi_ratio = st.slider("Existing EMI / monthly income", 0.0, 1.2, 0.25, 0.01)
        past_due = st.slider("Past-due count (historical)", 0, 5, 0)
        lti = loan / (income * 12)
        st.metric("Loan-to-annual-income", f"{lti:.2f}x")
    if st.button("Assess illustrative risk", type="primary"):
        result = score_applicant(model, {
            "bureau_score": bureau, "monthly_income": income, "loan_amount": loan,
            "loan_to_income": lti, "existing_emi_ratio": emi_ratio,
            "employment_years": employment, "past_due_count": past_due
        })
        st.metric("Model risk score", f"{result['risk_score']:.1%}")
        st.write(f"**Risk band:** {result['risk_band']}")
        st.write("**Largest coefficient-based drivers**")
        st.dataframe(pd.DataFrame(result["top_drivers"]), use_container_width=True)
        st.warning(result["disclaimer"])

with tab2:
    st.subheader("Search the fictional lending policy")
    question = st.text_input("Ask a policy question", placeholder="What should be checked for a thin-file applicant?")
    if st.button("Retrieve policy passages") and question.strip():
        results = retriever.search(question, top_k=3)
        if not results:
            st.info("No relevant passage found. Try a more specific question.")
        for i, item in enumerate(results, 1):
            with st.expander(f"Passage {i} · {item['source']} · similarity {item['score']:.3f}", expanded=True):
                st.write(item["text"])
        st.caption("This is retrieval, not generated legal or credit advice. Verify the original policy.")

with tab3:
    st.subheader("Held-out test-set metrics")
    m = model.metrics
    a, b, c, d = st.columns(4)
    a.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    b.metric("Precision", f"{m['precision']:.3f}")
    c.metric("Recall", f"{m['recall']:.3f}")
    d.metric("Test rows", m["test_rows"])
    st.write("Confusion matrix (rows = actual, columns = predicted)")
    st.dataframe(pd.DataFrame(m["confusion_matrix"],
                              index=["Actual 0", "Actual 1"],
                              columns=["Predicted 0", "Predicted 1"]))
    st.caption("Metrics are generated from synthetic data and are not evidence of real-world model performance.")
