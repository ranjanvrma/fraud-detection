import time

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fraud_detection.features import TX_TYPE_MAP, engineer_features, fraud_reasons

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Fraud Risk Simulator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

SCORE_CAP = 0.20  # fraud_prob / 0.20, capped at 100%

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_assets():
    model = joblib.load("models/fraud_model.pkl")
    columns = joblib.load("models/model_columns.pkl")
    return model, columns

model, model_columns = load_assets()


def score_transaction(step, tx_type, amount, sender_balance, receiver_balance):
    row = engineer_features(step, tx_type, amount, sender_balance, receiver_balance)
    input_df = pd.DataFrame([row]).reindex(columns=model_columns, fill_value=0)

    fraud_prob = model.predict_proba(input_df)[0][1]
    score = min(fraud_prob / SCORE_CAP, 1.0) * 100

    if score < 10:
        risk, emoji, box = "LOW RISK", "✅", "low-box"
    elif score < 25:
        risk, emoji, box = "MEDIUM RISK", "⚠️", "medium-box"
    else:
        risk, emoji, box = "HIGH RISK", "🚨", "high-box"

    return row, fraud_prob, score, risk, emoji, box


def risk_gauge(score, risk):
    color = {"LOW RISK": "#22c55e", "MEDIUM RISK": "#eab308", "HIGH RISK": "#ef4444"}[risk]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%", "font": {"size": 40, "color": "white"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white"},
            "bar": {"color": color},
            "bgcolor": "rgba(255,255,255,0.05)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 10], "color": "rgba(34,197,94,0.15)"},
                {"range": [10, 25], "color": "rgba(234,179,8,0.15)"},
                {"range": [25, 100], "color": "rgba(239,68,68,0.15)"},
            ],
        },
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig


def feature_importance_chart(top_n=6):
    importances = pd.Series(model.feature_importances_, index=model_columns)
    importances = importances.sort_values(ascending=True).tail(top_n)

    fig = go.Figure(go.Bar(
        x=importances.values,
        y=importances.index,
        orientation="h",
        marker_color="#6366f1",
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        xaxis={"title": "Model importance", "gridcolor": "rgba(255,255,255,0.08)"},
        yaxis={"title": ""},
    )
    return fig

# =====================================================
# PREMIUM CSS (NO BROKEN HTML VERSION)
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background:
    radial-gradient(circle at top left, #312e81 0%, transparent 30%),
    radial-gradient(circle at bottom right, #0f766e 0%, transparent 25%),
    linear-gradient(135deg, #020617 0%, #081028 100%);
    color: white;
}

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(18px);
    border-radius: 26px;
    padding: 1.5rem;
}

/* Buttons */
.stButton>button {
    width: 100%;
    height: 3.2rem;
    border-radius: 18px;
    border: none;
    color: white;
    font-weight: 600;
    font-size: 1rem;
    background: linear-gradient(90deg,#6366f1,#06b6d4);
    transition: 0.25s ease;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 24px rgba(99,102,241,.35);
}

/* Metrics */
.metric-box {
    background: rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 0.8rem;
    text-align:center;
}

/* Result Colors */
.low-box {
    background: linear-gradient(135deg,#052e16,#166534);
}

.medium-box {
    background: linear-gradient(135deg,#3f2b05,#a16207);
}

.high-box {
    background: linear-gradient(135deg,#450a0a,#b91c1c);
}

.center {
    text-align:center;
}

.small {
    color:#cbd5e1;
    font-size:0.95rem;
}

.big {
    font-size:3rem;
    font-weight:700;
}

.section-title {
    font-size:1.8rem;
    font-weight:700;
    margin-bottom:0.6rem;
}

.hero-title {
    font-size:2.5rem;
    font-weight:700;
}

.hero-sub {
    color:#cbd5e1;
    margin-top:0.35rem;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div class="card">
<div class="hero-title">🛡️ Fraud Risk Simulator</div>
<div class="hero-sub">
AI-powered payment fraud detection using Random Forest + engineered behavioral features.
</div>
</div>
""", unsafe_allow_html=True)

st.write("")

tab_single, tab_batch = st.tabs(["🔍 Single Transaction", "📂 Batch Upload"])

# =====================================================
# TAB 1 — SINGLE TRANSACTION
# =====================================================
with tab_single:

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Transaction Details</div>', unsafe_allow_html=True)

        step = st.number_input("Transaction Step", min_value=1, value=2)

        tx_type = st.selectbox("Transaction Type", list(TX_TYPE_MAP.keys()))

        amount = st.number_input("Amount", min_value=0.0, value=50000.0, step=1000.0)

        sender_balance = st.number_input(
            "Sender Current Balance", min_value=0.0, value=51000.0, step=1000.0
        )

        receiver_balance = st.number_input(
            "Receiver Current Balance", min_value=0.0, value=0.0, step=1000.0
        )

        run = st.button("Analyze Fraud Risk")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">AI Decision Engine</div>', unsafe_allow_html=True)

        if run:
            with st.spinner("Analyzing transaction..."):
                time.sleep(0.6)

            row, fraud_prob, score, risk, emoji, box = score_transaction(
                step, tx_type, amount, sender_balance, receiver_balance
            )

            st.markdown(f'<div class="{box}" style="border-radius:18px;padding:0.6rem;">'
                        f'<div class="center" style="font-size:1.3rem;font-weight:700;">'
                        f'{emoji} {risk}</div></div>', unsafe_allow_html=True)

            st.plotly_chart(risk_gauge(score, risk), use_container_width=True)

            st.subheader("Why this score?")
            for r in fraud_reasons(
                tx_type, row["amount_to_balance_ratio"], row["sender_drained"], row["dest_was_zero"]
            ):
                st.info(r)

            st.subheader("What the model weighs most")
            st.plotly_chart(feature_importance_chart(), use_container_width=True)

            st.subheader("Transaction Insights")
            c1, c2 = st.columns(2)

            with c1:
                st.metric("Amount Ratio", f"{row['amount_to_balance_ratio']:.2f}")
                st.metric("Sender Drained", "Yes" if row["sender_drained"] else "No")
                st.metric("Receiver Empty", "Yes" if row["dest_was_zero"] else "No")

            with c2:
                st.metric("New Sender Balance", f"{row['newbalanceOrig']:,.0f}")
                st.metric("New Receiver Balance", f"{row['newbalanceDest']:,.0f}")
                st.metric("Type", tx_type)

        else:
            st.info("Enter details and click Analyze Fraud Risk.")

        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# TAB 2 — BATCH UPLOAD
# =====================================================
with tab_batch:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Score a Batch of Transactions</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="small">Upload a CSV with columns: '
        '<code>step, type, amount, sender_balance, receiver_balance</code></div>',
        unsafe_allow_html=True,
    )
    st.write("")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            required_cols = {"step", "type", "amount", "sender_balance", "receiver_balance"}
            missing = required_cols - set(batch_df.columns)

            if missing:
                st.error(f"Missing required columns: {', '.join(sorted(missing))}")
            else:
                with st.spinner(f"Scoring {len(batch_df):,} transactions..."):
                    results = []
                    for _, r in batch_df.iterrows():
                        _, fraud_prob, score, risk, _, _ = score_transaction(
                            r["step"], r["type"], r["amount"],
                            r["sender_balance"], r["receiver_balance"]
                        )
                        results.append({
                            "fraud_probability": round(fraud_prob, 4),
                            "risk_score": round(score, 2),
                            "risk_level": risk,
                        })

                scored_df = pd.concat([batch_df.reset_index(drop=True), pd.DataFrame(results)], axis=1)

                st.success(f"Scored {len(scored_df):,} transactions — "
                           f"{(scored_df['risk_level'] == 'HIGH RISK').sum()} flagged HIGH RISK")

                st.dataframe(scored_df, use_container_width=True)

                st.download_button(
                    "Download Results CSV",
                    scored_df.to_csv(index=False).encode("utf-8"),
                    file_name="fraud_risk_results.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Could not process file: {e}")
    else:
        st.info("Upload a CSV to score multiple transactions at once.")

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.write("")
st.markdown("""
<center style="color:#94a3b8;">
Built by Ranjan • ML Fraud Detection • Streamlit + FastAPI
</center>
""", unsafe_allow_html=True)
