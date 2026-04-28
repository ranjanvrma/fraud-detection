import streamlit as st
import pandas as pd
import joblib
import time

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Fraud Risk Simulator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_assets():
    model = joblib.load("models/fraud_model.pkl")
    columns = joblib.load("models/model_columns.pkl")
    return model, columns

model, model_columns = load_assets()

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

# =====================================================
# LAYOUT
# =====================================================
left, right = st.columns([1.2, 1])

# =====================================================
# INPUT SIDE
# =====================================================
with left:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Transaction Details</div>', unsafe_allow_html=True)

    step = st.number_input("Transaction Step", min_value=1, value=2)

    tx_type = st.selectbox(
        "Transaction Type",
        ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    )

    amount = st.number_input(
        "Amount",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    sender_balance = st.number_input(
        "Sender Current Balance",
        min_value=0.0,
        value=51000.0,
        step=1000.0
    )

    receiver_balance = st.number_input(
        "Receiver Current Balance",
        min_value=0.0,
        value=0.0,
        step=1000.0
    )

    run = st.button("Analyze Fraud Risk")

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# MODEL SIDE
# =====================================================
with right:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">AI Decision Engine</div>', unsafe_allow_html=True)

    if run:

        with st.spinner("Analyzing transaction..."):
            time.sleep(1)

        # Encoding
        type_map = {
            "CASH_IN": 0,
            "CASH_OUT": 1,
            "DEBIT": 2,
            "PAYMENT": 3,
            "TRANSFER": 4
        }

        tx_code = type_map[tx_type]

        # Auto balances
        newbalanceOrig = max(sender_balance - amount, 0)
        newbalanceDest = receiver_balance + amount

        # Engineered features
        amount_to_balance_ratio = amount / (sender_balance + 1)

        sender_drained = int(
            newbalanceOrig < 0.1 * sender_balance
        )

        dest_was_zero = int(receiver_balance == 0)

        balance_error_orig = (
            sender_balance - newbalanceOrig - amount
        )

        balance_error_dest = (
            newbalanceDest - receiver_balance - amount
        )

        # Input
        row = {
            "step": step,
            "type": tx_code,
            "amount": amount,
            "oldbalanceOrg": sender_balance,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": receiver_balance,
            "newbalanceDest": newbalanceDest,
            "amount_to_balance_ratio": amount_to_balance_ratio,
            "sender_drained": sender_drained,
            "dest_was_zero": dest_was_zero,
            "balance_error_orig": balance_error_orig,
            "balance_error_dest": balance_error_dest
        }

        input_df = pd.DataFrame([row])
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # Predict
        fraud_prob = model.predict_proba(input_df)[0][1]
        score = min(fraud_prob / 0.20, 1.0) * 100

        # Threshold 0.10
        if score < 10:
            risk = "LOW RISK"
            box = "low-box"
            emoji = "✅"

        elif score < 25:
            risk = "MEDIUM RISK"
            box = "medium-box"
            emoji = "⚠️"

        else:
            risk = "HIGH RISK"
            box = "high-box"
            emoji = "🚨"

        # Result Card
        st.markdown(
            f"""
            <div class="card {box}">
                <div class="center">
                    <div style="font-size:1.3rem;font-weight:700;">
                        {emoji} {risk}
                    </div>
                    <div class="big">{score:.2f}%</div>
                    <div class="small">Fraud Risk Score</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        # Why This Score
        st.subheader("Why this score?")

        reasons = []

        if amount_to_balance_ratio > 0.80:
            reasons.append("High amount relative to sender balance")

        if sender_drained:
            reasons.append("Sender account nearly drained")

        if dest_was_zero:
            reasons.append("Receiver account started empty")

        if tx_type in ["TRANSFER", "CASH_OUT"]:
            reasons.append("Historically higher-risk transaction type")

        if not reasons:
            reasons.append("No major suspicious indicators found")

        for r in reasons:
            st.info(r)

        # Insights
        st.subheader("Transaction Insights")

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Amount Ratio", f"{amount_to_balance_ratio:.2f}")
            st.metric("Sender Drained", "Yes" if sender_drained else "No")
            st.metric("Receiver Empty", "Yes" if dest_was_zero else "No")

        with c2:
            st.metric("New Sender Balance", f"{newbalanceOrig:,.0f}")
            st.metric("New Receiver Balance", f"{newbalanceDest:,.0f}")
            st.metric("Type", tx_type)

    else:
        st.info("Enter details and click Analyze Fraud Risk.")

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.write("")
st.markdown("""
<center style="color:#94a3b8;">
Built by Ranjan • ML Fraud Detection • Streamlit
</center>
""", unsafe_allow_html=True)