"""Shared feature-engineering logic used by training, the API, and the Streamlit app.

Mirrors the preprocessing done in notebook/fraud_detection.ipynb so all three
consumers score transactions identically.
"""

TX_TYPE_MAP = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4,
}

MODEL_FEATURE_ORDER = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "amount_to_balance_ratio",
    "sender_drained",
    "dest_was_zero",
    "balance_error_orig",
    "balance_error_dest",
]


def engineer_features(step, tx_type, amount, sender_balance, receiver_balance):
    """Build the model input row for a single transaction.

    sender_balance / receiver_balance are the balances *before* the transaction
    (oldbalanceOrg / oldbalanceDest); newbalance* is derived the same way the
    simulator (and the original notebook) derives it.
    """
    tx_code = TX_TYPE_MAP[tx_type]

    newbalance_orig = max(sender_balance - amount, 0)
    newbalance_dest = receiver_balance + amount

    amount_to_balance_ratio = amount / (sender_balance + 1)
    sender_drained = int(newbalance_orig < 0.1 * sender_balance)
    dest_was_zero = int(receiver_balance == 0)
    balance_error_orig = sender_balance - newbalance_orig - amount
    balance_error_dest = newbalance_dest - receiver_balance - amount

    return {
        "step": step,
        "type": tx_code,
        "amount": amount,
        "oldbalanceOrg": sender_balance,
        "newbalanceOrig": newbalance_orig,
        "oldbalanceDest": receiver_balance,
        "newbalanceDest": newbalance_dest,
        "amount_to_balance_ratio": amount_to_balance_ratio,
        "sender_drained": sender_drained,
        "dest_was_zero": dest_was_zero,
        "balance_error_orig": balance_error_orig,
        "balance_error_dest": balance_error_dest,
    }


def fraud_reasons(tx_type, amount_to_balance_ratio, sender_drained, dest_was_zero):
    """Human-readable heuristic explanations shown alongside the model score."""
    reasons = []

    if amount_to_balance_ratio > 0.80:
        reasons.append("High amount relative to sender balance")

    if sender_drained:
        reasons.append("Sender account nearly drained")

    if dest_was_zero:
        reasons.append("Receiver account started empty")

    if tx_type in ("TRANSFER", "CASH_OUT"):
        reasons.append("Historically higher-risk transaction type")

    if not reasons:
        reasons.append("No major suspicious indicators found")

    return reasons
