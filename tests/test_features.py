from fraud_detection.features import engineer_features, fraud_reasons


def test_engineer_features_normal_payment():
    row = engineer_features(
        step=2, tx_type="PAYMENT", amount=1000,
        sender_balance=10000, receiver_balance=5000,
    )

    assert row["type"] == 3
    assert row["newbalanceOrig"] == 9000
    assert row["newbalanceDest"] == 6000
    assert row["sender_drained"] == 0
    assert row["dest_was_zero"] == 0


def test_engineer_features_sender_drained():
    row = engineer_features(
        step=1, tx_type="TRANSFER", amount=9900,
        sender_balance=10000, receiver_balance=0,
    )

    assert row["newbalanceOrig"] == 100
    assert row["sender_drained"] == 1
    assert row["dest_was_zero"] == 1


def test_engineer_features_amount_exceeds_balance_clamps_to_zero():
    row = engineer_features(
        step=1, tx_type="CASH_OUT", amount=5000,
        sender_balance=1000, receiver_balance=0,
    )

    assert row["newbalanceOrig"] == 0
    assert row["sender_drained"] == 1


def test_fraud_reasons_high_risk_transfer():
    reasons = fraud_reasons(
        tx_type="TRANSFER", amount_to_balance_ratio=0.95,
        sender_drained=1, dest_was_zero=1,
    )

    assert any("relative to sender balance" in r for r in reasons)
    assert any("drained" in r for r in reasons)
    assert any("empty" in r for r in reasons)
    assert any("higher-risk" in r for r in reasons)


def test_fraud_reasons_no_indicators():
    reasons = fraud_reasons(
        tx_type="PAYMENT", amount_to_balance_ratio=0.1,
        sender_drained=0, dest_was_zero=0,
    )

    assert reasons == ["No major suspicious indicators found"]
