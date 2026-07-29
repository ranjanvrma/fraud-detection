"""FastAPI service exposing the fraud detection model.

Run locally:
    uvicorn api.main:app --reload

Run in Docker: see api/Dockerfile / docker-compose.yml
"""

from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from fraud_detection.features import engineer_features, fraud_reasons

MODEL_PATH = "models/fraud_model.pkl"
COLUMNS_PATH = "models/model_columns.pkl"
SCORE_CAP = 0.20  # matches app.py: fraud_prob / 0.20, capped at 100%

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["model"] = joblib.load(MODEL_PATH)
    _state["columns"] = joblib.load(COLUMNS_PATH)
    yield
    _state.clear()


app = FastAPI(title="Fraud Risk API", version="1.0.0", lifespan=lifespan)


class TransactionRequest(BaseModel):
    step: int = Field(..., ge=1, description="Transaction time step")
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    amount: float = Field(..., ge=0)
    sender_balance: float = Field(..., ge=0, description="Sender balance before the transaction")
    receiver_balance: float = Field(..., ge=0, description="Receiver balance before the transaction")


class TransactionResponse(BaseModel):
    fraud_probability: float
    risk_score: float
    risk_level: Literal["LOW RISK", "MEDIUM RISK", "HIGH RISK"]
    reasons: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in _state}


@app.post("/predict", response_model=TransactionResponse)
def predict(tx: TransactionRequest):
    row = engineer_features(
        tx.step, tx.type, tx.amount, tx.sender_balance, tx.receiver_balance
    )
    input_df = pd.DataFrame([row]).reindex(columns=_state["columns"], fill_value=0)

    fraud_prob = float(_state["model"].predict_proba(input_df)[0][1])
    score = min(fraud_prob / SCORE_CAP, 1.0) * 100

    if score < 10:
        risk_level = "LOW RISK"
    elif score < 25:
        risk_level = "MEDIUM RISK"
    else:
        risk_level = "HIGH RISK"

    reasons = fraud_reasons(
        tx.type, row["amount_to_balance_ratio"], row["sender_drained"], row["dest_was_zero"]
    )

    return TransactionResponse(
        fraud_probability=fraud_prob,
        risk_score=round(score, 2),
        risk_level=risk_level,
        reasons=reasons,
    )
