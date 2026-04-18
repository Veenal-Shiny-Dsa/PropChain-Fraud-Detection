"""
api.py — PropChain FastAPI Backend
All endpoints serve the DApp frontend.
"""
import os, sys, json, time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

_SRC  = Path(__file__).parent
_BACK = _SRC.parent
if str(_SRC) not in sys.path: sys.path.insert(0, str(_SRC))

from fraud_engine import full_score, FEATURES

app = FastAPI(title="PropChain API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Schemas ────────────────────────────────────────────────────────
class ScoreRequest(BaseModel):
    tx_id:          str
    property_id:    str
    tx_type:        str   = "Registration"  # Registration | Sale | Transfer | Mortgage
    sale_price:     float
    assessed_value: float
    days_since_last_sale:  float = 730
    transfers_12mo:        int   = 0
    num_liens:             int   = 0
    docs_last_7_days:      int   = 0
    has_inspection:        bool  = True
    mortgage_to_value:     float = 0.75
    area_avg_price:        float = 0
    wallet_sent_count:     int   = 1
    wallet_recv_count:     int   = 1
    wallet_total_sent:     float = 0
    wallet_total_received: float = 0
    unique_agents:         int   = 1
    unique_senders:        int   = 2

class ManualScore(BaseModel):
    features: dict

# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/health")
def health():
    from fraud_engine import _ml
    return {"status":"ok","ml_ready":_ml.ready,"models":list(_ml.models.keys())}

@app.post("/score")
def score(req: ScoreRequest):
    tx = {"value": req.sale_price, "timestamp": int(time.time()), "tx_type": req.tx_type}
    history = {
        "assessedValue":       req.assessed_value or req.sale_price,
        "areaAvg":             req.area_avg_price or req.assessed_value or req.sale_price,
        "lastSaleTs":          int(time.time()) - req.days_since_last_sale*86400,
        "transfers12mo":       req.transfers_12mo,
        "numLiens":            req.num_liens,
        "docsWeek":            req.docs_last_7_days,
        "mortgageRatio":       req.mortgage_to_value,
        "walletTotalSent":     req.wallet_total_sent,
        "walletTotalReceived": req.wallet_total_received,
        "walletSentCount":     req.wallet_sent_count,
        "walletReceivedCount": req.wallet_recv_count,
        "uniqueSenders":       req.unique_senders,
        "uniqueAgents":        req.unique_agents,
        "avgMinSent": 1440, "avgMinRecv": 1440, "timeDiff": 43200,
        "contractsCreated": 0, "uniqueRecipients": req.wallet_sent_count,
        "minRecv": 0, "maxRecv": req.sale_price,
        "minSent": 0, "maxSent": req.sale_price,
    }
    result = full_score(tx, history)
    return {**result, "tx_id": req.tx_id, "property_id": req.property_id}

@app.post("/score/manual")
def score_manual(req: ManualScore):
    tx = {"value": req.features.get("sale_price",100), "timestamp": int(time.time())}
    history = {"assessedValue": req.features.get("assessed_value",100), **req.features}
    return full_score(tx, history)

@app.get("/features")
def get_features():
    return {"features": FEATURES, "count": len(FEATURES)}

@app.get("/models/metrics")
def get_metrics():
    path = os.path.join(_BACK, "models", "training_results.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
