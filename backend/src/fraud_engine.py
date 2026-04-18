"""
fraud_engine.py
Rule-based + ML fraud detection for real estate transactions.
Works without training data (rule engine).
Activates ML models if trained weights exist.

Fraud types detected:
  1. Fake ownership claims      → wallet profile anomalies
  2. Abnormal tx frequency      → rapid flip, high velocity
  3. Suspicious transfers       → price manipulation, mortgage fraud
"""
import os, sys, json, time, hashlib, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)

_SRC  = Path(__file__).parent
_BACK = _SRC.parent
_DATA = _BACK / "data"
_MDLS = _BACK / "models"

# ── Feature columns (25 features) ─────────────────────────────────
FEATURES = [
    "avg_min_between_sent","avg_min_between_received","time_diff_first_last",
    "sent_txns","received_txns","contracts_created",
    "unique_received_from","unique_sent_to",
    "min_val_received","max_val_received","avg_val_received",
    "min_val_sent","max_val_sent","avg_val_sent",
    "total_eth_sent","total_eth_received","total_eth_balance"
]

# ── Rules mapped to the 3 fraud types in the problem statement ─────
# Each tuple: (feature, op, threshold, weight, label, fraud_type, skip_for_registration)
RULES = [
    # Fake ownership claims — skip on Registration (new wallets are expected)
    ("unique_received_from", "<",   2,   0.30, "Fake ownership claim",       "Fake ownership / identity fraud", True),
    ("unique_agents",        ">",   4,   0.20, "Too many agents involved",   "Fake ownership / identity fraud", False),
    # Abnormal transaction frequency — skip day-based checks on Registration
    ("sent_txns",            ">", 200,   0.25, "High transaction velocity",  "Abnormal transaction frequency",  False),
    ("transfers_12mo",       ">",   3,   0.30, "Unusual transfer frequency", "Abnormal transaction frequency",  True),
    ("days_since_last_sale", "<",  30,   0.40, "Rapid flip (<30 days)",      "Abnormal transaction frequency",  True),
    ("days_since_last_sale", "<",  90,   0.20, "Short holding period",       "Abnormal transaction frequency",  True),
    # Suspicious ownership transfers
    ("price_to_assessed_ratio",">", 1.5, 0.45, "Price 1.5x above assessed",  "Suspicious ownership transfer",   False),
    ("price_to_assessed_ratio",">", 1.25,0.20, "Price above assessed value", "Suspicious ownership transfer",   False),
    ("mortgage_to_value",    ">",  1.0,  0.35, "Loan exceeds property value","Suspicious ownership transfer",   False),
    ("mortgage_to_value",    ">",  0.95, 0.15, "Near-full mortgage",         "Suspicious ownership transfer",   False),
    ("price_vs_avg",         ">",  0.4,  0.20, "Price above neighbourhood",  "Suspicious ownership transfer",   False),
    ("price_vs_avg",         "<", -0.35, 0.15, "Suspiciously below market",  "Suspicious ownership transfer",   False),
    # Document fraud
    ("doc_frequency",        ">",   5,   0.20, "Document spam",              "Fake ownership / identity fraud", False),
    ("num_liens",            ">",   2,   0.20, "Multiple liens stacked",     "Suspicious ownership transfer",   False),
    # Appraisal Fraud - Catching users lying about assessed value
    ("assessed_vs_avg",      ">",   1.0, 0.50, "Suspicious Assessed Value",  "Fake ownership / identity fraud", False),
]

def rule_score(f: dict, is_registration: bool = False):
    score = 0.0; flags = []; fraud_type = "Suspicious pattern"; reason = ""
    for feat,op,thresh,w,lbl,ftype,skip_reg in RULES:
        if is_registration and skip_reg:
            continue  # skip history-based rules for first-time registrations
        v = f.get(feat,0)
        hit=(op==">" and v>thresh)or(op=="<" and v<thresh)or(op=="==" and v==thresh)
        if hit:
            score += w; flags.append(lbl)
            if not reason: reason=lbl; fraud_type=ftype
    return min(score,1.0), reason or "No pattern", fraud_type, flags

# ── Optional ML models ─────────────────────────────────────────────
class MLScorer:
    def __init__(self):
        self.models={}; self.scaler=None; self.ready=False; self._load()

    def _load(self):
        dp = _DATA/"processed_data.pt"
        if not dp.exists(): return
        try:
            import torch, numpy as np
            sys.path.insert(0,str(_SRC))
            from model import create_model, XGBoostDetector
            b = torch.load(dp, weights_only=False)
            self.scaler = b["scaler"]
            nf = b["data"].x.shape[1]
            for m in ["GCN","GAT","GraphSAGE"]:
                p = _MDLS/f"{m}_best.pth"
                if p.exists():
                    mdl=create_model(m,nf); mdl.eval()
                    mdl.load_state_dict(torch.load(p,map_location="cpu"))
                    self.models[m]=mdl
            xp = _MDLS/"xgboost_best.json"
            if xp.exists():
                xgb=XGBoostDetector(); xgb.load(str(xp))
                self.models["XGBoost"]=xgb
            self.ready = len(self.models)>0
            log.info(f"[ML] Models loaded: {list(self.models.keys())}")
        except Exception as e:
            log.warning(f"[ML] Load error: {e}")

    def score(self, f: dict):
        if not self.ready: return {}
        import torch, numpy as np
        vec = np.array([[f.get(c,0.0) for c in FEATURES]],dtype=np.float32)
        if self.scaler: vec=self.scaler.transform(vec)
        out={}
        for name,m in self.models.items():
            try:
                if name=="XGBoost":
                    out[name]=float(m.predict_proba(vec)[0,1])
                else:
                    x=torch.tensor(vec,dtype=torch.float)
                    ei=torch.tensor([[0],[0]],dtype=torch.long)
                    with torch.no_grad():
                        out[name]=torch.exp(m(x,ei))[0,1].item()
            except: pass
        return out

_ml = MLScorer()

def extract_features(tx: dict, history: dict) -> dict:
    sale  = tx.get("value",0)
    av    = history.get("assessedValue", sale) or sale
    avg_n = history.get("areaAvg", av) or av
    last  = history.get("lastSaleTs",0)
    now   = tx.get("timestamp", int(time.time()))
    days  = (now-last)/86400 if last else 9999
    wsent = history.get("walletTotalSent",0)
    wrecv = history.get("walletTotalReceived",0)
    sc    = max(history.get("walletSentCount",1),1)
    rc    = max(history.get("walletReceivedCount",1),1)
    return {
        "avg_min_between_sent":    history.get("avgMinSent",1440),
        "avg_min_between_received":history.get("avgMinRecv",1440),
        "time_diff_first_last":    history.get("timeDiff",43200),
        "sent_txns":               sc,
        "received_txns":           rc,
        "contracts_created":       history.get("contractsCreated",0),
        "unique_received_from":    max(history.get("uniqueSenders",1),1),
        "unique_sent_to":          max(history.get("uniqueRecipients",1),1),
        "min_val_received":        history.get("minRecv",0),
        "max_val_received":        history.get("maxRecv",sale),
        "avg_val_received":        wrecv/rc,
        "min_val_sent":            history.get("minSent",0),
        "max_val_sent":            history.get("maxSent",sale),
        "avg_val_sent":            wsent/sc,
        "total_eth_sent":          wsent,
        "total_eth_received":      wrecv,
        "total_eth_balance":       max(wrecv-wsent,0),
        "price_to_assessed_ratio": sale/max(av,1),
        "days_since_last_sale":    days,
        "transfers_12mo":          history.get("transfers12mo",0),
        "num_liens":               history.get("numLiens",0),
        "doc_frequency":           history.get("docsWeek",0),
        "price_vs_avg":            (sale-avg_n)/max(avg_n,1),
        "assessed_vs_avg":         (av-avg_n)/max(avg_n,1),
        "unique_agents":           history.get("uniqueAgents",1),
        "mortgage_to_value":       history.get("mortgageRatio",0.75),
    }

def full_score(tx: dict, history: dict) -> dict:
    f = extract_features(tx, history)
    is_registration = tx.get("tx_type", "").lower() == "registration"
    rp, reason, fraud_type, flags = rule_score(f, is_registration)
    ml = _ml.score(f)

    if ml:
        # HYBRID SCORING: 70% Machine Learning + 30% Domain Rules
        ml_avg = sum(ml.values()) / len(ml)
        final_p = (0.7 * ml_avg) + (0.3 * rp)
        
        # SAFETY VETO: If rules detect CRITICAL real-estate fraud, force the score up
        if rp > 0.8:
            final_p = max(final_p, rp)
            
        model = "Hybrid Ensemble (" + "+".join(ml.keys()) + ")"
    else:
        # Fallback only if models aren't trained yet
        final_p = rp
        model   = "RuleEngine (Fallback)"

    final_p  = min(max(final_p,0),1)
    is_fraud = final_p >= 0.5
    risk_n   = 3 if final_p>.80 else 2 if final_p>.60 else 1 if final_p>.40 else 0
    risk_s   = ["Low","Medium","High","Critical"][risk_n]

    return {
        "is_fraud":      is_fraud,
        "confidence":    round(final_p,4),
        "confidence_pct":int(final_p*100),
        "risk_level":    risk_s,
        "risk_num":      risk_n,
        "fraud_type":    fraud_type if is_fraud else "None",
        "reason":        reason,
        "flags":         flags,
        "model_used":    model,
        "ml_scores":     {k:round(v,4) for k,v in ml.items()},
        "rule_score":    round(rp,4),
        "feature_hash":  hashlib.sha256(json.dumps(f,sort_keys=True).encode()).hexdigest()[:24],
        "features":      f,
    }
