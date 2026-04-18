"""
train.py
Trains GCN, GAT, GraphSAGE, and XGBoost on the preprocessed dataset.
Run once after data_preprocessing.py:
    python src/train.py

Saved weights are loaded automatically by fraud_engine.py and api.py.
"""
import os, sys, json, torch, torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score

_SRC  = os.path.dirname(os.path.abspath(__file__))
_BACK = os.path.dirname(_SRC)
if _SRC not in sys.path: sys.path.insert(0, _SRC)
_DATA = os.path.join(_BACK, "data")
_MDLS = os.path.join(_BACK, "models")
os.makedirs(_MDLS, exist_ok=True)

from model import create_model, XGBoostDetector
from data_preprocessing import SimpleData  # needed so pickle can find it when loading processed_data.pt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[train] Device: {DEVICE}")


def weighted_nll_loss(y_tensor):
    classes, counts = torch.unique(y_tensor, return_counts=True)
    weights = y_tensor.shape[0] / (len(classes) * counts.float())
    return torch.nn.NLLLoss(weight=weights.to(DEVICE))


def train_gnn(name: str, data, epochs=200, lr=0.005, patience=30):
    data = data.to(DEVICE)
    model = create_model(name, data.num_features).to(DEVICE)
    crit  = weighted_nll_loss(data.y[data.train_mask])
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)

    best_loss = float('inf')
    best_auc, patience_cnt = 0.0, 0
    save_path = os.path.join(_MDLS, f"{name}_best.pth")

    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = crit(out[data.train_mask], data.y[data.train_mask])
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            out  = model(data.x, data.edge_index)
            val_loss = crit(out[data.val_mask], data.y[data.val_mask]).item()
            probs = torch.exp(out[data.val_mask])[:,1].cpu().numpy()
            preds = out[data.val_mask].argmax(1).cpu().numpy()
            yv    = data.y[data.val_mask].cpu().numpy()
            f1    = f1_score(yv, preds, zero_division=0)
            try:    auc = roc_auc_score(yv, probs)
            except: auc = 0.5

        sched.step(val_loss)
        if ep % 50 == 0:
            print(f"  [{name}] ep={ep:3d}  loss={loss.item():.4f}  val_loss={val_loss:.4f}  f1={f1:.3f}  auc={auc:.3f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_auc = auc
            patience_cnt = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  [{name}] Early stop at ep={ep}  best_loss={best_loss:.4f}  best_auc={best_auc:.4f}")
                break

    print(f"  [{name}] Done  best_loss={best_loss:.4f}  best_auc={best_auc:.4f}")
    return best_auc


def evaluate_gnn(name: str, data):
    save_path = os.path.join(_MDLS, f"{name}_best.pth")
    if not os.path.exists(save_path):
        return {}
    data = data.to(DEVICE)
    model = create_model(name, data.num_features).to(DEVICE)
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        out   = model(data.x, data.edge_index)
        probs = torch.exp(out[data.test_mask])[:,1].cpu().numpy()
        preds = out[data.test_mask].argmax(1).cpu().numpy()
        yt    = data.y[data.test_mask].cpu().numpy()
    return {
        "accuracy":  round(float(accuracy_score(yt, preds)), 4),
        "precision": round(float(precision_score(yt, preds, zero_division=0)), 4),
        "recall":    round(float(recall_score(yt, preds, zero_division=0)), 4),
        "f1":        round(float(f1_score(yt, preds, zero_division=0)), 4),
        "auc":       round(float(roc_auc_score(yt, probs)), 4),
    }


def train_xgboost(data):
    X  = data.x.numpy()
    y  = data.y.numpy()
    tr = data.train_mask.numpy()
    v  = data.val_mask.numpy()
    te = data.test_mask.numpy()

    xgb = XGBoostDetector()
    xgb.fit(X[tr], y[tr], X[v], y[v])
    xgb.save(os.path.join(_MDLS, "xgboost_best.json"))

    probs = xgb.predict_proba(X[te])[:,1]
    preds = xgb.predict(X[te])
    yt    = y[te]
    metrics = {
        "accuracy":  round(float(accuracy_score(yt, preds)), 4),
        "precision": round(float(precision_score(yt, preds, zero_division=0)), 4),
        "recall":    round(float(recall_score(yt, preds, zero_division=0)), 4),
        "f1":        round(float(f1_score(yt, preds, zero_division=0)), 4),
        "auc":       round(float(roc_auc_score(yt, probs)), 4),
    }
    print(f"  [XGBoost] Done  f1={metrics['f1']:.4f}  auc={metrics['auc']:.4f}")
    return metrics


def main():
    dp = os.path.join(_DATA, "processed_data.pt")
    if not os.path.exists(dp):
        print("[train] processed_data.pt not found — run data_preprocessing.py first")
        sys.exit(1)

    bundle = torch.load(dp, weights_only=False)
    data   = bundle["data"]
    data.num_features = data.x.shape[1]

    results = {}

    for model_name in ["GCN", "GAT", "GraphSAGE"]:
        print(f"\n── Training {model_name} ──")
        try:
            train_gnn(model_name, data)
            results[model_name] = evaluate_gnn(model_name, data)
            print(f"  [{model_name}] Test metrics: {results[model_name]}")
        except ImportError as e:
            print(f"  [{model_name}] Skipped: {e}")

    print("\n── Training XGBoost ──")
    try:
        results["XGBoost"] = train_xgboost(data)
        print(f"  [XGBoost] Test metrics: {results['XGBoost']}")
    except ImportError as e:
        print(f"  [XGBoost] Skipped: {e}")

    out = os.path.join(_MDLS, "training_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[train] Results saved → {out}")
    print("[train] All done:", results)


if __name__ == "__main__":
    main()
