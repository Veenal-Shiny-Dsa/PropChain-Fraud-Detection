"""
data_preprocessing.py
Generates a synthetic real estate transaction dataset that mirrors the
Kaggle Ethereum fraud dataset structure (FLAG column, 25 features).
Run once before train.py:
    python src/data_preprocessing.py
"""
import torch
import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
warnings.filterwarnings("ignore")

_SRC   = os.path.dirname(os.path.abspath(__file__))
_BACK  = os.path.dirname(_SRC)
_DATA  = os.path.join(_BACK, "data")
_MDLS  = os.path.join(_BACK, "models")
os.makedirs(_DATA, exist_ok=True)
os.makedirs(_MDLS, exist_ok=True)

# ── IMPORTANT: Must be at MODULE level, NOT inside preprocess() ───────────────
# Defining a class inside a function makes it a "local class" that Python's
# pickle module cannot serialize. torch.save uses pickle internally, so
# saving an object of that local class raises:
#   AttributeError: Can't pickle local function 'preprocess.<locals>.Data'
# Moving it here (module scope) gives it a stable importable path and fixes it.
class SimpleData:
    """Fallback graph data container used when torch_geometric is absent.
    Implements .to() and .clone() so train.py works without torch_geometric.
    """
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def to(self, device):
        """Move all tensor attributes to the given device."""
        for key, val in self.__dict__.items():
            if hasattr(val, 'to'):
                setattr(self, key, val.to(device))
        return self

    def clone(self):
        import copy
        return copy.deepcopy(self)


FEATURE_COLS = [
    "avg_min_between_sent",
    "avg_min_between_received",
    "time_diff_first_last",
    "sent_txns",
    "received_txns",
    "contracts_created",
    "unique_received_from",
    "unique_sent_to",
    "min_val_received", "max_val_received", "avg_val_received",
    "min_val_sent",     "max_val_sent",     "avg_val_sent",
    "total_eth_sent",
    "total_eth_received",
    "total_eth_balance",
]
N_FEATURES = len(FEATURE_COLS)


def generate_dataset(n: int = 2000, fraud_rate: float = 0.18, seed: int = 42):
    csv_path = os.path.join(_DATA, "transaction_dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")
    
    print(f"[preprocess] Loading Real Kaggle Dataset from {csv_path}")
    raw_df = pd.read_csv(csv_path)
    
    col_mapping = {
        "Avg min between sent tnx": "avg_min_between_sent",
        "Avg min between received tnx": "avg_min_between_received",
        "Time Diff between first and last (Mins)": "time_diff_first_last",
        "Sent tnx": "sent_txns",
        "Received Tnx": "received_txns",
        "Number of Created Contracts": "contracts_created",
        "Unique Received From Addresses": "unique_received_from",
        "Unique Sent To Addresses": "unique_sent_to",
        "min value received": "min_val_received",
        "max value received ": "max_val_received",
        "avg val received": "avg_val_received",
        "min val sent": "min_val_sent",
        "max val sent": "max_val_sent",
        "avg val sent": "avg_val_sent",
        "total Ether sent": "total_eth_sent",   
        "total ether received": "total_eth_received",
        "total ether balance": "total_eth_balance"
    }
    
    df = pd.DataFrame()
    for raw_col, clean_col in col_mapping.items():
        df[clean_col] = raw_df[raw_col].fillna(0)
    df["FLAG"] = raw_df["FLAG"].fillna(0).astype(np.int64)
        
    return df


def build_knn_graph(X, k=5):
    k = min(k, X.shape[0] - 1)
    A = kneighbors_graph(X, n_neighbors=k, mode="connectivity", include_self=False)
    A = (A + A.T > 0).astype(float)
    cx = A.tocoo()
    return torch.tensor(np.vstack([cx.row, cx.col]), dtype=torch.long)


def preprocess(n=2000, k=5):
    print("[preprocess] Processing Kaggle Ethereum dataset (17 features)...")
    df = generate_dataset(n=n)
    csv_path = os.path.join(_DATA, "processed_kaggle_features.csv")
    df.to_csv(csv_path, index=False)
    print(f"[preprocess] CSV saved -> {csv_path}")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["FLAG"].values.astype(np.int64)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    edge_index = build_knn_graph(X_s, k=k)
    x_t = torch.tensor(X_s, dtype=torch.float)
    y_t = torch.tensor(y,   dtype=torch.long)

    # SimpleData is defined at module level above — pickle can handle it
    try:
        from torch_geometric.data import Data
        data = Data(x=x_t, edge_index=edge_index, y=y_t)
    except ImportError:
        data = SimpleData(x=x_t, edge_index=edge_index, y=y_t)

    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, train_size=0.70, stratify=y, random_state=42)
    v,  te  = train_test_split(tmp, train_size=0.50, stratify=y[tmp], random_state=42)

    def mask(ids):
        m = torch.zeros(len(y), dtype=torch.bool)
        m[ids] = True
        return m

    data.train_mask    = mask(tr)
    data.val_mask      = mask(v)
    data.test_mask     = mask(te)
    data.num_features  = len(FEATURE_COLS)
    data.num_classes   = 2
    data.feature_names = FEATURE_COLS

    out = os.path.join(_DATA, "processed_data.pt")
    torch.save({"data": data, "scaler": scaler}, out)

    u, c = np.unique(y, return_counts=True)
    print(f"[preprocess] {len(y)} samples — legit: {c[0]}, fraud: {c[1]}")
    print(f"[preprocess] Graph: {len(y)} nodes, {edge_index.shape[1]} edges")
    print(f"[preprocess] Saved -> {out}")
    return data, scaler


if __name__ == "__main__":
    preprocess()
