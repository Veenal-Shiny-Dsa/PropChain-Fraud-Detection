"""
model.py
GCN, GAT, GraphSAGE, XGBoost models for Ethereum fraud detection.
Aligns with Bandarupalli (IEEE ECAI 2025) — XGBoost baseline + GNN extension.
"""
import torch
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False


# ── Graph Neural Networks ─────────────────────────────────────────

class GCNDetector(torch.nn.Module):
    """Graph Convolutional Network — best for abnormal tx frequency patterns."""
    def __init__(self, num_features, hidden=64, num_classes=2, dropout=0.5):
        super().__init__()
        if not GNN_AVAILABLE:
            raise ImportError("torch-geometric required: pip install torch-geometric")
        self.conv1 = GCNConv(num_features, hidden)
        self.conv2 = GCNConv(hidden, hidden // 2)
        self.bn1   = torch.nn.BatchNorm1d(hidden)
        self.bn2   = torch.nn.BatchNorm1d(hidden // 2)
        self.clf   = torch.nn.Linear(hidden // 2, num_classes)
        self.drop  = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        return F.log_softmax(self.clf(x), dim=1)


class GATDetector(torch.nn.Module):
    """Graph Attention Network — best for suspicious ownership transfers."""
    def __init__(self, num_features, hidden=32, num_classes=2, heads=4, dropout=0.5):
        super().__init__()
        if not GNN_AVAILABLE:
            raise ImportError("torch-geometric required: pip install torch-geometric")
        self.conv1 = GATConv(num_features, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden * heads, hidden, heads=1, concat=False, dropout=dropout)
        self.bn    = torch.nn.BatchNorm1d(hidden * heads)
        self.clf   = torch.nn.Linear(hidden, num_classes)
        self.drop  = dropout

    def forward(self, x, edge_index):
        x = F.elu(self.bn(self.conv1(
            F.dropout(x, p=self.drop, training=self.training), edge_index)))
        x = F.elu(self.conv2(
            F.dropout(x, p=self.drop, training=self.training), edge_index))
        return F.log_softmax(self.clf(x), dim=1)


class SAGEDetector(torch.nn.Module):
    """GraphSAGE — best for fake ownership claim detection (inductive)."""
    def __init__(self, num_features, hidden=64, num_classes=2, dropout=0.5):
        super().__init__()
        if not GNN_AVAILABLE:
            raise ImportError("torch-geometric required: pip install torch-geometric")
        self.conv1 = SAGEConv(num_features, hidden)
        self.conv2 = SAGEConv(hidden, hidden // 2)
        self.bn    = torch.nn.BatchNorm1d(hidden)
        self.clf   = torch.nn.Linear(hidden // 2, num_classes)
        self.drop  = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.bn(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return F.log_softmax(self.clf(x), dim=1)


# ── XGBoost — paper baseline (98% accuracy, AUC=1.00) ─────────────

class XGBoostDetector:
    """
    XGBoost classifier. Paper best model.
    Params match Bandarupalli 2025: n_estimators=250, max_depth=6, lr=0.1.
    """
    def __init__(self):
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=250,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.5,
                colsample_bytree=0.7,
                scale_pos_weight=5,
                eval_metric='logloss',
                random_state=42
            )
        except ImportError:
            raise ImportError("xgboost required: pip install xgboost")

    def fit(self, X, y, Xv=None, yv=None):
        eval_set = [(Xv, yv)] if Xv is not None else None
        self.model.fit(X, y, eval_set=eval_set, verbose=False)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        self.model.load_model(path)
        return self


# ── Registry ──────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "GCN":       GCNDetector,
    "GAT":       GATDetector,
    "GraphSAGE": SAGEDetector,
}

def create_model(name: str, num_features: int, hidden: int = 64):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_features=num_features, hidden=hidden)
