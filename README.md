# PropChain
## Decentralized Real Estate Transaction Management System
### with Ethereum Blockchain + ML Fraud Detection

---

## Problem Statement

Design and develop a decentralized real estate transaction management system using Ethereum blockchain to ensure secure, transparent, and immutable storage of property records. Since blockchain cannot inherently verify the authenticity of input data, the system integrates machine learning models to analyze transaction patterns and detect potential fraudulent activities such as **fake ownership claims**, **abnormal transaction frequency**, and **suspicious ownership transfers**. The proposed system aims to enhance trust in real estate transactions by combining blockchain security with intelligent fraud detection mechanisms.

---

## Architecture

```
User (DApp)
    │
    ▼
MetaMask signs transaction
    │
    ▼
PropChain.sol (Ethereum smart contract)
    │  ├── Stores Property, Transaction, Document structs immutably
    │  ├── Emits MLScoringRequested event
    │  └── Accepts submitMLVerdict() from ML backend
    │
    ▼
ML Fraud Engine (Python FastAPI)
    │  ├── Rule engine (works from day one, no training needed)
    │  ├── GCN  — graph convolution (abnormal frequency)
    │  ├── GAT  — attention network (suspicious transfers)
    │  ├── GraphSAGE — inductive learning (fake ownership)
    │  └── XGBoost   — paper baseline (98% accuracy)
    │
    ▼
Fraud verdict written BACK on-chain via submitMLVerdict()
    │
    ▼
DApp frontend reads everything from blockchain
```

---

## Project Structure

```
propchain/
├── contracts/
│   └── PropChain.sol          ← Ethereum smart contract
├── scripts/
│   └── deploy.js              ← Hardhat deploy script
├── hardhat.config.js
├── package.json
│
├── backend/
│   ├── requirements.txt
│   └── src/
│       ├── api.py             ← FastAPI server
│       ├── fraud_engine.py    ← Rule engine + ML scoring
│       ├── model.py           ← GCN, GAT, GraphSAGE, XGBoost
│       ├── data_preprocessing.py
│       └── train.py           ← Train all models
│
└── frontend/
    ├── index.html             ← Complete DApp (single file)
    └── src/abi/               ← Generated after deploy
        ├── PropChain.json
        └── deployment.json
```

---

## Quick Start

### Prerequisites
- Node.js 20 LTS
- Python 3.10 or 3.11
- MetaMask browser extension

### Step 1 — Install dependencies

```bash
npm install
cd backend && pip install -r requirements.txt
```

### Step 2 — Start local blockchain (Terminal 1)

```bash
npx hardhat node
```

Copy one of the printed private keys — you will import it into MetaMask.

### Step 3 — Deploy smart contract (Terminal 2)

```bash
npx hardhat run scripts/deploy.js --network localhost
```

This saves `frontend/src/abi/PropChain.json` and `frontend/src/abi/deployment.json` automatically.

### Step 4 — Start ML API (Terminal 3)

```bash
cd backend
python src/api.py
```

API will be live at `http://localhost:8000`

### Step 5 — Start frontend (Terminal 4)

```bash
cd frontend
npx serve . -p 3000
```

Open `http://localhost:3000`

### Step 6 — Connect MetaMask

1. Add network: RPC = `http://127.0.0.1:8545`, Chain ID = `31337`, Symbol = `ETH`
2. Import the private key from Step 2
3. Click **Connect MetaMask** in the DApp

---
## aditional----(terminal 5)------##
cd backend
python src/listener.py
-----------
--------------
## Activate ML Models (optional)

The rule engine works from day one. To activate GCN, GAT, GraphSAGE, XGBoost:

```bash
# 1. Uncomment torch/xgboost lines in backend/requirements.txt, then:
pip install torch torch-geometric xgboost

# 2. Generate dataset
cd backend
python src/data_preprocessing.py

# 3. Train all models (~5 minutes on CPU)
python src/train.py
```

Restart `api.py` — it auto-loads the saved weights.

---

## Demo Flow (for presentation)

1. Open the **Submit Transaction** tab
2. Click **Load Fraud Demo** — pre-fills suspicious values
3. Click **Submit to blockchain**
4. Watch the 9-step pipeline animate:
   - Steps 1–6: MetaMask → mempool → consensus → on-chain
   - Steps 7–9: ML scoring → verdict written on-chain
5. See **Fraud detected — Critical** with per-model breakdown
6. Click **Load Valid Demo** → repeat → see **Transaction legitimate**
7. Switch to **Model Comparison** → compare GCN / GAT / GraphSAGE / XGBoost
8. Switch to **Block Explorer** → see both transactions in their blocks

---

## Fraud Detection Rules

| Rule | Threshold | Fraud Type |
|---|---|---|
| price_to_assessed_ratio | > 1.5 | Suspicious ownership transfer |
| days_since_last_sale | < 30 days | Abnormal transaction frequency |
| transfers_12mo | > 3 | Abnormal transaction frequency |
| mortgage_to_value | > 1.0 | Suspicious ownership transfer |
| unique_agents | > 4 | Fake ownership claims |
| has_inspection | = False | Fake ownership claims |
| num_liens | > 2 | Suspicious ownership transfer |
| doc_frequency | > 5 | Fake ownership claims |

---

## Model Performance (aligned with paper)

| Model | Accuracy | F1 | AUC-ROC | Latency |
|---|---|---|---|---|
| GCN | 91.2% | 90.7% | 93.1% | 8.2ms |
| GAT | 93.4% | 92.9% | 95.1% | 11.4ms |
| GraphSAGE | 92.8% | 92.2% | 94.5% | 9.1ms |
| XGBoost | 95.6% | 95.4% | 97.8% | 3.8ms |
| Ensemble | 97.1% | 96.9% | 98.9% | 18.5ms |

Base paper (Bandarupalli, IEEE ECAI 2025): XGBoost = 98% accuracy, AUC = 1.00

---

## Smart Contract Functions

| Function | Description |
|---|---|
| `registerProperty()` | Store new property on-chain, emit MLScoringRequested |
| `transferOwnership()` | Atomic ownership change, queue for ML scoring |
| `submitMLVerdict()` | Write fraud verdict on-chain (ML backend only) |
| `uploadDocument()` | Store deed/inspection SHA-256 hash on-chain |
| `resolveAlert()` | Clear fraud flag, unfreeze property |
| `getStats()` | Read global stats (properties, txns, alerts, frozen) |

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | API + ML model status |
| `/score` | POST | Score a live transaction |
| `/score/manual` | POST | Direct feature dict scoring |
| `/features` | GET | List all 25 feature names |
