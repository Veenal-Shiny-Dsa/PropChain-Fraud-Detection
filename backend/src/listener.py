"""
listener.py
Real-time blockchain event listener for PropChain.

Watches the smart contract for MLScoringRequested events.
When a new transaction is detected, it:
  1. Reads the transaction data from the blockchain
  2. Extracts 25 fraud features
  3. Scores with rule engine + ML models
  4. Calls submitMLVerdict() to write the verdict back on-chain

Run this in a separate terminal:
    python src/listener.py
"""

import os, sys, json, time, logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

_SRC  = Path(__file__).parent
_BACK = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ── Import fraud engine ───────────────────────────────────────────
from fraud_engine import full_score

# ── Try importing web3 ────────────────────────────────────────────
try:
    from web3 import Web3
    WEB3_OK = True
except ImportError:
    WEB3_OK = False
    log.error("web3 not installed. Run: pip install web3")
    sys.exit(1)


def load_contract(w3: Web3):
    """Load ABI and deployment info saved by deploy.js"""
    dep_path = _BACK / "deployment.json"
    abi_path = _BACK / "PropChain_abi.json"

    if not dep_path.exists():
        log.error("deployment.json not found — run: npx hardhat run scripts/deploy.js --network localhost")
        sys.exit(1)
    if not abi_path.exists():
        log.error("PropChain_abi.json not found — run deploy script first")
        sys.exit(1)

    dep = json.loads(dep_path.read_text())
    abi = json.loads(abi_path.read_text())
    addr = Web3.to_checksum_address(dep["address"])
    contract = w3.eth.contract(address=addr, abi=abi)
    log.info(f"Contract loaded: {addr}")
    return contract


def build_history_from_chain(w3, contract, tx_data: dict) -> dict:
    """
    Reconstruct wallet + property history from the blockchain.
    This replaces any CSV dataset — the chain IS the data source.
    """
    pid    = tx_data.get("propertyId")
    sender = tx_data.get("from_addr", "0x0000000000000000000000000000000000000000")

    history = {
        "assessedValue":        0,
        "lastSaleTs":           0,
        "transfers12mo":        0,
        "walletTotalSent":      0,
        "walletTotalReceived":  0,
        "walletSentCount":      1,
        "walletReceivedCount":  1,
        "avgMinSent":           1440,
        "avgMinRecv":           1440,
        "timeDiff":             43200,
        "contractsCreated":     0,
        "uniqueSenders":        2,
        "uniqueRecipients":     1,
        "minRecv":              0,
        "maxRecv":              tx_data.get("value", 0),
        "minSent":              0,
        "maxSent":              tx_data.get("value", 0),
        "numLiens":             0,
        "docsWeek":             0,
        "areaAvg":              tx_data.get("value", 0),
        "uniqueAgents":         1,
        "mortgageRatio":        0.75,
    }

    try:
        # Get assessed value from property on-chain
        prop = contract.functions.getProperty(pid).call()
        history["assessedValue"] = prop[4] / 1e18   # wei → ETH

        # Get wallet profile
        if sender and sender != "0x" + "0"*40:
            wp = contract.functions.getWallet(
                Web3.to_checksum_address(sender)
            ).call()
            history["walletTotalSent"]     = wp[1] / 1e18
            history["walletTotalReceived"] = wp[2] / 1e18
            history["walletSentCount"]     = max(wp[0], 1)
            history["walletReceivedCount"] = max(wp[0], 1)

        # Count transfers in last 12 months from on-chain history
        tx_ids   = contract.functions.getPropTxns(pid).call()
        now      = int(time.time())
        count_12 = 0
        last_ts  = 0
        for tid in tx_ids:
            t = contract.functions.getTxCore(tid).call()
            t_ts = t[6]  # timestamp field
            if now - t_ts < 365 * 86400:
                count_12 += 1
            if t_ts > last_ts:
                last_ts = t_ts
        history["transfers12mo"] = count_12
        history["lastSaleTs"]    = last_ts

        # Count documents uploaded recently
        docs     = contract.functions.getDocuments(pid).call()
        week_ago = now - 7 * 86400
        history["docsWeek"]  = sum(1 for d in docs if d[4] > week_ago)
        history["areaAvg"]   = history["assessedValue"]

    except Exception as e:
        log.warning(f"Chain read error (using defaults): {e}")

    return history


def process_transaction(w3, contract, account, tx_id: bytes):
    """
    Fetch a transaction from the chain, score it, write verdict back.
    This is called every time MLScoringRequested fires.
    """
    try:
        # 1. Read transaction from chain
        core   = contract.functions.getTxCore(tx_id).call()
        result_data = contract.functions.getTxResult(tx_id).call()

        if result_data[0]:  # mlScored already
            log.info(f"txId {tx_id.hex()[:12]}… already scored, skipping")
            return

        tx_data = {
            "txId":       tx_id,
            "propertyId": core[1],
            "from_addr":  core[2],
            "to_addr":    core[3],
            "value":      core[4] / 1e18,    # wei → ETH
            "txType":     core[5],
            "timestamp":  core[6],
        }

        log.info(f"Scoring txId: 0x{tx_id.hex()[:12]}… value={tx_data['value']:.2f} ETH")

        # 2. Build features from on-chain data
        history  = build_history_from_chain(w3, contract, tx_data)
        result   = full_score(tx_data, history)

        log.info(
            f"  Result: {'FRAUD' if result['is_fraud'] else 'CLEAN'} "
            f"| {result['confidence_pct']}% | {result['risk_level']} "
            f"| {result['fraud_type']} | {result['model_used'].split(' ')[0]}"
        )

        # 3. Write verdict back on-chain
        tx_hash = contract.functions.submitMLVerdict(
            tx_id,
            result["is_fraud"],
            result["risk_num"],
            result["confidence_pct"],
            result["model_used"][:64],
            result["fraud_type"][:64],
            result["reason"][:128],
        ).transact({
            "from": account,
            "gas":  1000000,
        })

        log.info(f"  Verdict written on-chain: 0x{tx_hash.hex()[:12]}…")

        if result["is_fraud"] and result["risk_level"] == "Critical":
            log.warning(f"  Property FROZEN — Critical fraud detected")

    except Exception as e:
        log.error(f"Error processing tx: {e}")


def run_listener(rpc_url="http://127.0.0.1:8545", poll_interval=2):
    """
    Main loop. Two modes:
    1. Event filter (preferred) — catches events in real time
    2. Pending queue poll (fallback) — reads pendingML array from contract
    """
    log.info("=" * 50)
    log.info("PropChain ML Event Listener")
    log.info("=" * 50)

    # Connect to chain
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        log.error(f"Cannot connect to {rpc_url}")
        log.error("Make sure Terminal 1 is running: npx hardhat node")
        sys.exit(1)

    log.info(f"Connected to chain. Block: #{w3.eth.block_number}")

    # Load contract
    contract = load_contract(w3)

    # Use Account #0 as the ML node (same one that deployed the contract)
    account = w3.eth.accounts[0]
    log.info(f"ML node account: {account}")
    log.info(f"Watching for MLScoringRequested events...")
    log.info("-" * 50)

    # Try to create event filter
    try:
        event_filter = contract.events.MLScoringRequested.create_filter(
            from_block="latest"
        )
        use_filter = True
        log.info("Event filter active — real-time mode")
    except Exception:
        use_filter = False
        log.info("Event filter unavailable — using pending queue polling")

    # Main loop
    processed = set()

    while True:
        try:
            if use_filter:
                # Mode 1: Event filter
                for event in event_filter.get_new_entries():
                    tx_id = event["args"]["txId"]
                    if tx_id not in processed:
                        processed.add(tx_id)
                        log.info(f"\n[EVENT] MLScoringRequested — txId: 0x{tx_id.hex()[:12]}…")
                        process_transaction(w3, contract, account, tx_id)
            else:
                # Mode 2: Poll pending queue
                pending = contract.functions.getPending().call()
                for tx_id in pending:
                    if tx_id not in processed:
                        processed.add(tx_id)
                        log.info(f"\n[QUEUE] Pending tx found — txId: 0x{tx_id.hex()[:12]}…")
                        process_transaction(w3, contract, account, tx_id)

        except KeyboardInterrupt:
            log.info("\nListener stopped.")
            break
        except Exception as e:
            log.error(f"Loop error: {e}")

        time.sleep(poll_interval)


if __name__ == "__main__":
    run_listener()
