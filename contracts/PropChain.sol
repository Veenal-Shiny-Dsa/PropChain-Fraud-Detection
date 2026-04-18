// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title PropChain
 * @dev Decentralized Real Estate Transaction Management System
 *      Fix: Transaction split into TxCore + TxMLResult structs
 *      to resolve "stack too deep" compilation error.
 */
contract PropChain {

    address public owner;

    enum PropertyStatus  { Active, UnderContract, Sold, Frozen }
    enum TransactionType { Registration, Sale, Transfer, Mortgage }
    enum RiskLevel       { Low, Medium, High, Critical }
    enum DocType         { Deed, Mortgage, Inspection, Appraisal }

    struct Property {
        bytes32        id;
        string         location;
        string         propType;
        uint256        squareFeet;
        uint256        assessedValue;
        address        currentOwner;
        uint256        registeredAt;
        PropertyStatus status;
        uint256        transferCount;
        bool           fraudFlagged;
        bool           exists;
    }

    struct TxCore {
        bytes32         txId;
        bytes32         propertyId;
        address         from;
        address         to;
        uint256         value;
        TransactionType txType;
        uint256         timestamp;
        bytes32         docHash;
    }

    struct TxMLResult {
        bool      mlScored;
        bool      isFraud;
        RiskLevel riskLevel;
        uint8     confidence;
        string    modelUsed;
        string    fraudType;
        string    reason;
    }

    struct Document {
        bytes32  docHash;
        DocType  docType;
        string   ipfsCid;
        address  uploadedBy;
        uint256  uploadedAt;
        bool     verified;
    }

    struct FraudAlert {
        bytes32   alertId;
        bytes32   propertyId;
        bytes32   txId;
        address   suspect;
        RiskLevel risk;
        uint8     confidence;
        string    model;
        string    fraudType;
        string    reason;
        uint256   timestamp;
        bool      resolved;
    }

    struct WalletProfile {
        uint256 totalTxns;
        uint256 totalSent;
        uint256 totalReceived;
        uint256 propertiesOwned;
        uint256 propertiesSold;
        uint256 fraudAlerts;
        uint256 lastActivity;
        bool    blacklisted;
    }

    mapping(bytes32 => Property)       public properties;
    mapping(bytes32 => TxCore)         public txCores;
    mapping(bytes32 => TxMLResult)     public txResults;
    mapping(bytes32 => Document[])     public documents;
    mapping(bytes32 => FraudAlert)     public alerts;
    mapping(address => WalletProfile)  public wallets;
    mapping(address => bytes32[])      public ownerProps;
    mapping(bytes32 => bytes32[])      public propTxns;
    mapping(bytes32 => bytes32[])      public propAlerts;
    mapping(address => bool)           public mlNodes;

    bytes32[] public allProps;
    bytes32[] public allTxns;
    bytes32[] public allAlerts;
    bytes32[] public pendingML;

    uint256 public totalProperties;
    uint256 public totalTransactions;
    uint256 public totalFraudAlerts;
    uint256 public totalFrozen;

    event PropertyRegistered(bytes32 indexed pid, address indexed owner, string location, uint256 value, uint256 timestamp);
    event TransactionSubmitted(bytes32 indexed txId, bytes32 indexed pid, address from, address to, uint256 value, uint8 txType);
    event MLScoringRequested(bytes32 indexed txId, bytes32 indexed pid, uint256 value, address from, address to);
    event FraudAlertCreated(bytes32 indexed alertId, bytes32 indexed pid, address suspect, uint8 risk, uint8 confidence, string fraudType);
    event PropertyFrozen(bytes32 indexed pid, string reason);
    event PropertyUnfrozen(bytes32 indexed pid);
    event WalletBlacklisted(address indexed wallet);
    event MLNodeAuthorized(address indexed node);

    modifier onlyOwner()             { require(msg.sender == owner, "Not owner"); _; }
    modifier onlyML()                { require(mlNodes[msg.sender] || msg.sender == owner, "Not ML node"); _; }
    modifier propExists(bytes32 pid) { require(properties[pid].exists, "Property not found"); _; }
    modifier notFrozen(bytes32 pid)  { require(properties[pid].status != PropertyStatus.Frozen, "Frozen"); _; }
    modifier notBlacklisted()        { require(!wallets[msg.sender].blacklisted, "Blacklisted"); _; }

    constructor() { owner = msg.sender; mlNodes[msg.sender] = true; }

    function registerProperty(
        string  calldata _location,
        string  calldata _propType,
        uint256 _sqft,
        uint256 _assessedValue,
        bytes32 _deedHash,
        string  calldata _ipfsCid
    ) external notBlacklisted returns (bytes32 pid, bytes32 txId) {
        require(_assessedValue > 0, "Value required");
        require(bytes(_location).length > 0, "Location required");

        pid  = keccak256(abi.encodePacked(_location, msg.sender, block.timestamp, block.number));
        txId = keccak256(abi.encodePacked(pid, msg.sender, block.timestamp, "REG"));
        require(!properties[pid].exists, "Already registered");

        properties[pid] = Property({
            id: pid, location: _location, propType: _propType,
            squareFeet: _sqft, assessedValue: _assessedValue,
            currentOwner: msg.sender, registeredAt: block.timestamp,
            status: PropertyStatus.Active, transferCount: 0,
            fraudFlagged: false, exists: true
        });

        txCores[txId] = TxCore({
            txId: txId, propertyId: pid, from: address(0), to: msg.sender,
            value: _assessedValue, txType: TransactionType.Registration,
            timestamp: block.timestamp, docHash: _deedHash
        });

        txResults[txId] = TxMLResult({
            mlScored: false, isFraud: false, riskLevel: RiskLevel.Low,
            confidence: 0, modelUsed: "", fraudType: "", reason: ""
        });

        if (_deedHash != bytes32(0)) {
            documents[pid].push(Document({
                docHash: _deedHash, docType: DocType.Deed, ipfsCid: _ipfsCid,
                uploadedBy: msg.sender, uploadedAt: block.timestamp, verified: false
            }));
        }

        ownerProps[msg.sender].push(pid);
        propTxns[pid].push(txId);
        allProps.push(pid);
        allTxns.push(txId);
        pendingML.push(txId);

        wallets[msg.sender].totalTxns++;
        wallets[msg.sender].propertiesOwned++;
        wallets[msg.sender].lastActivity = block.timestamp;
        totalProperties++;
        totalTransactions++;

        emit PropertyRegistered(pid, msg.sender, _location, _assessedValue, block.timestamp);
        emit TransactionSubmitted(txId, pid, address(0), msg.sender, _assessedValue, 0);
        emit MLScoringRequested(txId, pid, _assessedValue, address(0), msg.sender);
    }

    function transferOwnership(
        bytes32 _pid,
        address _to,
        uint256 _salePrice,
        bytes32 _docHash,
        string  calldata _ipfsCid,
        uint8   _txType
    ) external propExists(_pid) notFrozen(_pid) notBlacklisted returns (bytes32 txId) {
        Property storage prop = properties[_pid];
        require(prop.currentOwner == msg.sender || mlNodes[msg.sender], "Not owner");
        require(_to != address(0) && _to != msg.sender, "Invalid recipient");

        txId = keccak256(abi.encodePacked(_pid, msg.sender, _to, block.timestamp));

        txCores[txId] = TxCore({
            txId: txId, propertyId: _pid, from: msg.sender, to: _to,
            value: _salePrice, txType: TransactionType(_txType),
            timestamp: block.timestamp, docHash: _docHash
        });
        txResults[txId] = TxMLResult({
            mlScored: false, isFraud: false, riskLevel: RiskLevel.Low,
            confidence: 0, modelUsed: "", fraudType: "", reason: ""
        });

        prop.currentOwner  = _to;
        prop.transferCount += 1;

        if (_docHash != bytes32(0)) {
            documents[_pid].push(Document({
                docHash: _docHash, docType: DocType.Deed, ipfsCid: _ipfsCid,
                uploadedBy: msg.sender, uploadedAt: block.timestamp, verified: false
            }));
        }

        ownerProps[_to].push(_pid);
        propTxns[_pid].push(txId);
        allTxns.push(txId);
        pendingML.push(txId);

        wallets[msg.sender].totalTxns++;
        wallets[msg.sender].propertiesSold++;
        wallets[msg.sender].totalSent += _salePrice;
        wallets[_to].totalTxns++;
        wallets[_to].propertiesOwned++;
        wallets[_to].totalReceived += _salePrice;
        wallets[msg.sender].lastActivity = block.timestamp;
        wallets[_to].lastActivity = block.timestamp;
        totalTransactions++;

        emit TransactionSubmitted(txId, _pid, msg.sender, _to, _salePrice, _txType);
        emit MLScoringRequested(txId, _pid, _salePrice, msg.sender, _to);
    }

    function submitMLVerdict(
        bytes32 _txId,
        bool    _isFraud,
        uint8   _risk,
        uint8   _confidence,
        string  calldata _model,
        string  calldata _fraudType,
        string  calldata _reason
    ) external onlyML {
        require(txCores[_txId].txId != bytes32(0), "Tx not found");
        require(!txResults[_txId].mlScored, "Already scored");

        txResults[_txId].mlScored   = true;
        txResults[_txId].isFraud    = _isFraud;
        txResults[_txId].riskLevel  = RiskLevel(_risk);
        txResults[_txId].confidence = _confidence;
        txResults[_txId].modelUsed  = _model;
        txResults[_txId].fraudType  = _fraudType;
        txResults[_txId].reason     = _reason;
        _removePending(_txId);

        if (_isFraud) {
            bytes32 alertId = keccak256(abi.encodePacked(_txId, block.timestamp));
            TxCore storage core = txCores[_txId];
            address suspect = core.from != address(0) ? core.from : core.to;

            alerts[alertId] = FraudAlert({
                alertId: alertId, propertyId: core.propertyId, txId: _txId,
                suspect: suspect, risk: RiskLevel(_risk), confidence: _confidence,
                model: _model, fraudType: _fraudType, reason: _reason,
                timestamp: block.timestamp, resolved: false
            });

            propAlerts[core.propertyId].push(alertId);
            allAlerts.push(alertId);
            totalFraudAlerts++;
            properties[core.propertyId].fraudFlagged = true;
            wallets[suspect].fraudAlerts++;

            if (RiskLevel(_risk) == RiskLevel.Critical) {
                properties[core.propertyId].status = PropertyStatus.Frozen;
                totalFrozen++;
                emit PropertyFrozen(core.propertyId, _reason);
            }
            if (wallets[suspect].fraudAlerts >= 3) {
                wallets[suspect].blacklisted = true;
                emit WalletBlacklisted(suspect);
            }
            emit FraudAlertCreated(alertId, core.propertyId, suspect, _risk, _confidence, _fraudType);
        }
    }

    function uploadDocument(bytes32 _pid, bytes32 _docHash, uint8 _docType, string calldata _ipfsCid)
        external propExists(_pid) {
        documents[_pid].push(Document({
            docHash: _docHash, docType: DocType(_docType), ipfsCid: _ipfsCid,
            uploadedBy: msg.sender, uploadedAt: block.timestamp, verified: false
        }));
    }

    function resolveAlert(bytes32 _alertId) external onlyML {
        alerts[_alertId].resolved = true;
        bytes32 pid = alerts[_alertId].propertyId;
        bool hasOpen = false;
        for (uint i = 0; i < propAlerts[pid].length; i++) {
            if (!alerts[propAlerts[pid][i]].resolved) { hasOpen = true; break; }
        }
        if (!hasOpen) {
            properties[pid].status       = PropertyStatus.Active;
            properties[pid].fraudFlagged = false;
            if (totalFrozen > 0) totalFrozen--;
            emit PropertyUnfrozen(pid);
        }
    }

    function authorizeMlNode(address _node) external onlyOwner {
        mlNodes[_node] = true;
        emit MLNodeAuthorized(_node);
    }

    function getProperty(bytes32 _id)  external view returns (Property memory)    { return properties[_id]; }
    function getTxCore(bytes32 _id)    external view returns (TxCore memory)      { return txCores[_id]; }
    function getTxResult(bytes32 _id)  external view returns (TxMLResult memory)  { return txResults[_id]; }
    function getDocuments(bytes32 _pid)external view returns (Document[] memory)  { return documents[_pid]; }
    function getAlert(bytes32 _id)     external view returns (FraudAlert memory)  { return alerts[_id]; }
    function getWallet(address _w)     external view returns (WalletProfile memory){ return wallets[_w]; }
    function getPropTxns(bytes32 _pid) external view returns (bytes32[] memory)   { return propTxns[_pid]; }
    function getPropAlerts(bytes32 _pid)external view returns (bytes32[] memory)  { return propAlerts[_pid]; }
    function getOwnerProps(address _w) external view returns (bytes32[] memory)   { return ownerProps[_w]; }
    function getAllProps()              external view returns (bytes32[] memory)   { return allProps; }
    function getAllTxns()               external view returns (bytes32[] memory)   { return allTxns; }
    function getAllAlerts()             external view returns (bytes32[] memory)   { return allAlerts; }
    function getPending()              external view returns (bytes32[] memory)   { return pendingML; }

    function getStats() external view returns (
        uint256 props, uint256 txns, uint256 alertCount, uint256 frozen, uint256 pending
    ) {
        return (totalProperties, totalTransactions, totalFraudAlerts, totalFrozen, pendingML.length);
    }

    function _removePending(bytes32 _txId) internal {
        for (uint i = 0; i < pendingML.length; i++) {
            if (pendingML[i] == _txId) {
                pendingML[i] = pendingML[pendingML.length - 1];
                pendingML.pop();
                break;
            }
        }
    }
}
