# Traceability

RadioNoise provides three complementary traceability mechanisms to prove when, how, and from what data a password was generated.

## Overview

```mermaid
flowchart TD
    Gen["Password Generation"] --> P["🔏 Proof of Generation\nCryptographic hash chain\nLinks capture → password"]
    Gen --> A["📋 Forensic Audit Trail\nSQLite blockchain\nTimeline of all generations"]
    Gen --> B["💾 Encrypted Backup\nAES-256-GCM\nRecover password from IQ data"]

    P --> Verify["Verify proof\n--verify proof.json"]
    A --> Chain["Verify chain\n--verify-chain audit.db"]
    B --> Recover["Recover password\n--recover backup_id"]

    style P fill:#2d5986,color:#fff
    style A fill:#059669,color:#fff
    style B fill:#8b5cf6,color:#fff
```

| Feature | Purpose | Storage | Requires RTL-SDR |
|---------|---------|---------|------------------|
| Proof of Generation | Prove a password came from specific radio noise | JSON file | Yes |
| Audit Trail | Log all generations with blockchain chaining | SQLite database | No |
| Encrypted Backup | Recover a password from its original IQ data | Encrypted files | Yes |

## Proof of Generation

A cryptographic proof links each step of the pipeline with SHA-256 hashes, creating an unforgeable chain from radio capture to password.

### Hash Chain Construction

```mermaid
sequenceDiagram
    participant IQ as Raw IQ Samples
    participant VN as Von Neumann Output
    participant WH as SHA-512 Whitened
    participant PW as Password

    Note over IQ: capture_hash = SHA-256(iq_bytes)
    IQ->>VN: von_neumann_extract()
    Note over VN: entropy_hash = SHA-256(extracted_bytes)
    VN->>WH: hash_entropy()
    Note over WH: processed_hash = SHA-256(whitened_bytes)
    WH->>PW: generate_password()
    Note over PW: password_hash = SHA-256(password)

    Note over IQ,PW: signature = SHA-256(timestamp + capture_hash + entropy_hash + password_hash)
```

### Proof JSON Structure

```json
{
  "timestamp": "2026-02-04T20:18:23.456789Z",
  "capture_hash": "a1b2c3d4...",
  "capture_size": 500000,
  "entropy_hash": "e5f6a7b8...",
  "processed_hash": "c9d0e1f2...",
  "password_hash": "3a4b5c6d...",
  "password_length": 24,
  "charset": "safe",
  "signature": "7e8f9a0b...",
  "metadata": {
    "frequency": 100000000,
    "sample_rate": 2400000,
    "samples": 500000
  }
}
```

The password itself is **never stored** — only its SHA-256 hash. Verification requires the original password.

### API

```python
from radionoise.traceability.proof import ProofOfGeneration

pog = ProofOfGeneration()

# Step 1: Capture with proof
raw_data, proof = pog.capture_with_proof(
    samples=500000,
    frequency=100e6
)

# Step 2: Process with proof
processed, proof = pog.process_with_proof(raw_data, proof)

# Step 3: Generate password with proof
password, proof = pog.generate_password_with_proof(processed, proof,
    length=24, charset="safe")

# Save proof
pog.save_proof(proof, "proof.json")

# Verify proof
valid, message = pog.verify_proof(password, proof)
```

### CLI

```bash
# Generate with proof
python RadioNoise.py --proof --proof-output my_proof.json -n 1 -l 24

# Verify (prompts for password)
python RadioNoise.py --verify my_proof.json
```

## Forensic Audit Trail

The audit trail maintains a blockchain-style SQLite database where each entry is cryptographically linked to the previous one.

### Database Schema

```mermaid
erDiagram
    generations {
        INTEGER id PK
        TEXT gen_id "UUID"
        TEXT timestamp "ISO 8601"
        TEXT capture_hash "SHA-256"
        TEXT entropy_hash "SHA-256"
        TEXT password_hash "SHA-256"
        TEXT signature "SHA-256"
        TEXT chain_hash "SHA-256(prev + data)"
        TEXT prev_hash "Previous chain_hash"
        TEXT metadata "JSON"
        TEXT context "JSON"
    }

    events {
        INTEGER id PK
        TEXT event_id "UUID"
        TEXT gen_id FK "References generations"
        TEXT event_type "capture|process|generate|verify"
        TEXT timestamp "ISO 8601"
        TEXT description "Human-readable"
        TEXT data "JSON details"
    }

    generations ||--o{ events : "has"
```

### Blockchain Chaining

```mermaid
flowchart LR
    G1["Generation 1\nhash: abc123"] -->|"prev_hash"| G2["Generation 2\nhash: def456"]
    G2 -->|"prev_hash"| G3["Generation 3\nhash: ghi789"]
    G3 -->|"prev_hash"| G4["Generation 4\nhash: jkl012"]

    subgraph Verification
        V["chain_hash[n] =\nSHA-256(\n  chain_hash[n-1]\n  + gen_id\n  + timestamp\n  + password_hash\n  + signature\n)"]
    end

    style G1 fill:#2d5986,color:#fff
    style G2 fill:#2d5986,color:#fff
    style G3 fill:#2d5986,color:#fff
    style G4 fill:#2d5986,color:#fff
```

If any entry is modified, deleted, or inserted, the chain breaks and `--verify-chain` detects it.

### API

```python
from radionoise.traceability.audit import ForensicAuditTrail

# Create/open audit trail
audit = ForensicAuditTrail("audit.db")

# Add a generation
gen_id, chain_hash = audit.add_generation(proof)

# Log an event
audit.log_event(gen_id, "verify", "Password verified by user")

# Verify chain integrity
valid, errors = audit.verify_chain_integrity()

# Get timeline
timeline = audit.get_timeline(limit=10)

# Get statistics
stats = audit.get_statistics()

# Export report
audit.export_audit_report("report.json")

# Prove generation time (returns proof with chain context)
time_proof = audit.prove_generation_time(gen_id)

audit.close()
```

### CLI

```bash
# Add generation to audit trail
python RadioNoise.py --proof --audit ./audit.db -n 1

# Verify chain integrity
python RadioNoise.py --verify-chain ./audit.db

# Export audit report
python RadioNoise.py --audit-report report.json --audit ./audit.db
```

## Encrypted Backup

Backups store the raw IQ samples encrypted with AES-256-GCM, allowing the original password to be **regenerated** from the physical data. The password itself is never stored.

### Backup/Recovery Process

```mermaid
sequenceDiagram
    participant U as User
    participant B as SecureBackupSystem
    participant K as PBKDF2
    participant E as AES-256-GCM
    participant D as Disk

    Note over U,D: === BACKUP ===
    U->>B: backup_password(password, proof, iq_data, master_pass)
    B->>K: derive_key(master_pass, salt)
    K-->>B: 256-bit key
    B->>E: encrypt(iq_data, key, nonce)
    E-->>B: ciphertext + auth_tag
    B->>D: Save encrypted_iq.bin, proof.json, crypto_metadata.json

    Note over U,D: === RECOVERY ===
    U->>B: recover_password(backup_id, master_pass)
    B->>D: Load encrypted_iq.bin + metadata
    B->>K: derive_key(master_pass, stored_salt)
    K-->>B: 256-bit key
    B->>E: decrypt(ciphertext, key, stored_nonce)
    E-->>B: original iq_data
    Note over B: Reprocess: Von Neumann → SHA-512 → generate_password()
    B-->>U: Regenerated password
```

### Backup Directory Structure

```
backups/{timestamp}/
├── encrypted_iq.bin        # AES-256-GCM encrypted IQ samples
├── proof.json              # Proof of generation (unencrypted)
└── crypto_metadata.json    # Salt, nonce, iterations, charset, length
```

### Crypto Metadata

```json
{
  "salt": "base64-encoded 32 bytes",
  "nonce": "base64-encoded 12 bytes",
  "iterations": 600000,
  "password_length": 24,
  "charset": "safe",
  "capture_size": 500000,
  "timestamp": "2026-02-04T20:18:23Z"
}
```

### API

```python
from radionoise.traceability.backup import SecureBackupSystem

backup = SecureBackupSystem("./backups")

# Create backup
backup_id = backup.backup_password(
    password, proof, raw_iq_data, master_password
)

# List backups
backups = backup.get_backup_list()

# Recover password
password, proof = backup.recover_password(backup_id, master_password)

# Export as portable bundle
backup.export_backup_bundle(backup_id, "bundle.tar.gz")

# Import bundle (with path traversal protection)
backup.import_backup_bundle("bundle.tar.gz")

# Delete backup
backup.delete_backup(backup_id)
```

### CLI

```bash
# Create backup (requires RTL-SDR for raw IQ data)
python RadioNoise.py --proof --backup ./backups --master-password "s3cur3" -n 1

# Using environment variable (recommended)
export RADIONOISE_MASTER_PASS="s3cur3"
python RadioNoise.py --proof --backup ./backups -n 1

# Recover password
python RadioNoise.py --recover "backup_2026-02-04" --backup ./backups

# Interactive master password (no command-line exposure)
python RadioNoise.py --recover "backup_2026-02-04" --backup ./backups
# → prompts for password
```

### Security Notes

- The master password is **not stored** — losing it means losing the backup
- Each backup has a unique salt and nonce
- PBKDF2 uses 600,000 iterations (OWASP 2024 recommendation)
- Bundle import validates all paths against directory traversal and rejects symlinks
- Recovery reprocesses IQ data through the full pipeline (Von Neumann → SHA-512 → generate)
