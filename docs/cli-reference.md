# CLI Reference

RadioNoise provides two CLI entry points:

| Entry Point | Description |
|-------------|-------------|
| `python RadioNoise.py` | Direct script execution |
| `radionoise` | Installed package command (after `pip install -e .`) |

Both use the same code path (`radionoise.cli:main`).

## Options by Category

### Generation

| Option | Default | Description |
|--------|---------|-------------|
| `-l, --length LENGTH` | 16 | Password length (characters) or passphrase word count |
| `-n, --count COUNT` | 5 | Number of passwords to generate |
| `-c, --charset CHARSET` | `safe` | Character set (see table below) |
| `-p, --passphrase` | off | Generate passphrases instead of passwords |
| `-f, --file FILE` | — | Use an existing entropy file instead of capturing |
| `--save-entropy FILE` | — | Save generated entropy to a file |

### Character Sets

| Name | Characters | Size | Bits/char |
|------|-----------|------|-----------|
| `safe` | `a-zA-Z0-9` + `-_!@#$%` | 70 | 6.13 |
| `alnum` | `a-zA-Z0-9` | 62 | 5.95 |
| `alpha` | `a-zA-Z` | 52 | 5.70 |
| `full` | `a-zA-Z0-9` + 25 symbols | 87 | 6.44 |
| `hex` | `0-9a-f` | 16 | 4.00 |
| `digits` | `0-9` | 10 | 3.32 |

### Passphrase

Passphrases use the EFF large wordlist (7776 words, ~12.925 bits/word). Words are separated by hyphens.

| Words | Entropy |
|-------|---------|
| 4 | ~51.7 bits |
| 5 | ~64.6 bits |
| 6 | ~77.5 bits |
| 7 | ~90.5 bits |
| 8 | ~103.4 bits |

### Processing

| Option | Default | Description |
|--------|---------|-------------|
| `--no-hash` | off | Disable SHA-512 whitening (**not recommended**) |
| `--hash-algo {sha256,sha512}` | `sha512` | Hash algorithm for whitening |

### RTL-SDR

| Option | Default | Description |
|--------|---------|-------------|
| `--frequency HZ` | 100000000 (100 MHz) | Capture frequency in Hz |
| `--no-fallback` | off | Fail instead of falling back to RDRAND/CSPRNG |
| `--use-rdseed` | off | Prefer RDSEED over RDRAND for fallback |

### NIST Tests

| Option | Default | Description |
|--------|---------|-------------|
| `--test-only` | off | Run NIST tests only (no password generation) |
| `--no-test` | off | Skip NIST tests (faster) |
| `--full-test` | off | Run all 15 tests instead of 9 fast tests (~30s) |

### Traceability

| Option | Default | Description |
|--------|---------|-------------|
| `--proof` | off | Generate a cryptographic proof of generation |
| `--proof-output FILE` | `proof.json` | Proof output file path |
| `--audit DB` | — | SQLite database for audit trail |
| `--backup DIR` | — | Directory for encrypted IQ backup |
| `--master-password PASS` | — | Master password for backup (or env `RADIONOISE_MASTER_PASS`) |

### Verification

| Option | Description |
|--------|-------------|
| `--verify PROOF.json` | Verify a proof file (prompts for password) |
| `--verify-chain DB` | Verify audit trail chain integrity |
| `--audit-report OUTPUT.json` | Export audit report as JSON |
| `--recover BACKUP_ID` | Recover password from encrypted backup |

### Output

| Option | Description |
|--------|-------------|
| `-q, --quiet` | Suppress all output except passwords |

## Examples

### Basic Usage

```bash
# Generate 5 passwords, 16 characters, safe charset (default)
python RadioNoise.py

# Generate 10 passwords, 24 characters, full charset
python RadioNoise.py -n 10 -l 24 -c full

# Generate 5 passphrases, 6 words each
python RadioNoise.py -p -l 6

# Quiet mode (just passwords, no headers)
python RadioNoise.py -q -n 3
```

### Entropy Source Control

```bash
# Force RTL-SDR only (fail if unavailable)
python RadioNoise.py --no-fallback

# Use RDSEED for CPU fallback
python RadioNoise.py --use-rdseed

# Use an existing entropy file
python RadioNoise.py -f captured_entropy.bin

# Capture and save entropy for later use
python RadioNoise.py --save-entropy my_entropy.bin
```

### NIST Test Modes

```bash
# Fast mode (default): 9 tests, ~1-2s
python RadioNoise.py

# Full suite: 15 tests, ~30s
python RadioNoise.py --full-test

# Test entropy quality without generating passwords
python RadioNoise.py --test-only

# Test an existing entropy file
python RadioNoise.py -f entropy.bin --test-only

# Skip tests for faster generation
python RadioNoise.py --no-test
```

### End-to-End Traceability

```bash
# 1. Generate with cryptographic proof
python RadioNoise.py --proof --proof-output my_proof.json -n 1 -l 24

# 2. Verify the proof (prompts for the password)
python RadioNoise.py --verify my_proof.json

# 3. Add to audit trail
python RadioNoise.py --proof --audit ./audit.db -n 1 -l 24

# 4. Verify chain integrity
python RadioNoise.py --verify-chain ./audit.db

# 5. Export audit report
python RadioNoise.py --audit-report report.json --audit ./audit.db

# 6. Create encrypted backup (requires RTL-SDR)
python RadioNoise.py --proof --backup ./backups --master-password "s3cur3" -n 1

# 7. Recover password from backup
python RadioNoise.py --recover "backup_2026-02-04" --backup ./backups --master-password "s3cur3"
```

### Using Environment Variables

```bash
# Set master password via environment (avoids command-line exposure)
export RADIONOISE_MASTER_PASS="my_secure_password"
python RadioNoise.py --proof --backup ./backups -n 1

# Recover using env variable
python RadioNoise.py --recover "backup_2026-02-04" --backup ./backups
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (insufficient entropy, verification failed, file not found) |
| 130 | User interrupt (Ctrl+C) |

## Package CLI

When installed as a package (`pip install -e .`), the `radionoise` command is available:

```bash
radionoise -n 5 -l 20
radionoise --proof --audit ~/.radionoise/audit.db -n 1
```

This is equivalent to `python RadioNoise.py` with the same options.
