# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RadioNoise is a True Random Number Generator (TRNG) that uses RTL-SDR (Software Defined Radio) USB dongles (~€30) to capture radio noise and extract cryptographic-quality entropy. The project exploits thermal noise, atmospheric noise, and other electromagnetic fluctuations as an entropy source.

**Important scientific note**: The noise captured is primarily thermal (Johnson-Nyquist), which is classical physics, not quantum. It's practically unpredictable but not fundamentally indeterministic like true quantum RNG.

## Package Structure

```
QuantumNoise/
├── radionoise/                      # Main Python package
│   ├── __init__.py                  # Package exports
│   ├── cli.py                       # CLI entry point with traceability options
│   ├── core/
│   │   ├── entropy.py               # HardwareRNG, von_neumann_extract, capture_entropy
│   │   ├── nist.py                  # NISTTests (15 NIST SP 800-22 tests)
│   │   ├── generator.py             # generate_password, generate_passphrase, CHARSETS
│   │   └── security.py              # secure_zero
│   ├── traceability/
│   │   ├── proof.py                 # ProofOfGeneration (cryptographic proofs)
│   │   ├── audit.py                 # ForensicAuditTrail (SQLite blockchain)
│   │   └── backup.py                # SecureBackupSystem (AES-256-GCM)
│   └── gui/                         # PyQt6 graphical interface
│       ├── main_window.py           # Main window with 4 tabs
│       ├── styles/
│       │   └── dark_theme.py        # Dark theme (security-focused)
│       ├── widgets/
│       │   ├── entropy.py           # RTL-SDR/RDRAND capture widget
│       │   ├── nist.py              # NIST tests with results table
│       │   ├── generator.py         # Password/passphrase generation
│       │   ├── traceability.py      # Proofs/audit/backup management
│       │   └── secure_widgets.py    # SecurePasswordDisplay, SecurePasswordList
│       └── workers/
│           ├── capture_worker.py    # QThread for async capture
│           ├── nist_worker.py       # QThread for NIST tests
│           └── generator_worker.py  # QThread for generation
├── RadioNoise.py                    # Compatibility shim (imports from radionoise)
├── radionoise_gui.py                # GUI launcher
├── tests/
│   ├── conftest.py                  # Pytest fixtures
│   ├── unit/                        # Unit tests
│   └── integration/                 # Integration tests
```

## Commands

### Main CLI (RadioNoise.py)
```bash
# Standard generation (fast mode, 9 NIST tests)
python RadioNoise.py -n 5 -l 20

# Full NIST test suite (15 tests, slower)
python RadioNoise.py --full-test -n 5

# Quick generation (no tests)
python RadioNoise.py --no-test -n 5

# Passphrases
python RadioNoise.py -p -l 6 -n 5

# Test entropy only
python RadioNoise.py --test-only
```

### Traceability Features
```bash
# Generate with cryptographic proof
python RadioNoise.py --proof -n 1 -l 24
# Creates proof.json with hash chain

# Verify a proof
python RadioNoise.py --verify proof.json

# Add to audit trail
python RadioNoise.py --proof --audit ./audit.db -n 1

# Verify chain integrity
python RadioNoise.py --verify-chain ./audit.db

# Create encrypted backup
python RadioNoise.py --proof --backup ./backups --master-password "secret" -n 1

# Recover from backup
python RadioNoise.py --recover "backup_2026-02-04" --master-password "secret"

# Export audit report
python RadioNoise.py --audit-report report.json
```

### Package Usage
```python
# Import from package
from radionoise import capture_entropy, generate_password, NISTTests
from radionoise.traceability import ProofOfGeneration, ForensicAuditTrail

# Generate password
entropy = capture_entropy(samples=500000)
password = generate_password(entropy, length=20, charset='safe')

# Run NIST tests
results = NISTTests.run_all_tests(entropy, verbose=True, fast_mode=True)

# Create proof
pog = ProofOfGeneration()
raw_data, proof = pog.capture_with_proof()
processed, proof = pog.process_with_proof(raw_data, proof)
password, proof = pog.generate_password_with_proof(processed, proof)
```

### Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=radionoise --cov-report=html

# Run specific test module
pytest tests/unit/test_nist.py -v
```

### GUI (PyQt6)
```bash
# Launch graphical interface
python radionoise_gui.py
```

**4 Tabs workflow:**
1. **Entropie** - Configure RTL-SDR (frequency, samples), fallback options, capture
2. **Tests NIST** - Run fast (9 tests) or full (15 tests) suite, view results table
3. **Génération** - Generate passwords/passphrases, copy to clipboard
4. **Traçabilité** - Create/verify proofs, audit trail, encrypted backups

**GUI Architecture:**
- **Dark theme**: Security-focused design (`styles/dark_theme.py`)
- **Async workers**: All blocking operations run in QThreads (no UI freeze)
- **Secure widgets**: `SecurePasswordDisplay` with auto-hide, clipboard auto-clear
- **Memory cleanup**: `secure_clear_widget()` overwrites sensitive data before clearing

**GUI Usage (programmatic):**
```python
from radionoise.gui import MainWindow
from radionoise.gui.styles import apply_dark_theme
from radionoise.gui.widgets import SecurePasswordDisplay, SecurePasswordList
from radionoise.gui.workers import CaptureWorker, NistWorker, GeneratorWorker
```

## Architecture

### Entropy Sources (Priority Order)
1. **RTL-SDR** - Radio noise capture (thermal + atmospheric)
2. **Intel/AMD RDRAND** - CPU hardware RNG (DRBG seeded by hardware entropy)
3. **Intel/AMD RDSEED** - Direct CPU entropy source (slower, use with `--use-rdseed`)
4. **CSPRNG système** - `secrets.token_bytes()` (fallback ultime)

### Entropy Extraction Pipeline (RTL-SDR)
1. **Capture**: `rtl_sdr` captures raw I/Q samples at 2.4 MS/s
2. **Von Neumann**: Removes bias (~3% efficiency, 8 bits/byte entropy)
3. **SHA-512 Whitening**: Ensures uniform distribution
4. **NIST Validation**: 9-15 statistical tests verify quality

### Traceability Pipeline
```
capture_entropy() → raw_data + capture_hash
       ↓
von_neumann_extract() → extracted + entropy_hash
       ↓
hash_entropy() → processed + processed_hash
       ↓
generate_password() → password + password_hash
       ↓
signature = SHA256(timestamp + capture_hash + entropy_hash + password_hash)
       ↓
proof.json saved + audit trail updated
```

### Key Configuration
- Optimal frequency: 100 MHz (FM band)
- Gain: 0 (automatic)
- Sample rate: 2.4 MS/s
- Extraction: Von Neumann for cryptographic use

### Dependencies
- `numpy>=1.20.0`, `scipy>=1.7.0` - Core computation
- `cryptography>=41.0.0` - AES-256-GCM for backups
- `PyQt6>=6.5.0` - GUI interface
- `rtl_sdr` command (optional) - RTL-SDR driver
- GCC (optional) - Compiles RDRAND library on first use

### NIST SP 800-22 Tests (15 total)
Fast mode (9 tests): Monobit, Block Frequency, Runs, Longest Run, Spectral, Serial, Approximate Entropy, Cumulative Sums (2)

Full mode adds: Matrix Rank, Non-overlapping Template, Overlapping Template, Maurer's Universal, Linear Complexity, Random Excursions (2)

### Security Notes
- `secure_zero()` provides best-effort memory clearing
- Proofs use SHA-256 hash chains for tamper detection
- Backups use AES-256-GCM with PBKDF2 key derivation (100k iterations)
- Audit trails use blockchain-style chaining for integrity verification

### GUI Security
- `SecurePasswordDisplay`: Hidden by default, temporary reveal (5s), auto-hide
- `SecurePasswordList`: Masked display, double-click to copy
- Clipboard auto-clear after 30 seconds
- `secure_clear_widget()`: Overwrites with `*` before clearing
- All workers capture stdout to prevent console leak of sensitive data
- Memory cleanup on window close (`closeEvent`)

### Data Storage
All data is stored in `~/.radionoise/`:
```
~/.radionoise/
├── entropy/                          # Validated entropy files
│   └── entropy_2026-02-04_201823_RTL-SDR.bin
├── proofs/                           # Cryptographic proofs
│   └── proof_2026-02-04_201823.json
├── backups/                          # Encrypted backups (AES-256-GCM)
└── audit.db                          # SQLite audit trail
```

**Entropy save workflow:**
1. Capture entropy → memory only
2. Run NIST tests → validation
3. If pass_rate ≥ 95% → prompt to save
4. User confirms → saved to `~/.radionoise/entropy/`

**Proof creation workflow:**
1. Capture with "Conserver données IQ brutes" checked (requires RTL-SDR)
2. Run NIST tests
3. Generate password
4. Go to Traçabilité tab → Create proof
5. Saved to `~/.radionoise/proofs/`

**Audit trail workflow:**
1. Generate password (any source)
2. Go to Traçabilité tab → Audit Trail sub-tab
3. Click "Ajouter la génération actuelle"
4. Database created/updated at `~/.radionoise/audit.db`

**Backup workflow (requires RTL-SDR):**
1. Capture with "Conserver données IQ brutes" checked
2. Generate password
3. Go to Traçabilité tab → Backup sub-tab
4. Enter master password (twice for confirmation)
5. Click "Créer le backup"
6. Backup saved to `~/.radionoise/backups/{timestamp}/`

**Backup recovery:**
1. Go to Traçabilité tab → Backup sub-tab
2. Click "Récupérer..."
3. Select backup from list
4. Enter master password
5. Password is **regenerated** from encrypted IQ data (not stored directly)

**Important:** Backup stores encrypted IQ samples + proof metadata. Recovery reprocesses IQ data (Von Neumann → SHA-512 → generate_password) to regenerate the original password. The password itself is never stored.

**Filename formats:**
- Entropy: `entropy_{YYYY-MM-DD}_{HHMMSS}_{source}.bin`
- Proof: `proof_{YYYY-MM-DD}_{HHMMSS}.json`
- Audit: `audit.db` (SQLite, blockchain-style chain)
- Backup: `{timestamp}/` directory containing `encrypted_iq.bin`, `proof.json`, `crypto_metadata.json`

## Recent Bug Fixes (2026-02-06)

| Bug | File | Fix |
|-----|------|-----|
| **Signal mismatch** | `gui/widgets/generator.py:256` | `_on_progress(num)` → `_on_progress(current, total)` to match `GeneratorWorker.progress` signal `(int, int)` |
| **Passphrase entropy display** | `gui/widgets/generator.py:195` | Changed `12.9 bits/word` → `8 bits/word` (WORDLIST has 256 words = 2^8) |
| **Charset name mismatch** | `gui/widgets/generator.py:206-207` | `"alphanum"` → `"alnum"`, `"numeric"` → `"digits"` to match `CHARSETS` keys in core |
| **UTF-8 encoding** | 8 GUI files | Added `# -*- coding: utf-8 -*-` declaration to prevent mojibake on French strings |
| **Missing exports** | `__init__.py` | Added `capture_entropy_raw`, `calculate_password_entropy`, `calculate_passphrase_entropy` |

## Known Limitations & Security Considerations

### Critical (should fix before production)

1. ~~**Wordlist too small**~~ ✅ FIXED (2026-02-06)
   - Upgraded to EFF large wordlist (7776 words, 12.925 bits/word)
   - Added `core/wordlist.py` with full EFF diceware list
   - Implemented rejection sampling in `generate_passphrase()` (2 bytes/attempt, ~11.9% acceptance)
   - Updated CLI and GUI workers for higher entropy requirements (~20 bytes/word)

2. ~~**tarfile.extractall path traversal**~~ ✅ FIXED (2026-02-06)
   - `import_backup_bundle()` now validates all members before extraction
   - Rejects path traversal (`../`) and symlinks/hardlinks
   - Uses `Path.resolve()` + `is_relative_to()` to verify all paths stay within backup_dir

3. ~~**PBKDF2 iterations too low**~~ ✅ FIXED (2026-02-06)
   - Increased from 100,000 to 600,000 iterations (OWASP recommended for SHA-256)
   - Note: Existing backups created with 100k iterations will need old iteration count for recovery

### Medium (security hardening)

4. ~~**C compilation in /tmp**~~ ✅ FIXED (2026-02-06)
   - Switched from `NamedTemporaryFile` to `mkstemp()` for both `.c` and `.so` files
   - `.so` permissions set to 0o500 (owner read+execute only) before `ctypes.CDLL()` load

5. **No integrity check on compiled .so** (`core/entropy.py`)
   - After compiling RDRAND module, should verify hash before `ctypes.CDLL()` load
   - Prevents tampering between compile and load

6. ~~**Clipboard comparison in plaintext**~~ ✅ FIXED (2026-02-06)
   - `copy_to_clipboard()` now stores SHA-256 hash of the copied password
   - `_clear_clipboard_if_same()` compares hashes instead of plaintext

7. **CSPRNG fallback not post-processed** (`core/entropy.py`)
   - `secrets.token_bytes()` returned directly without Von Neumann + hash pipeline
   - RTL-SDR and RDRAND paths go through full pipeline, CSPRNG should too for uniformity

### Documented Limitations

8. **Passwords as Python str** (all password handling)
   - Python strings are immutable, impossible to securely erase from memory
   - `secure_zero()` only works on numpy arrays
   - For true secure memory: would need `ctypes`/`mmap` buffers (out of scope)
   - **Document this limitation in user-facing docs**

## Future Improvements Roadmap

### Phase 1: Security Hardening
- [x] Upgrade wordlist to EFF large (7776 words) ✅
- [x] Fix tarfile path traversal vulnerability ✅
- [x] Increase PBKDF2 to 600k iterations ✅
- [x] Secure temp file handling for RDRAND compilation ✅
- [x] Hash-based clipboard comparison ✅

### Phase 2: Pipeline Uniformity
- [ ] Apply Von Neumann + hash to CSPRNG fallback
- [ ] Add .so integrity verification
- [ ] Document memory security limitations

### Phase 3: Testing & QA
- [ ] Add GUI unit tests (currently 0% coverage on 3,300+ LOC)
- [ ] Add integration tests for GUI-to-core workflows
- [ ] Security audit of traceability features
