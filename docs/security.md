# Security Model

This document describes RadioNoise's security properties, cryptographic primitives, threat model, and known limitations.

## Threat Model

### What RadioNoise protects against

- **Weak entropy**: Statistical tests catch biased or patterned entropy before use
- **Modulo bias**: Rejection sampling ensures uniform character/word selection
- **Clipboard exposure**: Auto-clear after 30 seconds, hash-based comparison
- **Memory residue**: Best-effort zeroing of numpy arrays after use
- **Tampered proofs**: SHA-256 hash chains detect modification
- **Backup theft**: AES-256-GCM encryption with PBKDF2-derived keys

### What RadioNoise does NOT protect against

- **Compromised host**: If the OS is compromised, all bets are off
- **Physical access**: An attacker with physical access to the RTL-SDR could replace the antenna with a known signal
- **Side channels**: Python is not designed for constant-time operations
- **Memory forensics**: Python strings are immutable and cannot be reliably erased (see below)
- **RTL-SDR firmware attacks**: The dongle firmware is not verified

## Cryptographic Primitives

| Primitive | Usage | Standard |
|-----------|-------|----------|
| SHA-512 | Entropy whitening | FIPS 180-4 |
| SHA-256 | Hash chains, proof signatures | FIPS 180-4 |
| AES-256-GCM | Backup encryption | NIST SP 800-38D |
| PBKDF2-HMAC-SHA256 | Key derivation from master password | NIST SP 800-132 |
| Von Neumann extractor | Bias removal | Von Neumann (1951) |
| NIST SP 800-22 | Statistical validation | NIST SP 800-22 rev1a |

### PBKDF2 Configuration

- **Algorithm**: PBKDF2-HMAC-SHA256
- **Iterations**: 600,000 (OWASP 2024 recommendation for SHA-256)
- **Salt**: 32 bytes from `os.urandom()`
- **Key length**: 32 bytes (256 bits)

### AES-256-GCM Configuration

- **Key**: 256-bit derived from PBKDF2
- **Nonce**: 12 bytes from `os.urandom()`
- **Authentication**: Built-in GCM authentication tag (16 bytes)

## Entropy Quality

### RTL-SDR Path (highest quality)

```
Raw I/Q samples → Von Neumann extraction → SHA-512 whitening → NIST validation
```

- Von Neumann output: 8 bits/byte entropy (provably unbiased)
- SHA-512 ensures uniform distribution
- NIST tests validate statistical quality

### RDRAND Path

Intel/AMD RDRAND draws from a DRBG (Deterministic Random Bit Generator) seeded by on-chip hardware entropy. It is:
- NIST SP 800-90A compliant
- Already uniform (no extraction needed)
- Validated by NIST tests in RadioNoise

### RDSEED Path

RDSEED provides direct access to the CPU's entropy source (slower than RDRAND, bypasses the DRBG). It is:
- Higher quality than RDRAND for seed generation
- Rate-limited by hardware
- Already uniform

### CSPRNG Path

`secrets.token_bytes()` uses the OS entropy pool (`/dev/urandom` on Linux, `CryptGenRandom` on Windows). This is:
- Cryptographically secure
- Already uniform
- The lowest priority fallback

## Password Generation Security

### Rejection Sampling

RadioNoise uses rejection sampling to convert random bytes to characters without modulo bias:

```
threshold = (256 / charset_size) * charset_size
if byte < threshold:
    char = charset[byte % charset_size]
else:
    discard byte
```

This ensures each character has exactly equal probability, unlike naive `byte % charset_size` which introduces measurable bias for charset sizes that don't divide 256.

### Passphrase Generation

Passphrases use the **EFF large wordlist** (7776 words = 6⁵, designed for dice rolls):
- Two bytes combined into a 16-bit value
- Rejection sampling: accept if value < 7776 (acceptance rate: ~11.9%)
- Entropy: ~12.925 bits/word
- 6 words ≈ 77.5 bits of entropy

## Memory Security

### What RadioNoise does

- **`secure_zero()`**: Overwrites numpy arrays with zeros after use
- **Entropy cleanup**: Raw and extracted entropy are zeroed after processing
- **Temp file overwrite**: RTL-SDR capture files are overwritten with random data before deletion
- **GUI auto-hide**: Passwords hidden after 5 seconds
- **Clipboard auto-clear**: Clipboard cleared 30 seconds after copy
- **Secure widgets**: `SecurePasswordDisplay` overwrites with `*` before clearing

### Known Limitations

**Python strings are immutable.** Once a password is created as a `str`, Python may:
- Keep it in memory until garbage collected (timing unpredictable)
- Intern short strings (cached indefinitely)
- Leave copies in interpreter buffers

This is a fundamental limitation of Python's memory model. `secure_zero()` only works on mutable types (numpy arrays, bytearrays). For true secure memory handling, you would need `ctypes` or `mmap` buffers, which is out of scope for this project.

**Mitigations:**
- Keep passwords in memory for the minimum time needed
- GUI auto-hides and auto-clears passwords
- Users should close the application when done

## GUI Security

### SecurePasswordDisplay

- Passwords are hidden by default (masked with `•`)
- Temporary reveal: click to show for 5 seconds, then auto-hide
- `secure_clear_widget()` overwrites the internal text with `*` before clearing

### Clipboard Security

- Passwords are copied to clipboard only on user action
- SHA-256 hash stored for comparison (not the password itself)
- Clipboard auto-cleared after 30 seconds
- Only clears if clipboard still contains the copied password (hash comparison)

### Worker Thread Security

- All blocking operations run in QThread workers
- Workers redirect stdout to prevent sensitive data from appearing in console
- Thread-safe cancellation via `threading.Event`

## Traceability Security

### Proof of Generation

Proofs use SHA-256 hash chains to link each step:

```
signature = SHA-256(timestamp + capture_hash + entropy_hash + password_hash)
```

The password itself is never stored in the proof — only its SHA-256 hash. Verification requires the original password.

### Audit Trail

The SQLite audit trail uses blockchain-style chaining:

```
chain_hash[n] = SHA-256(chain_hash[n-1] + data[n])
```

This ensures:
- Any modification to a past entry invalidates all subsequent chain hashes
- Insertion or deletion of entries is detectable
- The chain can be verified with `--verify-chain`

### Encrypted Backups

Backups store the raw IQ samples (encrypted), not the password. Recovery reprocesses the IQ data through the full pipeline to regenerate the password.

- Encryption: AES-256-GCM (authenticated encryption)
- Key derivation: PBKDF2-HMAC-SHA256, 600,000 iterations
- Each backup has a unique salt and nonce
- Path traversal protection on import (validates all paths, rejects `../` and symlinks)

## Secure File Handling

### RDRAND Compilation

RadioNoise compiles a small C module for RDRAND/RDSEED at runtime:

- Source and output use `mkstemp()` (not `NamedTemporaryFile`)
- Compiled `.so` permissions set to `0o500` (owner read+execute only)
- Temporary files cleaned up after loading

### RTL-SDR Capture Files

- Capture files are written to a temporary directory
- After reading, files are overwritten with `os.urandom()` data before deletion
- Prevents recovery of raw entropy from disk

## Recommendations

1. **Use RTL-SDR** when possible — it provides the highest quality entropy from a physical source
2. **Run full NIST tests** (`--full-test`) for high-security applications
3. **Use passphrases** of 6+ words for accounts you need to remember
4. **Use passwords** of 20+ characters with `safe` or `full` charset for maximum entropy
5. **Enable traceability** (`--proof`) to create an auditable record of password generation
6. **Close the application** after generating passwords to minimize memory exposure
7. **Do not share** proof files — they contain the password hash
