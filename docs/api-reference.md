# API Reference

## Package Exports

```python
import radionoise

radionoise.__version__  # "2.0.0"
```

All public symbols are available from the top-level `radionoise` package:

```python
from radionoise import (
    # Entropy
    HardwareRNG,
    RDRANDError,
    capture_entropy,
    capture_entropy_fallback,
    capture_entropy_raw,
    von_neumann_extract,
    hash_entropy,
    load_entropy_from_file,
    is_rtl_sdr_available,
    get_last_entropy_source,
    # NIST
    NISTTests,
    # Generator
    generate_password,
    generate_passphrase,
    calculate_password_entropy,
    calculate_passphrase_entropy,
    CHARSETS,
    CharsetName,
    # Security
    secure_zero,
)
```

---

## Entropy (`radionoise.core.entropy`)

### `capture_entropy()`

Capture entropy from RTL-SDR with automatic fallback.

```python
def capture_entropy(
    samples: int = 500000,
    frequency: float = 100e6,
    sample_rate: float = 2.4e6,
    gain: int = 0,
    allow_fallback: bool = True,
    use_rdseed: bool = False
) -> np.ndarray
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `samples` | `int` | 500000 | Number of samples to capture |
| `frequency` | `float` | 100e6 | Capture frequency in Hz |
| `sample_rate` | `float` | 2.4e6 | Sample rate in Hz |
| `gain` | `int` | 0 | RTL-SDR gain (0 = auto) |
| `allow_fallback` | `bool` | True | Fall back to RDRAND/CSPRNG if RTL-SDR unavailable |
| `use_rdseed` | `bool` | False | Prefer RDSEED over RDRAND for fallback |

**Returns:** `np.ndarray` — Processed entropy as uint8 array (Von Neumann + SHA-512 applied).

**Raises:** `RuntimeError` if RTL-SDR unavailable and `allow_fallback=False`.

**Example:**
```python
entropy = capture_entropy(samples=500000, frequency=100e6)
print(f"Got {len(entropy)} bytes of entropy")
```

### `capture_entropy_raw()`

Capture raw I/Q samples without processing. Used for traceability proofs.

```python
def capture_entropy_raw(
    samples: int = 500000,
    frequency: float = 100e6,
    sample_rate: float = 2.4e6,
    gain: int = 0,
) -> np.ndarray
```

**Returns:** `np.ndarray` — Raw I/Q samples as uint8 array (unprocessed).

**Raises:** `RuntimeError` if RTL-SDR not available.

### `capture_entropy_fallback()`

Capture entropy from CPU hardware RNG or system CSPRNG.

```python
def capture_entropy_fallback(
    bytes_needed: int,
    use_rdseed: bool = False
) -> np.ndarray
```

**Returns:** `np.ndarray` — Entropy as uint8 array.

### `von_neumann_extract()`

Remove bias from raw data using the Von Neumann extractor.

```python
def von_neumann_extract(data: np.ndarray) -> np.ndarray
```

**Parameters:**
- `data` — Raw entropy as uint8 array

**Returns:** `np.ndarray` — Unbiased entropy as uint8 array (~3% of input size).

### `hash_entropy()`

Apply SHA-512 whitening to entropy data.

```python
def hash_entropy(
    raw_entropy: np.ndarray,
    hash_algo: Literal['sha256', 'sha512'] = 'sha512'
) -> np.ndarray
```

**Parameters:**
- `raw_entropy` — Entropy to whiten
- `hash_algo` — Hash algorithm (`'sha256'` or `'sha512'`)

**Returns:** `np.ndarray` — Whitened entropy as uint8 array (same size as input).

### `load_entropy_from_file()`

Load entropy from a binary file.

```python
def load_entropy_from_file(
    filepath: str,
    apply_hash: bool = True
) -> np.ndarray
```

**Parameters:**
- `filepath` — Path to binary entropy file
- `apply_hash` — Apply SHA-512 whitening after loading

**Returns:** `np.ndarray` — Entropy as uint8 array.

### `is_rtl_sdr_available()`

Check if RTL-SDR is available and a device is connected.

```python
def is_rtl_sdr_available() -> bool
```

### `get_last_entropy_source()`

Get the name of the last entropy source used.

```python
def get_last_entropy_source() -> str
```

**Returns:** One of `"RTL-SDR"`, `"RDRAND"`, `"RDSEED"`, `"CSPRNG"`, `"file"`, `"unknown"`.

### `HardwareRNG`

Interface for Intel/AMD RDRAND/RDSEED instructions.

```python
class HardwareRNG:
    @staticmethod
    def cleanup() -> None
        """Clean up compiled shared libraries."""
```

### Exceptions

```python
class RDRANDError(Exception): ...
class RTLSDRError(Exception): ...
class DeviceNotFoundError(RTLSDRError): ...
class DeviceDisconnectedError(RTLSDRError): ...
```

---

## NIST Tests (`radionoise.core.nist`)

### `NISTTests`

```python
class NISTTests:
    @classmethod
    def run_all_tests(
        cls,
        data: np.ndarray,
        verbose: bool = True,
        fast_mode: bool = False
    ) -> dict
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `np.ndarray` | — | Entropy as uint8 array |
| `verbose` | `bool` | True | Print results to logger |
| `fast_mode` | `bool` | False | Run 9 fast tests only (vs 15 full) |

**Returns:**
```python
{
    "tests": [
        {
            "name": "Frequency (Monobit)",
            "p_value": 0.534146,
            "passed": True,
            "statistic": 0.6196
        },
        # ... more tests
    ],
    "passed": 9,      # Number of tests passed
    "total": 9,        # Total tests run
    "pass_rate": 1.0   # Ratio (0.0 to 1.0)
}
```

### Individual Tests

All test methods are static and take a bit array (`np.ndarray`):

```python
NISTTests.frequency_monobit_test(bits)        # Test 1
NISTTests.frequency_block_test(bits, 128)     # Test 2
NISTTests.runs_test(bits)                     # Test 3
NISTTests.longest_run_test(bits)              # Test 4
NISTTests.spectral_test(bits)                 # Test 5
NISTTests.serial_test(bits, 2)                # Test 6
NISTTests.approximate_entropy_test(bits, 2)   # Test 7
NISTTests.cumulative_sums_test(bits, 'forward')   # Test 8
NISTTests.cumulative_sums_test(bits, 'backward')  # Test 9
NISTTests.binary_matrix_rank_test(bits)       # Test 10
NISTTests.non_overlapping_template_test(bits) # Test 11
NISTTests.overlapping_template_test(bits)     # Test 12
NISTTests.maurers_universal_test(bits)        # Test 13
NISTTests.linear_complexity_test(bits)        # Test 14
NISTTests.random_excursions_test(bits)        # Test 15a
NISTTests.random_excursions_variant_test(bits)# Test 15b
```

Each returns:
```python
{"name": str, "p_value": float, "passed": bool, "statistic": float}
```

Or `None` if insufficient data.

---

## Generator (`radionoise.core.generator`)

### `generate_password()`

Generate a password using rejection sampling for uniform distribution.

```python
def generate_password(
    random_bytes: np.ndarray,
    length: int = 16,
    charset: CharsetName = "safe"
) -> str
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_bytes` | `np.ndarray` | — | Random entropy as uint8 array |
| `length` | `int` | 16 | Password length in characters |
| `charset` | `CharsetName` | `"safe"` | Character set to use |

**Returns:** `str` — Generated password.

**Raises:** `ValueError` if not enough entropy.

### `generate_passphrase()`

Generate a passphrase from the EFF large wordlist (7776 words).

```python
def generate_passphrase(
    random_bytes: np.ndarray,
    words: int = 6
) -> str
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_bytes` | `np.ndarray` | — | Random entropy as uint8 array |
| `words` | `int` | 6 | Number of words (~12.925 bits/word) |

**Returns:** `str` — Passphrase with words joined by hyphens.

**Raises:** `ValueError` if not enough entropy (~20 bytes/word needed).

### `calculate_password_entropy()`

```python
def calculate_password_entropy(
    length: int,
    charset: CharsetName = "safe"
) -> float
```

**Returns:** Entropy in bits.

### `calculate_passphrase_entropy()`

```python
def calculate_passphrase_entropy(words: int) -> float
```

**Returns:** Entropy in bits (~12.925 × words).

### `CHARSETS`

Dictionary of available character sets:

```python
CHARSETS = {
    "alnum": "a-zA-Z0-9",                      # 62 chars, 5.95 bits/char
    "alpha": "a-zA-Z",                          # 52 chars, 5.70 bits/char
    "digits": "0-9",                            # 10 chars, 3.32 bits/char
    "hex": "0-9a-f",                            # 16 chars, 4.00 bits/char
    "full": "a-zA-Z0-9 + 25 symbols",           # 87 chars, 6.44 bits/char
    "safe": "a-zA-Z0-9 + -_!@#$%",              # 70 chars, 6.13 bits/char
}
```

### `CharsetName`

Type alias: `Literal["alnum", "alpha", "digits", "hex", "full", "safe"]`

---

## Security (`radionoise.core.security`)

### `secure_zero()`

Overwrite a numpy array or bytearray with zeros.

```python
def secure_zero(data) -> None
```

**Note:** Only works on mutable types. Python `str` cannot be securely erased.

---

## Traceability

### `ProofOfGeneration` (`radionoise.traceability.proof`)

```python
class ProofOfGeneration:
    def __init__(self, iq_samples_path=None, entropy_path=None)

    def capture_with_proof(self, samples=500000, frequency=100e6) -> Tuple[np.ndarray, dict]
    def process_with_proof(self, raw_data, proof) -> Tuple[np.ndarray, dict]
    def generate_password_with_proof(self, processed, proof, length=16, charset="safe") -> Tuple[str, dict]

    def verify_proof(self, password: str, proof: dict) -> Tuple[bool, str]
    def verify_from_iq_samples(self, iq_data, proof) -> Tuple[bool, str]

    def save_proof(self, proof: dict, filepath: str) -> None
    def load_proof(self, filepath: str) -> dict
    def export_audit_trail(self, filepath: str) -> None
```

### `ForensicAuditTrail` (`radionoise.traceability.audit`)

```python
class ForensicAuditTrail:
    def __init__(self, db_path: str = "./audit_trail.db")

    def add_generation(self, proof: dict) -> Tuple[str, str]
    def log_event(self, gen_id: str, event_type: str, description: str) -> None
    def verify_chain_integrity(self) -> Tuple[bool, List[str]]

    def get_generation(self, gen_id: str) -> dict
    def get_timeline(self, limit: int = 50) -> List[dict]
    def get_statistics(self) -> dict
    def prove_generation_time(self, gen_id: str) -> dict
    def search_by_context(self, context_filter: str) -> List[dict]

    def export_audit_report(self, filepath: str) -> None
    def close(self) -> None
```

### `SecureBackupSystem` (`radionoise.traceability.backup`)

```python
class SecureBackupSystem:
    def __init__(self, backup_dir: str = "./secure_backup")

    def backup_password(self, password, proof, raw_iq_data, master_password) -> str
    def recover_password(self, backup_id: str, master_password: str) -> Tuple[str, dict]

    def list_backups(self) -> None
    def get_backup_list(self) -> dict
    def delete_backup(self, backup_id: str) -> bool

    def export_backup_bundle(self, backup_id: str, output_file: str) -> None
    def import_backup_bundle(self, bundle_file: str) -> None
```

---

## Configuration (`radionoise.config`)

### `Config`

```python
class Config:
    def __init__(self, config_file: Optional[Path] = None)

    def get(self, section: str, key: str) -> Any
    def set(self, section: str, key: str, value: Any) -> None
    def save(self) -> None
```

**Default config file:** `~/.radionoise/config.json`

**Sections and defaults:**

| Section | Key | Default |
|---------|-----|---------|
| `capture` | `frequency_mhz` | 100.0 |
| `capture` | `samples` | 500000 |
| `capture` | `allow_fallback` | True |
| `capture` | `use_rdseed` | False |
| `capture` | `capture_raw` | True |
| `nist` | `fast_mode` | True |
| `generator` | `type` | "password" |
| `generator` | `length` | 16 |
| `generator` | `count` | 5 |
| `generator` | `charset` | "safe" |
| `gui` | `theme` | "dark" |

---

## GUI (`radionoise.gui`)

### Main Window

```python
from radionoise.gui.main_window import MainWindow
```

### Styles

```python
from radionoise.gui.styles.dark_theme import apply_dark_theme
```

### Secure Widgets

```python
from radionoise.gui.widgets.secure_widgets import (
    SecurePasswordDisplay,
    SecurePasswordList,
)
```

### Workers

```python
from radionoise.gui.workers.capture_worker import CaptureWorker
from radionoise.gui.workers.nist_worker import NistWorker
from radionoise.gui.workers.generator_worker import GeneratorWorker
```

All workers inherit from `BaseWorker(QThread)` and support:
- `cancel()` — Thread-safe cancellation
- `is_cancelled` — Check cancellation state
- Standard Qt signals for progress and completion
