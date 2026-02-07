# Architecture

## High-Level Overview

```mermaid
flowchart TD
    User([User])

    subgraph Interface["Interface Layer"]
        CLI["radionoise.cli\nCommand-line interface"]
        GUI["radionoise.gui\nPyQt6 graphical interface"]
    end

    subgraph Core["Core Layer"]
        Entropy["core.entropy\nCapture & extraction"]
        NIST["core.nist\nStatistical validation"]
        Generator["core.generator\nPassword generation"]
        Security["core.security\nMemory cleanup"]
    end

    subgraph Traceability["Traceability Layer"]
        Proof["traceability.proof\nCryptographic proofs"]
        Audit["traceability.audit\nSQLite audit trail"]
        Backup["traceability.backup\nEncrypted backups"]
    end

    subgraph External["External"]
        RTLSDR["RTL-SDR\nHardware dongle"]
        RDRAND["RDRAND/RDSEED\nCPU instructions"]
        CSPRNG["OS CSPRNG\nsecrets module"]
    end

    User --> CLI
    User --> GUI
    CLI --> Core
    GUI --> Core
    Core --> Traceability
    Entropy --> RTLSDR
    Entropy --> RDRAND
    Entropy --> CSPRNG
    Entropy --> Security
    Generator --> Security

    style Interface fill:#1e3a5f,color:#fff
    style Core fill:#2d5986,color:#fff
    style Traceability fill:#5b21b6,color:#fff
    style External fill:#374151,color:#fff
```

## Module Dependency Graph

```mermaid
flowchart LR
    subgraph Package["radionoise"]
        init["__init__.py\n(public API)"]
        cli["cli.py"]
        config["config.py"]

        subgraph core["core/"]
            entropy["entropy.py"]
            nist["nist.py"]
            generator["generator.py"]
            security["security.py"]
            wordlist["wordlist.py"]
            log["log.py"]
        end

        subgraph trace["traceability/"]
            proof["proof.py"]
            audit["audit.py"]
            backup["backup.py"]
        end

        subgraph gui["gui/"]
            main_window["main_window.py"]
            widgets["widgets/"]
            workers["workers/"]
            styles["styles/"]
        end
    end

    cli --> entropy
    cli --> nist
    cli --> generator
    cli --> proof
    cli --> audit
    cli --> backup
    cli --> security

    init --> entropy
    init --> nist
    init --> generator
    init --> security

    generator --> wordlist
    entropy --> security
    entropy --> log

    proof --> entropy
    backup --> entropy
    backup --> generator

    main_window --> widgets
    main_window --> styles
    widgets --> workers
    workers --> entropy
    workers --> nist
    workers --> generator

    style Package fill:#1e293b,color:#fff
    style core fill:#1e3a5f,color:#fff
    style trace fill:#5b21b6,color:#fff
    style gui fill:#059669,color:#fff
```

## Data Flow: Capture to Password

```mermaid
sequenceDiagram
    participant U as User
    participant C as CLI/GUI
    participant E as entropy.py
    participant R as RTL-SDR
    participant VN as Von Neumann
    participant H as SHA-512
    participant N as nist.py
    participant G as generator.py

    U->>C: Request passwords
    C->>E: capture_entropy(samples=500000)
    E->>R: rtl_sdr -f 100MHz -n 500000
    R-->>E: Raw I/Q data (488 KB)
    E->>VN: von_neumann_extract(raw)
    VN-->>E: Unbiased bits (~15 KB)
    E->>H: hash_entropy(extracted)
    H-->>E: Whitened data (~15 KB)
    E-->>C: Processed entropy

    C->>N: run_all_tests(entropy, fast_mode)
    N-->>C: {pass_rate: 1.0, tests: [...]}

    loop For each password
        C->>G: generate_password(chunk, length, charset)
        G-->>C: "x7$kL9m#Qp2..."
    end

    C-->>U: Display passwords
```

## GUI Architecture

```mermaid
flowchart TD
    subgraph MainWindow["MainWindow"]
        TabWidget["QTabWidget"]
        StatusBar["Status Bar\n(info_label + progress_bar)"]
    end

    subgraph Tabs["Tabs (progressive unlock)"]
        Tab0["Tab 0: EntropyWidget\n🔓 Always enabled"]
        Tab1["Tab 1: NistWidget\n🔒 After capture"]
        Tab2["Tab 2: GeneratorWidget\n🔒 After capture"]
        Tab3["Tab 3: TraceabilityWidget\n🔒 After generation"]
    end

    subgraph Workers["QThread Workers"]
        CW["CaptureWorker\nRTL-SDR / RDRAND"]
        NW["NistWorker\n9 or 15 tests"]
        GW["GeneratorWorker\nPasswords/Passphrases"]
    end

    subgraph SecWidgets["Secure Widgets"]
        SPD["SecurePasswordDisplay\nAuto-hide, masked"]
        SPL["SecurePasswordList\nMasked list, click to copy"]
    end

    TabWidget --> Tabs
    Tab0 -.->|"entropy_captured"| Tab1
    Tab0 -.->|"entropy_captured"| Tab2
    Tab2 -.->|"passwords_generated"| Tab3

    Tab0 --> CW
    Tab1 --> NW
    Tab2 --> GW
    Tab2 --> SecWidgets

    style MainWindow fill:#1e3a5f,color:#fff
    style Tabs fill:#374151,color:#fff
    style Workers fill:#059669,color:#fff
    style SecWidgets fill:#dc2626,color:#fff
```

### GUI Design Principles

- **Progressive disclosure**: Tabs unlock as the workflow progresses (capture → test → generate → trace)
- **Async operations**: All blocking work runs in QThread workers to keep the UI responsive
- **Security by default**: Passwords hidden, clipboard auto-cleared, memory overwritten on close
- **System fonts**: Uses `QFontDatabase.systemFont(FixedFont)` for monospace, system default for UI

### Worker Communication

All workers inherit from `BaseWorker(QThread)`:
- **Cancellation**: `threading.Event` for thread-safe cancel (`self.is_cancelled`)
- **Stdout suppression**: `contextlib.redirect_stdout` prevents sensitive data in console
- **Signals**: Type-safe Qt signals for progress and completion

Widget cleanup is handled by `WorkerWidgetMixin`:
- `_cleanup_worker()`: Disconnects signals, waits for thread, sets worker to None
- Used by all three widget classes

## File Storage

All persistent data is stored in `~/.radionoise/`:

```
~/.radionoise/
├── config.json              # User preferences
├── entropy/                 # Saved entropy files
│   └── entropy_{date}_{time}_{source}.bin
├── proofs/                  # Cryptographic proofs
│   └── proof_{date}_{time}.json
├── backups/                 # Encrypted backups
│   └── {timestamp}/
│       ├── encrypted_iq.bin
│       ├── proof.json
│       └── crypto_metadata.json
└── audit.db                 # SQLite audit trail
```

### Configuration

`config.json` stores user preferences with deep-merge defaults:

```json
{
  "capture": {
    "frequency_mhz": 100.0,
    "samples": 500000,
    "allow_fallback": true,
    "use_rdseed": false,
    "capture_raw": true
  },
  "nist": {
    "fast_mode": true
  },
  "generator": {
    "type": "password",
    "length": 16,
    "count": 5,
    "charset": "safe"
  },
  "gui": {
    "theme": "dark"
  }
}
```

Missing keys are filled from defaults automatically.

## Package Entry Points

| Entry Point | Module | Description |
|-------------|--------|-------------|
| `python RadioNoise.py` | `radionoise.cli:main` | Direct script (compatibility shim) |
| `radionoise` | `radionoise.cli:main` | Installed package command |
| `python radionoise_gui.py` | `radionoise.gui:MainWindow` | GUI launcher |
| `import radionoise` | `radionoise.__init__` | Python library |
