# Installation Guide

## Prerequisites

- **Python 3.10+** (3.11 or 3.12 recommended)
- **pip** (included with Python)
- **RTL-SDR dongle** (optional but recommended for true hardware entropy)

## Quick Install

```bash
git clone https://github.com/Sherub1/RadioNoise.git
cd RadioNoise
pip install -r requirements.txt
```

## Developer Install

```bash
git clone https://github.com/Sherub1/RadioNoise.git
cd RadioNoise
pip install -e ".[dev]"
```

This installs the package in editable mode with development dependencies (pytest, ruff, coverage).

## GUI Install

The GUI requires PyQt6:

```bash
pip install -e ".[gui]"
# or
pip install PyQt6>=6.5.0
```

## RTL-SDR Setup

An RTL-SDR USB dongle (~$10-30) provides true hardware entropy from radio noise. Without it, RadioNoise falls back to CPU hardware RNG (RDRAND/RDSEED) or system CSPRNG.

### Linux

```bash
# Install rtl-sdr tools
sudo apt install rtl-sdr librtlsdr-dev    # Debian/Ubuntu
sudo dnf install rtl-sdr rtl-sdr-devel     # Fedora
sudo pacman -S rtl-sdr                     # Arch

# Blacklist the DVB-T kernel module (conflicts with rtl-sdr)
echo "blacklist dvb_usb_rtl28xxu" | sudo tee /etc/modprobe.d/blacklist-rtlsdr.conf
sudo modprobe -r dvb_usb_rtl28xxu

# Add udev rules for non-root access
sudo cp /usr/share/doc/rtl-sdr/rtl-sdr.rules /etc/udev/rules.d/
# Or create manually:
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2838", MODE="0666"' \
  | sudo tee /etc/udev/rules.d/20-rtlsdr.rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# Unplug and replug the RTL-SDR dongle
```

### macOS

```bash
# Install via Homebrew
brew install librtlsdr

# Verify
rtl_test -t
```

### Windows

1. Download the RTL-SDR release package from [osmocom.org](https://osmocom.org/projects/rtl-sdr/wiki)
2. Install [Zadig](https://zadig.akeo.ie/) to replace the default driver:
   - Open Zadig, select the RTL-SDR device
   - Replace the driver with **WinUSB**
3. Add the `rtl_sdr.exe` directory to your system PATH
4. Verify: `rtl_test -t` in a terminal

## Optional: GCC (for RDRAND/RDSEED)

RadioNoise compiles a small C module at runtime for Intel/AMD hardware RNG instructions. GCC is needed for this.

```bash
# Linux
sudo apt install gcc       # Debian/Ubuntu
sudo dnf install gcc       # Fedora
sudo pacman -S gcc         # Arch

# macOS
xcode-select --install     # Installs Apple Clang (GCC-compatible)

# Windows
# Install MinGW-w64 or use WSL
```

If GCC is not available, RadioNoise skips RDRAND/RDSEED and falls back to the system CSPRNG (`secrets.token_bytes()`).

## Verifying the Installation

### Test CLI

```bash
# Basic test (uses fallback if no RTL-SDR)
python RadioNoise.py --test-only

# Generate passwords
python RadioNoise.py -n 3 -l 20
```

### Test RTL-SDR connection

```bash
# Check if device is detected
rtl_test -t

# Quick capture test
rtl_sdr -f 100000000 -s 2400000 -n 1000000 /dev/null
```

### Run test suite

```bash
pytest tests/ -v
```

### Launch GUI

```bash
python radionoise_gui.py
```

## Troubleshooting

### "No supported devices found"
- Verify the RTL-SDR dongle is plugged in: `lsusb | grep RTL` (Linux)
- Ensure the DVB-T kernel module is blacklisted (see Linux setup above)
- Try a different USB port or cable

### "usb_claim_interface error"
- Another program is using the device (close SDR#, GQRX, etc.)
- On Linux: `sudo rmmod dvb_usb_rtl28xxu`

### "Permission denied" on Linux
- Ensure udev rules are installed (see Linux setup above)
- Alternatively, run with `sudo` (not recommended for production)

### GCC compilation fails
- This only affects RDRAND/RDSEED support
- RadioNoise will automatically fall back to system CSPRNG
- To fix: install GCC (see above)

### PyQt6 won't install
- Requires Python 3.10+
- On Linux, you may need: `sudo apt install python3-pyqt6`
- On older systems, try: `pip install PyQt6==6.5.0`
