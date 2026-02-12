#!/usr/bin/env python3
"""
RadioNoise CLI - Command-line interface for password generation with NIST validation.
"""

import argparse
import os
import sys

import numpy as np

from radionoise.core.entropy import (
    HardwareRNG,
    capture_entropy,
    load_entropy_from_file,
    get_last_entropy_source,
)
from radionoise.core.nist import NISTTests
from radionoise.core.generator import (
    generate_password,
    generate_passphrase,
    CHARSETS,
    calculate_password_entropy,
    calculate_passphrase_entropy,
)
from radionoise.core.security import secure_zero


def main():
    parser = argparse.ArgumentParser(
        description="RadioNoise - RTL-SDR password generator with NIST tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Fast standard generation (9 tests)
  %(prog)s --full-test               # Complete slow suite (15 tests)
  %(prog)s --test-only               # NIST tests only (no passwords)
  %(prog)s --no-test                 # Disable tests (faster)
  %(prog)s -f entropy.bin --test-only  # Test existing entropy file

Traceability:
  %(prog)s --proof -n 1              # Generate with cryptographic proof
  %(prog)s --verify proof.json       # Verify a proof file
  %(prog)s --audit ./audit.db -n 1   # Add to audit trail
  %(prog)s --verify-chain ./audit.db # Verify chain integrity
        """
    )

    # Generation options
    gen_group = parser.add_argument_group('Generation')
    gen_group.add_argument("-l", "--length", type=int, default=16,
                          help="Password length (default: 16)")
    gen_group.add_argument("-n", "--count", type=int, default=5,
                          help="Number of passwords (default: 5)")
    gen_group.add_argument("-c", "--charset", choices=CHARSETS.keys(), default="safe",
                          help="Character type (default: safe)")
    gen_group.add_argument("-p", "--passphrase", action="store_true",
                          help="Generate passphrases")
    gen_group.add_argument("-f", "--file", metavar="FILE",
                          help="Use entropy file")
    gen_group.add_argument("--save-entropy", metavar="FILE",
                          help="Save generated entropy")

    # Processing options
    proc_group = parser.add_argument_group('Processing')
    proc_group.add_argument("--no-hash", action="store_true",
                           help="Disable hash (NOT RECOMMENDED)")
    proc_group.add_argument("--hash-algo", choices=['sha256', 'sha512'], default='sha512',
                           help="Hash algorithm (default: sha512)")

    # RTL-SDR options
    sdr_group = parser.add_argument_group('RTL-SDR')
    sdr_group.add_argument("--frequency", type=float, default=100e6,
                          help="Capture frequency in Hz (default: 100 MHz)")
    sdr_group.add_argument("--no-fallback", action="store_true",
                          help="Disable fallback if RTL-SDR unavailable")
    sdr_group.add_argument("--use-rdseed", action="store_true",
                          help="Use RDSEED instead of RDRAND (slower, direct entropy)")

    # Test options
    test_group = parser.add_argument_group('NIST Tests')
    test_group.add_argument("--test-only", action="store_true",
                           help="Run NIST tests only (no generation)")
    test_group.add_argument("--no-test", action="store_true",
                           help="Disable NIST tests (faster)")
    test_group.add_argument("--full-test", action="store_true",
                           help="Run full 15 test suite (slow, ~30s)")

    # Traceability options
    trace_group = parser.add_argument_group('Traceability')
    trace_group.add_argument("--proof", action="store_true",
                            help="Generate cryptographic proof")
    trace_group.add_argument("--proof-output", metavar="FILE", default="proof.json",
                            help="Proof output file (default: proof.json)")
    trace_group.add_argument("--audit", metavar="DB",
                            help="Add to SQLite audit trail")
    trace_group.add_argument("--backup", metavar="DIR",
                            help="Create encrypted backup of IQ samples")
    trace_group.add_argument("--master-password", metavar="PASS",
                            help="Master password (or env RADIONOISE_MASTER_PASS)")

    # Verification options
    verify_group = parser.add_argument_group('Verification')
    verify_group.add_argument("--verify", metavar="PROOF.json",
                             help="Verify a proof file")
    verify_group.add_argument("--recover", metavar="BACKUP_ID",
                             help="Recover password from backup")
    verify_group.add_argument("--audit-report", metavar="OUTPUT.json",
                             help="Export audit report")
    verify_group.add_argument("--verify-chain", metavar="DB",
                             help="Verify audit chain integrity")

    # Output options
    out_group = parser.add_argument_group('Output')
    out_group.add_argument("-q", "--quiet", action="store_true",
                          help="Quiet mode")

    args = parser.parse_args()

    # Handle verification commands first
    if args.verify:
        return _handle_verify(args)

    if args.verify_chain:
        return _handle_verify_chain(args)

    if args.audit_report:
        return _handle_audit_report(args)

    if args.recover:
        return _handle_recover(args)

    # Normal generation flow
    return _handle_generation(args)


def _handle_verify(args):
    """Handle --verify command."""
    import json
    from radionoise.traceability.proof import ProofOfGeneration

    try:
        with open(args.verify, 'r') as f:
            proof = json.load(f)

        print(f"Loaded proof: {args.verify}")
        print(f"   Timestamp: {proof['timestamp']}")
        print(f"   Password hash: {proof['password_hash'][:16]}...")
        print(f"   Signature: {proof['signature'][:16]}...")

        # Ask for password to verify
        import getpass
        password = getpass.getpass("Enter password to verify: ")

        pog = ProofOfGeneration()
        valid, msg = pog.verify_proof(password, proof)

        if valid:
            print(f"\nVALID: {msg}")
            return 0
        else:
            print(f"\nINVALID: {msg}")
            return 1

    except FileNotFoundError:
        print(f"Error: File not found: {args.verify}", file=sys.stderr)
        return 1
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {args.verify}", file=sys.stderr)
        return 1


def _handle_verify_chain(args):
    """Handle --verify-chain command."""
    from radionoise.traceability.audit import ForensicAuditTrail

    if not os.path.exists(args.verify_chain):
        print(f"Error: Database not found: {args.verify_chain}", file=sys.stderr)
        return 1

    audit = ForensicAuditTrail(args.verify_chain)
    valid, errors = audit.verify_chain_integrity()
    audit.close()

    if valid:
        print("Chain integrity: OK")
        stats = audit.get_statistics() if hasattr(audit, 'get_statistics') else {}
        if stats:
            print(f"   Total generations: {stats.get('total_generations', 'N/A')}")
        return 0
    else:
        print("Chain integrity: CORRUPTED")
        for error in errors:
            print(f"   - {error}")
        return 1


def _handle_audit_report(args):
    """Handle --audit-report command."""
    from radionoise.traceability.audit import ForensicAuditTrail

    # Find the audit database
    audit_db = args.audit if args.audit else "./audit_trail.db"
    if not os.path.exists(audit_db):
        print(f"Error: Database not found: {audit_db}", file=sys.stderr)
        return 1

    audit = ForensicAuditTrail(audit_db)
    audit.export_audit_report(args.audit_report)
    audit.close()
    return 0


def _handle_recover(args):
    """Handle --recover command."""
    from radionoise.traceability.backup import SecureBackupSystem
    import getpass

    # Get backup directory
    backup_dir = args.backup if args.backup else "./secure_backup"
    if not os.path.exists(backup_dir):
        print(f"Error: Backup directory not found: {backup_dir}", file=sys.stderr)
        return 1

    # Get master password
    master_password = args.master_password or os.environ.get('RADIONOISE_MASTER_PASS')
    if not master_password:
        master_password = getpass.getpass("Master password: ")

    backup = SecureBackupSystem(backup_dir)

    try:
        password, proof = backup.recover_password(args.recover, master_password)
        if password:
            print(f"\nRecovered password: {password}")
            return 0
        else:
            print("\nRecovery failed")
            return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _handle_generation(args):
    """Handle normal password generation."""
    if not args.quiet and not args.test_only:
        print("=" * 60)
        print("RADIONOISE - RTL-SDR PASSWORD GENERATOR")
        print("=" * 60)
        print()

    try:
        # Get entropy
        if args.file:
            if not args.quiet:
                print(f"Reading {args.file}...")
            random_bytes = load_entropy_from_file(args.file, apply_hash=not args.no_hash)
            raw_iq_data = None
        else:
            if not args.quiet:
                print("RTL-SDR capture in progress...")

            bytes_per_password = args.length * 3
            total_bytes_needed = args.count * bytes_per_password
            samples_needed = total_bytes_needed * 2
            samples = max(500000, samples_needed)

            if not args.quiet:
                print(f"   Samples: {samples:,}")
                print(f"   Frequency: {args.frequency / 1e6:.1f} MHz")

            # If proof/backup needed, capture raw first
            if args.proof or args.backup:
                raw_iq_data = _capture_raw_for_proof(args, samples)
                random_bytes = _process_raw_entropy(raw_iq_data)
            else:
                random_bytes = capture_entropy(
                    samples=samples,
                    frequency=args.frequency,
                    allow_fallback=not args.no_fallback,
                    use_rdseed=args.use_rdseed
                )
                raw_iq_data = None

        # Check quantity
        bytes_needed = args.count * args.length * 3
        if len(random_bytes) < bytes_needed and not args.test_only:
            print(f"ERROR: Not enough entropy!", file=sys.stderr)
            print(f"   Available: {len(random_bytes)} bytes", file=sys.stderr)
            print(f"   Needed: {bytes_needed} bytes", file=sys.stderr)
            sys.exit(1)

        if not args.quiet:
            print(f"Entropy available: {len(random_bytes):,} bytes")
            if not args.no_hash:
                print(f"Hash: {args.hash_algo.upper()}")

        # NIST tests
        if not args.no_test:
            if args.full_test:
                test_size = min(len(random_bytes), 100000)
            else:
                test_size = min(len(random_bytes), 10000)

            test_data = random_bytes[:test_size]

            results = NISTTests.run_all_tests(
                test_data,
                verbose=not args.quiet,
                fast_mode=not args.full_test
            )

            if results['pass_rate'] < 0.95:
                print("\nWARNING: Entropy quality is DOUBTFUL!", file=sys.stderr)
                print("    NOT recommended for crypto use.", file=sys.stderr)
                if not args.test_only:
                    response = input("\nContinue anyway? (y/N): ")
                    if response.lower() != 'y':
                        print("Generation cancelled.")
                        sys.exit(1)

        if args.test_only:
            sys.exit(0)

        if args.save_entropy:
            random_bytes.tofile(args.save_entropy)
            if not args.quiet:
                print(f"\nEntropy saved: {args.save_entropy}")

        if not args.quiet:
            print()

        # Generate passwords
        passwords = []
        offset = 0

        if args.passphrase:
            if not args.quiet:
                print(f"Passphrases ({args.length} words):")
                print("-" * 60)

            for i in range(args.count):
                # EFF wordlist uses rejection sampling: ~11.9% acceptance
                # Need ~17 bytes/word average, use 20 for safety margin
                chunk_size = args.length * 20
                if offset + chunk_size > len(random_bytes):
                    print(f"Entropy exhausted after {i} passphrases", file=sys.stderr)
                    break

                chunk = random_bytes[offset:offset + chunk_size]
                offset += chunk_size

                try:
                    passphrase = generate_passphrase(chunk, words=args.length)
                    passwords.append(passphrase)

                    if args.quiet:
                        print(passphrase)
                    else:
                        entropy_bits = calculate_passphrase_entropy(args.length)
                        print(f"  {passphrase}")
                        print(f"    Entropy: ~{entropy_bits:.0f} bits")
                except ValueError as e:
                    print(f"{e}", file=sys.stderr)
                    break
        else:
            chars = CHARSETS[args.charset]
            bits_per_char = np.log2(len(chars))

            if not args.quiet:
                print(f"Passwords ({args.length} chars, charset={args.charset}):")
                print("-" * 60)

            for i in range(args.count):
                chunk_size = args.length * 3
                if offset + chunk_size > len(random_bytes):
                    print(f"Entropy exhausted after {i} passwords", file=sys.stderr)
                    break

                chunk = random_bytes[offset:offset + chunk_size]
                offset += chunk_size

                try:
                    password = generate_password(chunk, length=args.length, charset=args.charset)
                    passwords.append(password)

                    if args.quiet:
                        print(password)
                    else:
                        entropy_bits = calculate_password_entropy(args.length, args.charset)
                        print(f"  {password}")
                        print(f"    Entropy: ~{entropy_bits:.1f} bits")
                except ValueError as e:
                    print(f"{e}", file=sys.stderr)
                    break

        # Handle traceability
        if args.proof and passwords and raw_iq_data is not None:
            _create_proof(args, passwords[0], raw_iq_data)

        if args.audit and passwords:
            _add_to_audit(args, passwords[0], raw_iq_data)

        if args.backup and passwords and raw_iq_data is not None:
            _create_backup(args, passwords[0], raw_iq_data)

        if not args.quiet:
            print()
            print("=" * 60)
            source_descriptions = {
                "RTL-SDR": "Source: RTL-SDR radio noise (hardware)",
                "RDRAND": "Source: Intel/AMD RDRAND (CPU hardware RNG)",
                "RDSEED": "Source: Intel/AMD RDSEED (CPU hardware entropy)",
                "CSPRNG": "Source: System CSPRNG (urandom)",
                "file": f"Source: File {args.file}",
            }
            last_source = get_last_entropy_source()
            print(source_descriptions.get(last_source, f"Source: {last_source}"))
            if last_source == "RTL-SDR":
                print(f"   Pipeline: Von Neumann -> {args.hash_algo.upper()} -> NIST validated")
            elif last_source in ("RDRAND", "RDSEED"):
                print("   Pipeline: Hardware RNG -> NIST validated")
            else:
                print(f"   Pipeline: {args.hash_algo.upper()} -> NIST validated")
            print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nUser interrupt", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if 'random_bytes' in locals():
            secure_zero(random_bytes)
        HardwareRNG.cleanup()


def _capture_raw_for_proof(args, samples):
    """Capture raw IQ data for proof generation."""
    from radionoise.core.entropy import capture_entropy_raw, is_rtl_sdr_available

    if not is_rtl_sdr_available():
        print("WARNING: RTL-SDR required for proof generation", file=sys.stderr)
        print("   Using fallback source (no proof possible)", file=sys.stderr)
        return None

    try:
        return capture_entropy_raw(
            samples=samples,
            frequency=args.frequency
        )
    except Exception as e:
        print(f"WARNING: Raw capture failed: {e}", file=sys.stderr)
        return None


def _process_raw_entropy(raw_data):
    """Process raw IQ data into usable entropy."""
    from radionoise.core.entropy import von_neumann_extract, hash_entropy
    from radionoise.core.security import secure_zero

    extracted = von_neumann_extract(raw_data)
    hashed = hash_entropy(extracted)
    secure_zero(extracted)
    return hashed


def _create_proof(args, password, raw_iq_data):
    """Create and save proof of generation."""
    import hashlib
    import json
    from datetime import datetime

    if raw_iq_data is None:
        print("WARNING: Cannot create proof without raw IQ data", file=sys.stderr)
        return

    from radionoise.core.entropy import von_neumann_extract, hash_entropy

    timestamp = datetime.utcnow().isoformat() + 'Z'
    capture_hash = hashlib.sha256(raw_iq_data.tobytes()).hexdigest()

    entropy = von_neumann_extract(raw_iq_data)
    entropy_hash = hashlib.sha256(entropy.tobytes()).hexdigest()

    processed = hash_entropy(entropy)
    processed_hash = hashlib.sha256(processed.tobytes()).hexdigest()

    password_hash = hashlib.sha256(password.encode()).hexdigest()

    signature_data = timestamp + capture_hash + entropy_hash + password_hash
    signature = hashlib.sha256(signature_data.encode()).hexdigest()

    proof = {
        "timestamp": timestamp,
        "capture_hash": capture_hash,
        "capture_size": len(raw_iq_data),
        "entropy_hash": entropy_hash,
        "processed_hash": processed_hash,
        "password_hash": password_hash,
        "password_length": args.length,
        "charset": args.charset,
        "signature": signature,
        "metadata": {
            "frequency": int(args.frequency),
            "sample_rate": 2400000,
            "samples": len(raw_iq_data)
        }
    }

    with open(args.proof_output, 'w') as f:
        json.dump(proof, f, indent=2)

    print(f"\nProof saved: {args.proof_output}")


def _add_to_audit(args, password, raw_iq_data):
    """Add generation to audit trail."""
    import hashlib
    import json
    from datetime import datetime
    from radionoise.traceability.audit import ForensicAuditTrail

    if raw_iq_data is None:
        # Create minimal proof for audit
        timestamp = datetime.utcnow().isoformat() + 'Z'
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        signature = hashlib.sha256((timestamp + password_hash).encode()).hexdigest()

        proof = {
            "timestamp": timestamp,
            "capture_hash": "N/A (fallback source)",
            "password_hash": password_hash,
            "signature": signature,
            "metadata": {
                "frequency": int(args.frequency),
                "sample_rate": 2400000,
                "samples": 0
            }
        }
    else:
        from radionoise.core.entropy import von_neumann_extract, hash_entropy

        timestamp = datetime.utcnow().isoformat() + 'Z'
        capture_hash = hashlib.sha256(raw_iq_data.tobytes()).hexdigest()
        entropy = von_neumann_extract(raw_iq_data)
        entropy_hash = hashlib.sha256(entropy.tobytes()).hexdigest()
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        signature = hashlib.sha256((timestamp + capture_hash + entropy_hash + password_hash).encode()).hexdigest()

        proof = {
            "timestamp": timestamp,
            "capture_hash": capture_hash,
            "entropy_hash": entropy_hash,
            "password_hash": password_hash,
            "signature": signature,
            "metadata": {
                "frequency": int(args.frequency),
                "sample_rate": 2400000,
                "samples": len(raw_iq_data)
            }
        }

    audit = ForensicAuditTrail(args.audit)
    gen_id, chain_hash = audit.add_generation(proof)
    audit.close()

    print(f"\nAdded to audit trail: {args.audit}")
    print(f"   Generation ID: {gen_id}")
    print(f"   Chain hash: {chain_hash[:16]}...")


def _create_backup(args, password, raw_iq_data):
    """Create encrypted backup."""
    import hashlib
    import getpass
    import os
    from datetime import datetime
    from radionoise.traceability.backup import SecureBackupSystem
    from radionoise.core.entropy import von_neumann_extract, hash_entropy

    if raw_iq_data is None:
        print("WARNING: Cannot create backup without raw IQ data", file=sys.stderr)
        return

    # Get master password
    master_password = args.master_password or os.environ.get('RADIONOISE_MASTER_PASS')
    if not master_password:
        master_password = getpass.getpass("Master password for backup: ")
        confirm = getpass.getpass("Confirm master password: ")
        if master_password != confirm:
            print("ERROR: Passwords don't match", file=sys.stderr)
            return

    timestamp = datetime.utcnow().isoformat() + 'Z'
    capture_hash = hashlib.sha256(raw_iq_data.tobytes()).hexdigest()
    entropy = von_neumann_extract(raw_iq_data)
    entropy_hash = hashlib.sha256(entropy.tobytes()).hexdigest()
    processed = hash_entropy(entropy)
    processed_hash = hashlib.sha256(processed.tobytes()).hexdigest()
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    signature = hashlib.sha256((timestamp + capture_hash + entropy_hash + password_hash).encode()).hexdigest()

    proof = {
        "timestamp": timestamp,
        "capture_hash": capture_hash,
        "entropy_hash": entropy_hash,
        "processed_hash": processed_hash,
        "password_hash": password_hash,
        "password_length": args.length,
        "charset": args.charset,
        "signature": signature,
        "metadata": {
            "frequency": int(args.frequency),
            "sample_rate": 2400000,
            "samples": len(raw_iq_data)
        }
    }

    backup = SecureBackupSystem(args.backup)
    backup_id = backup.backup_password(password, proof, raw_iq_data, master_password)

    print(f"\nBackup created: {backup_id}")


if __name__ == "__main__":
    sys.exit(main() or 0)
