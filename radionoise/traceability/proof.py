"""
RadioNoise Proof of Generation - Cryptographic proof system for password generation.
"""

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np

from radionoise.core.entropy import von_neumann_extract, hash_entropy
from radionoise.core.generator import generate_password, CHARSETS
from radionoise.core.log import get_logger

logger = get_logger('proof')


class ProofOfGeneration:
    """
    Creates cryptographic proof of password generation.

    Proof structure:
    {
        "timestamp": "2025-01-30T16:50:32Z",
        "capture_hash": "sha256 of raw IQ samples",
        "entropy_hash": "sha256 of entropy after Von Neumann",
        "processed_hash": "sha256 of processed entropy",
        "password_hash": "sha256 of generated password",
        "metadata": {
            "frequency": 100000000,
            "sample_rate": 2400000,
            "samples": 500000,
            "location": "optional"
        },
        "signature": "hash(timestamp + capture_hash + entropy_hash + password_hash)"
    }
    """

    def __init__(self, iq_samples_path: Optional[str] = None, entropy_path: Optional[str] = None):
        """
        Initialize with optional paths to save data.

        Args:
            iq_samples_path: Directory to save raw IQ samples
            entropy_path: Directory to save processed entropy
        """
        self.iq_samples_path = iq_samples_path
        self.entropy_path = entropy_path
        self.proofs = []

    def capture_with_proof(
        self,
        frequency: float = 100e6,
        sample_rate: float = 2.4e6,
        samples: int = 500000
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Capture RTL-SDR data and generate initial proof.

        Args:
            frequency: Capture frequency in Hz
            sample_rate: Sample rate in Hz
            samples: Number of samples to capture

        Returns:
            Tuple of (raw_data, proof_dict)
        """
        timestamp = datetime.utcnow().isoformat() + 'Z'

        # Capture
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as temp_file:
            temp_path = temp_file.name

        try:
            result = subprocess.run([
                "rtl_sdr",
                "-f", str(int(frequency)),
                "-s", str(int(sample_rate)),
                "-n", str(samples),
                temp_path
            ], capture_output=True, timeout=30, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"rtl_sdr failed: {result.stderr}")

            # Read raw data
            raw_data = np.fromfile(temp_path, dtype=np.uint8)

            # Hash raw IQ samples
            capture_hash = hashlib.sha256(raw_data.tobytes()).hexdigest()

            # Save IQ samples if requested
            if self.iq_samples_path:
                iq_dir = Path(self.iq_samples_path)
                iq_dir.mkdir(parents=True, exist_ok=True)
                iq_path = iq_dir / f"capture_{timestamp.replace(':', '-')}.iq"
                raw_data.tofile(iq_path)
                logger.info("IQ samples saved: %s", iq_path)

            # Create proof
            proof = {
                "timestamp": timestamp,
                "capture_hash": capture_hash,
                "capture_size": len(raw_data),
                "metadata": {
                    "frequency": int(frequency),
                    "sample_rate": int(sample_rate),
                    "samples": samples,
                    "rtl_sdr_version": result.stderr.split('\n')[0] if result.stderr else "unknown"
                }
            }

            return raw_data, proof

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def process_with_proof(
        self,
        raw_data: np.ndarray,
        initial_proof: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process data (Von Neumann + Hash) and enrich proof.

        Args:
            raw_data: Raw IQ samples
            initial_proof: Proof from capture_with_proof

        Returns:
            Tuple of (processed_data, enriched_proof)
        """
        # Von Neumann extraction
        entropy = von_neumann_extract(raw_data)
        entropy_hash = hashlib.sha256(entropy.tobytes()).hexdigest()

        # SHA-512 whitening
        processed = hash_entropy(entropy)
        processed_hash = hashlib.sha256(processed.tobytes()).hexdigest()

        # Save entropy if requested
        if self.entropy_path:
            entropy_dir = Path(self.entropy_path)
            entropy_dir.mkdir(parents=True, exist_ok=True)
            timestamp = initial_proof['timestamp']
            entropy_file = entropy_dir / f"entropy_{timestamp.replace(':', '-')}.bin"
            processed.tofile(entropy_file)
            logger.info("Entropy saved: %s", entropy_file)

        # Enrich proof
        proof = initial_proof.copy()
        proof.update({
            "entropy_hash": entropy_hash,
            "processed_hash": processed_hash,
            "entropy_size": len(entropy),
            "processed_size": len(processed)
        })

        return processed, proof

    def generate_password_with_proof(
        self,
        processed_data: np.ndarray,
        proof: Dict[str, Any],
        length: int = 16,
        charset: str = "safe"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate password and finalize proof.

        Args:
            processed_data: Processed entropy
            proof: Proof from process_with_proof
            length: Password length
            charset: Character set to use

        Returns:
            Tuple of (password, finalized_proof)
        """
        password = generate_password(processed_data, length, charset)

        # Hash password (for verification without revealing it)
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        # Signature = Hash of all key elements
        signature_data = (
            proof['timestamp'] +
            proof['capture_hash'] +
            proof.get('entropy_hash', '') +
            password_hash
        )
        signature = hashlib.sha256(signature_data.encode()).hexdigest()

        # Finalize proof
        proof.update({
            "password_hash": password_hash,
            "password_length": length,
            "charset": charset,
            "signature": signature
        })

        self.proofs.append(proof)

        return password, proof

    def verify_proof(self, password: str, proof: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Verify that a password matches a given proof.

        Args:
            password: Password to verify
            proof: Proof to verify against

        Returns:
            Tuple of (valid, message)
        """
        # Verify password hash
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        if password_hash != proof['password_hash']:
            return False, "Password hash mismatch"

        # Verify signature
        signature_data = (
            proof['timestamp'] +
            proof['capture_hash'] +
            proof.get('entropy_hash', '') +
            password_hash
        )
        expected_signature = hashlib.sha256(signature_data.encode()).hexdigest()

        if expected_signature != proof['signature']:
            return False, "Signature mismatch"

        return True, "Valid proof"

    def verify_from_iq_samples(
        self,
        iq_file_path: str,
        proof: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Verify a proof by reprocessing original IQ samples.

        Args:
            iq_file_path: Path to IQ samples file
            proof: Proof to verify

        Returns:
            Tuple of (valid, password_or_error_message)
        """
        # Load IQ samples
        raw_data = np.fromfile(iq_file_path, dtype=np.uint8)

        # Verify capture hash
        capture_hash = hashlib.sha256(raw_data.tobytes()).hexdigest()
        if capture_hash != proof['capture_hash']:
            return False, "IQ samples hash mismatch"

        # Reprocess
        entropy = von_neumann_extract(raw_data)
        entropy_hash = hashlib.sha256(entropy.tobytes()).hexdigest()

        if entropy_hash != proof.get('entropy_hash'):
            return False, "Entropy hash mismatch"

        processed = hash_entropy(entropy)
        processed_hash = hashlib.sha256(processed.tobytes()).hexdigest()

        if processed_hash != proof.get('processed_hash'):
            return False, "Processed entropy hash mismatch"

        # Regenerate password
        password = generate_password(processed, proof['password_length'], proof['charset'])

        return True, password

    def save_proof(self, proof: Dict[str, Any], filepath: str) -> None:
        """Save a proof to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(proof, f, indent=2)
        logger.info("Proof saved: %s", filepath)

    def load_proof(self, filepath: str) -> Dict[str, Any]:
        """Load a proof from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)

    def export_audit_trail(self, filepath: str) -> None:
        """Export all proofs as an audit trail."""
        audit = {
            "generated_at": datetime.utcnow().isoformat() + 'Z',
            "total_passwords": len(self.proofs),
            "proofs": self.proofs
        }

        with open(filepath, 'w') as f:
            json.dump(audit, f, indent=2)

        logger.info("Audit trail exported: %s", filepath)
        logger.info("  %d passwords generated", len(self.proofs))
