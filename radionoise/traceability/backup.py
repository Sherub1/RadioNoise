"""
RadioNoise Secure Backup - Encrypted backup and recovery system.
"""

import hashlib
import json
import os
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from radionoise.core.entropy import von_neumann_extract, hash_entropy
from radionoise.core.generator import generate_password


class SecureBackupSystem:
    """
    Secure backup/recovery system for RadioNoise.

    Architecture:
    1. IQ samples encrypted with AES-256-GCM
    2. Key derived from master password (PBKDF2)
    3. Metadata stored separately
    4. Recovery possible with master password + proof
    """

    def __init__(self, backup_dir: str = "./secure_backup"):
        """
        Initialize backup system.

        Args:
            backup_dir: Directory for backups
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.backup_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _derive_key(self, master_password: str, salt: bytes) -> bytes:
        """Derive AES-256 key from master password."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 256 bits
            salt=salt,
            iterations=600000,
            backend=default_backend()
        )
        return kdf.derive(master_password.encode())

    def _encrypt_iq_samples(
        self,
        iq_data: np.ndarray,
        master_password: str
    ) -> Tuple[bytes, bytes, bytes, bytes]:
        """
        Encrypt IQ samples with AES-256-GCM.

        Args:
            iq_data: Raw IQ samples
            master_password: Master password for encryption

        Returns:
            Tuple of (encrypted_data, salt, nonce, tag)
        """
        # Generate random salt and nonce
        salt = os.urandom(16)
        nonce = os.urandom(12)  # GCM recommends 12 bytes

        # Derive key
        key = self._derive_key(master_password, salt)

        # Encrypt with AES-GCM (authenticated)
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(nonce),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()

        ciphertext = encryptor.update(iq_data.tobytes()) + encryptor.finalize()

        return ciphertext, salt, nonce, encryptor.tag

    def _decrypt_iq_samples(
        self,
        encrypted_data: bytes,
        salt: bytes,
        nonce: bytes,
        tag: bytes,
        master_password: str
    ) -> np.ndarray:
        """
        Decrypt IQ samples.

        Args:
            encrypted_data: Encrypted IQ data
            salt: PBKDF2 salt
            nonce: GCM nonce
            tag: GCM authentication tag
            master_password: Master password

        Returns:
            Decrypted IQ samples as numpy array
        """
        # Derive key
        key = self._derive_key(master_password, salt)

        # Decrypt
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(nonce, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()

        plaintext = decryptor.update(encrypted_data) + decryptor.finalize()

        return np.frombuffer(plaintext, dtype=np.uint8)

    def backup_password(
        self,
        password: str,
        proof: Dict[str, Any],
        iq_data: np.ndarray,
        master_password: str,
        label: Optional[str] = None
    ) -> str:
        """
        Complete backup of a password with encrypted IQ samples.

        Structure:
        - backup_<timestamp>/
            - encrypted_iq.bin (encrypted IQ)
            - proof.json (public proof)
            - crypto_metadata.json (salt, nonce, tag)

        Args:
            password: The generated password (for verification)
            proof: Proof of generation
            iq_data: Raw IQ samples
            master_password: Master password for encryption
            label: Optional label for the backup

        Returns:
            Backup ID
        """
        timestamp = proof['timestamp']
        backup_id = timestamp.replace(':', '-').replace('.', '-')

        if label:
            backup_id = f"{label}_{backup_id}"

        backup_path = self.backup_dir / backup_id
        backup_path.mkdir(exist_ok=True)

        print(f"Creating backup: {backup_id}")

        # 1. Encrypt IQ samples
        encrypted, salt, nonce, tag = self._encrypt_iq_samples(iq_data, master_password)

        with open(backup_path / "encrypted_iq.bin", 'wb') as f:
            f.write(encrypted)

        # 2. Save proof (public)
        with open(backup_path / "proof.json", 'w') as f:
            json.dump(proof, f, indent=2)

        # 3. Save crypto metadata (public)
        crypto_metadata = {
            "salt": salt.hex(),
            "nonce": nonce.hex(),
            "tag": tag.hex(),
            "iq_size": len(iq_data),
            "encrypted_size": len(encrypted)
        }

        with open(backup_path / "crypto_metadata.json", 'w') as f:
            json.dump(crypto_metadata, f, indent=2)

        # 4. Add to global index
        self.metadata[backup_id] = {
            "timestamp": timestamp,
            "label": label,
            "password_hash": proof['password_hash'],
            "backup_path": str(backup_path),
            "created_at": datetime.utcnow().isoformat() + 'Z'
        }
        self._save_metadata()

        print(f"Backup created: {backup_path}")
        print(f"   - Encrypted IQ: {len(encrypted):,} bytes")
        print(f"   - Proof: proof.json")
        print(f"   - Crypto metadata: crypto_metadata.json")

        return backup_id

    def recover_password(
        self,
        backup_id: str,
        master_password: str
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Recover a password from a backup.
        Requires correct master password.

        Args:
            backup_id: Backup identifier
            master_password: Master password for decryption

        Returns:
            Tuple of (password, proof) or (None, None) on failure
        """
        backup_path = self.backup_dir / backup_id

        if not backup_path.exists():
            raise ValueError(f"Backup not found: {backup_id}")

        print(f"Recovering backup: {backup_id}")

        # 1. Load metadata
        with open(backup_path / "crypto_metadata.json", 'r') as f:
            crypto_meta = json.load(f)

        with open(backup_path / "proof.json", 'r') as f:
            proof = json.load(f)

        # 2. Load encrypted IQ
        with open(backup_path / "encrypted_iq.bin", 'rb') as f:
            encrypted = f.read()

        # 3. Decrypt
        salt = bytes.fromhex(crypto_meta['salt'])
        nonce = bytes.fromhex(crypto_meta['nonce'])
        tag = bytes.fromhex(crypto_meta['tag'])

        try:
            iq_data = self._decrypt_iq_samples(encrypted, salt, nonce, tag, master_password)
            print(f"IQ samples decrypted: {len(iq_data):,} bytes")
        except Exception as e:
            print(f"Decryption failed: {e}")
            print("   Incorrect master password?")
            return None, None

        # 4. Reprocess to regenerate password
        entropy = von_neumann_extract(iq_data)
        processed = hash_entropy(entropy)
        # Use defaults for old backups without these fields
        password_length = proof.get('password_length', 16)
        charset = proof.get('charset', 'safe')
        password = generate_password(processed, password_length, charset)

        # 5. Verify
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        if password_hash != proof['password_hash']:
            print("ERROR: Regenerated password does not match proof!")
            print("   IQ samples may be corrupted.")
            return None, proof

        print("Password regenerated and verified")

        return password, proof

    def list_backups(self) -> None:
        """List all available backups."""
        print("=" * 70)
        print("AVAILABLE BACKUPS")
        print("=" * 70)

        if not self.metadata:
            print("No backups found.")
            return

        for backup_id, meta in self.metadata.items():
            print(f"\n{backup_id}")
            print(f"   Timestamp: {meta['timestamp']}")
            print(f"   Label: {meta.get('label', 'N/A')}")
            print(f"   Password hash: {meta['password_hash'][:16]}...")
            print(f"   Created: {meta['created_at']}")

    def get_backup_list(self) -> Dict[str, Dict[str, Any]]:
        """Return backup metadata dictionary."""
        return self.metadata.copy()

    def export_backup_bundle(self, backup_id: str, output_file: str) -> None:
        """
        Export a complete backup as a single file (for archiving).
        Format: tar.gz containing all backup files.

        Args:
            backup_id: Backup identifier
            output_file: Output file path
        """
        backup_path = self.backup_dir / backup_id

        if not backup_path.exists():
            raise ValueError(f"Backup not found: {backup_id}")

        with tarfile.open(output_file, "w:gz") as tar:
            tar.add(backup_path, arcname=backup_id)

        print(f"Backup exported: {output_file}")
        print("   Can be restored with import_backup_bundle()")

    def import_backup_bundle(self, bundle_file: str) -> None:
        """
        Import a backup from a tar.gz file.

        Args:
            bundle_file: Path to the bundle file

        Raises:
            ValueError: If the archive contains path traversal attempts
        """
        with tarfile.open(bundle_file, "r:gz") as tar:
            # Validate all members before extracting any
            dest = self.backup_dir.resolve()
            for member in tar.getmembers():
                member_path = (dest / member.name).resolve()
                if not member_path.is_relative_to(dest):
                    raise ValueError(
                        f"Path traversal detected in archive: {member.name}"
                    )
                if member.issym() or member.islnk():
                    raise ValueError(
                        f"Symlink not allowed in archive: {member.name}"
                    )
            tar.extractall(self.backup_dir)

        # Rebuild metadata
        self._rebuild_metadata()

        print(f"Backup imported: {bundle_file}")

    def _rebuild_metadata(self) -> None:
        """Rebuild metadata.json from existing backups."""
        self.metadata = {}

        for backup_dir in self.backup_dir.iterdir():
            if backup_dir.is_dir() and backup_dir.name != "metadata.json":
                proof_file = backup_dir / "proof.json"
                if proof_file.exists():
                    with open(proof_file, 'r') as f:
                        proof = json.load(f)

                    self.metadata[backup_dir.name] = {
                        "timestamp": proof['timestamp'],
                        "label": None,
                        "password_hash": proof['password_hash'],
                        "backup_path": str(backup_dir),
                        "created_at": proof['timestamp']
                    }

        self._save_metadata()
        print(f"Metadata rebuilt: {len(self.metadata)} backups")

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self) -> None:
        """Save metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def delete_backup(self, backup_id: str) -> bool:
        """
        Delete a backup.

        Args:
            backup_id: Backup identifier

        Returns:
            True if deleted, False if not found
        """
        backup_path = self.backup_dir / backup_id

        if not backup_path.exists():
            return False

        # Remove files
        import shutil
        shutil.rmtree(backup_path)

        # Update metadata
        if backup_id in self.metadata:
            del self.metadata[backup_id]
            self._save_metadata()

        print(f"Backup deleted: {backup_id}")
        return True
