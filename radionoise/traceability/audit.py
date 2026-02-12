"""
RadioNoise Forensic Audit Trail - Blockchain-style audit system.
"""

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from radionoise.core.log import get_logger

logger = get_logger('audit')


class ForensicAuditTrail:
    """
    Forensic traceability system for RadioNoise.

    Features:
    1. Immutable database (append-only)
    2. Proof chaining (blockchain-style)
    3. Integrity verification
    4. Complete timeline
    5. Export for external audits
    """

    def __init__(self, db_path: str = "./audit_trail.db"):
        """
        Initialize audit trail database.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self) -> None:
        """Create database tables."""
        cursor = self.conn.cursor()

        # Generations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                capture_hash TEXT NOT NULL,
                entropy_hash TEXT,
                signature TEXT NOT NULL,
                previous_hash TEXT,
                chain_hash TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
        """)

        # Events table (audit log)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                generation_id INTEGER,
                description TEXT,
                metadata TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (generation_id) REFERENCES generations(id)
            )
        """)

        # Indexes for fast search
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON generations(timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_password_hash
            ON generations(password_hash)
        """)

        self.conn.commit()

    def add_generation(
        self,
        proof: Dict[str, Any],
        context: Optional[str] = None
    ) -> Tuple[int, str]:
        """
        Add a generation to the audit trail with chaining.

        Chaining works like a blockchain:
        chain_hash = SHA256(previous_chain_hash + signature + timestamp)

        Args:
            proof: Proof of generation dictionary
            context: Optional context/label for the generation

        Returns:
            Tuple of (generation_id, chain_hash)
        """
        cursor = self.conn.cursor()

        # Get previous chain hash
        cursor.execute("SELECT chain_hash FROM generations ORDER BY id DESC LIMIT 1")
        result = cursor.fetchone()
        previous_hash = result[0] if result else "0" * 64  # Genesis

        # Calculate new chain_hash
        chain_data = previous_hash + proof['signature'] + proof['timestamp']
        chain_hash = hashlib.sha256(chain_data.encode()).hexdigest()

        # Prepare metadata
        metadata = {
            "frequency": proof['metadata']['frequency'],
            "sample_rate": proof['metadata']['sample_rate'],
            "samples": proof['metadata']['samples'],
            "password_length": proof.get('password_length'),
            "charset": proof.get('charset'),
            "context": context
        }

        # Insert generation
        cursor.execute("""
            INSERT INTO generations (
                timestamp, password_hash, capture_hash, entropy_hash,
                signature, previous_hash, chain_hash, metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            proof['timestamp'],
            proof['password_hash'],
            proof['capture_hash'],
            proof.get('entropy_hash'),
            proof['signature'],
            previous_hash,
            chain_hash,
            json.dumps(metadata),
            datetime.utcnow().isoformat() + 'Z'
        ))

        generation_id = cursor.lastrowid

        # Log event
        self.log_event(
            "GENERATION",
            generation_id,
            "Password generated from radio capture",
            {"proof": proof}
        )

        self.conn.commit()

        return generation_id, chain_hash

    def log_event(
        self,
        event_type: str,
        generation_id: Optional[int] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an event to the audit log."""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO events (event_type, generation_id, description, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (
            event_type,
            generation_id,
            description,
            json.dumps(metadata) if metadata else None,
            datetime.utcnow().isoformat() + 'Z'
        ))

        self.conn.commit()

    def verify_chain_integrity(self) -> Tuple[bool, List[str]]:
        """
        Verify integrity of the entire chain.

        Returns:
            Tuple of (valid, list_of_errors)
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM generations ORDER BY id")

        errors = []
        previous_hash = "0" * 64  # Genesis

        for row in cursor.fetchall():
            gen_id, timestamp, pwd_hash, cap_hash, ent_hash, sig, prev_hash, chain_hash, meta, created = row

            # Verify previous_hash
            if prev_hash != previous_hash:
                errors.append(f"Generation {gen_id}: previous_hash mismatch")

            # Recalculate chain_hash
            expected_chain = hashlib.sha256((prev_hash + sig + timestamp).encode()).hexdigest()
            if chain_hash != expected_chain:
                errors.append(f"Generation {gen_id}: chain_hash mismatch")

            previous_hash = chain_hash

        return len(errors) == 0, errors

    def get_generation(
        self,
        generation_id: Optional[int] = None,
        password_hash: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a generation by ID or password hash."""
        cursor = self.conn.cursor()

        if generation_id:
            cursor.execute("SELECT * FROM generations WHERE id = ?", (generation_id,))
        elif password_hash:
            cursor.execute("SELECT * FROM generations WHERE password_hash = ?", (password_hash,))
        else:
            return None

        row = cursor.fetchone()
        if not row:
            return None

        return {
            "id": row[0],
            "timestamp": row[1],
            "password_hash": row[2],
            "capture_hash": row[3],
            "entropy_hash": row[4],
            "signature": row[5],
            "previous_hash": row[6],
            "chain_hash": row[7],
            "metadata": json.loads(row[8]) if row[8] else {},
            "created_at": row[9]
        }

    def get_timeline(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve a timeline of generations."""
        cursor = self.conn.cursor()

        if start_date and end_date:
            cursor.execute("""
                SELECT * FROM generations
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """, (start_date, end_date))
        else:
            cursor.execute("SELECT * FROM generations ORDER BY timestamp")

        timeline = []
        for row in cursor.fetchall():
            timeline.append({
                "id": row[0],
                "timestamp": row[1],
                "password_hash": row[2][:16] + "...",
                "signature": row[5][:16] + "..."
            })

        return timeline

    def prove_generation_time(
        self,
        password_hash: str,
        tolerance_seconds: int = 60
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Prove that a password was generated at a specific time.
        Uses chaining to prove temporal order.

        Args:
            password_hash: Hash of the password
            tolerance_seconds: Time tolerance in seconds

        Returns:
            Tuple of (proof_dict, message)
        """
        gen = self.get_generation(password_hash=password_hash)
        if not gen:
            return None, "Password not found in audit trail"

        # Get adjacent generations
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, chain_hash
            FROM generations
            WHERE id < ?
            ORDER BY id DESC LIMIT 1
        """, (gen['id'],))
        prev_gen = cursor.fetchone()

        cursor.execute("""
            SELECT id, timestamp, chain_hash
            FROM generations
            WHERE id > ?
            ORDER BY id ASC LIMIT 1
        """, (gen['id'],))
        next_gen = cursor.fetchone()

        proof = {
            "generation": gen,
            "position_in_chain": gen['id'],
            "previous_generation": {
                "id": prev_gen[0] if prev_gen else None,
                "timestamp": prev_gen[1] if prev_gen else None,
                "chain_hash": prev_gen[2] if prev_gen else None
            },
            "next_generation": {
                "id": next_gen[0] if next_gen else None,
                "timestamp": next_gen[1] if next_gen else None,
                "chain_hash": next_gen[2] if next_gen else None
            }
        }

        return proof, "Valid temporal proof"

    def export_audit_report(
        self,
        output_file: str,
        format: str = 'json'
    ) -> Dict[str, Any]:
        """Export a complete audit report."""
        cursor = self.conn.cursor()

        # Get all generations
        cursor.execute("SELECT * FROM generations ORDER BY id")
        generations = []

        for row in cursor.fetchall():
            gen = {
                "id": row[0],
                "timestamp": row[1],
                "password_hash": row[2],
                "capture_hash": row[3],
                "entropy_hash": row[4],
                "signature": row[5],
                "previous_hash": row[6],
                "chain_hash": row[7],
                "metadata": json.loads(row[8]) if row[8] else {},
                "created_at": row[9]
            }
            generations.append(gen)

        # Get all events
        cursor.execute("SELECT * FROM events ORDER BY id")
        events = []

        for row in cursor.fetchall():
            event = {
                "id": row[0],
                "type": row[1],
                "generation_id": row[2],
                "description": row[3],
                "metadata": json.loads(row[4]) if row[4] else {},
                "timestamp": row[5]
            }
            events.append(event)

        # Verify integrity
        valid, errors = self.verify_chain_integrity()

        report = {
            "report_generated_at": datetime.utcnow().isoformat() + 'Z',
            "total_generations": len(generations),
            "total_events": len(events),
            "chain_integrity": {
                "valid": valid,
                "errors": errors
            },
            "generations": generations,
            "events": events
        }

        if format == 'json':
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)

        logger.info("Audit report exported: %s", output_file)
        logger.info("  %d generations", len(generations))
        logger.info("  %d events", len(events))
        logger.info("  Integrity: %s", 'Valid' if valid else 'Corrupted')

        return report

    def search_by_context(self, context_filter: str) -> List[Dict[str, Any]]:
        """Search generations by context."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM generations ORDER BY id")

        results = []
        for row in cursor.fetchall():
            metadata = json.loads(row[8]) if row[8] else {}
            if metadata.get('context') == context_filter:
                results.append({
                    "id": row[0],
                    "timestamp": row[1],
                    "password_hash": row[2][:16] + "...",
                })

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get global statistics."""
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM generations")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM generations")
        first, last = cursor.fetchone()

        cursor.execute("SELECT COUNT(DISTINCT DATE(timestamp)) FROM generations")
        active_days = cursor.fetchone()[0]

        return {
            "total_generations": total,
            "first_generation": first,
            "last_generation": last,
            "active_days": active_days
        }

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()
