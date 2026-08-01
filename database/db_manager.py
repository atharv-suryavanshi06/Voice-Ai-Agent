"""
db_manager.py

Manages PostgreSQL connection and JSONB operations for persisting customer profiles.
Performs automatic table creation, GIN index initialization, and JSONB upserts.
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from core import config

logger = logging.getLogger(__name__)

try:
    import psycopg2
    from psycopg2.extras import Json
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    logger.warning("psycopg2 package not installed. Install via `pip install psycopg2-binary`.")


class PostgresDBManager:
    """Handles PostgreSQL connection and JSONB customer profile persistence."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        dbname: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database_url: Optional[str] = None,
    ) -> None:
        self.host = host or config.POSTGRES_HOST
        self.port = port or config.POSTGRES_PORT
        self.dbname = dbname or config.POSTGRES_DB
        self.user = user or config.POSTGRES_USER
        self.password = password or config.POSTGRES_PASSWORD
        self.database_url = database_url or config.DATABASE_URL
        self.enabled = False

        if not PSYCOPG2_AVAILABLE:
            print("[Database Warning] psycopg2-binary not available. Database storage disabled.")
            return

        # Attempt initial database connection and table setup
        self.init_db()

    def _get_connection(self):
        """Creates and returns a new psycopg2 connection."""
        if self.database_url:
            return psycopg2.connect(self.database_url)
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password,
        )

    def _ensure_database_exists(self) -> None:
        """Connects to default 'postgres' database and auto-creates target database if it does not exist."""
        if self.database_url:
            return
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname="postgres",
                user=self.user,
                password=self.password,
            )
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (self.dbname,))
                if not cur.fetchone():
                    safe_dbname = psycopg2.extensions.quote_ident(self.dbname, conn)
                    cur.execute(f"CREATE DATABASE {safe_dbname};")
                    print(f"[Database] Automatically created target database '{self.dbname}'.")
            conn.close()
        except Exception as e:
            print(f"[Database Warning] Auto-database creation check failed: {e}")

    def init_db(self) -> bool:
        """
        Creates the database, `customer_profiles`, `policy_documents`, and `sent_email_logs` tables if missing.
        Returns True if successful, False otherwise.
        """
        if not PSYCOPG2_AVAILABLE:
            return False

        # Step 1: Ensure the database exists on the PostgreSQL server
        self._ensure_database_exists()

        # Step 2: Connect to target database and create tables & indexes
        try:
            conn = self._get_connection()
            with conn:
                with conn.cursor() as cur:
                    # 1. Customer Profiles Table (JSONB)
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS customer_profiles (
                            session_id VARCHAR(255) PRIMARY KEY,
                            profile_data JSONB NOT NULL,
                            conversation_history JSONB NOT NULL DEFAULT '[]'::jsonb,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        );
                        """
                    )
                    # Migrate databases created before conversation persistence
                    # was added. Existing profile rows retain an empty history.
                    cur.execute(
                        """
                        ALTER TABLE customer_profiles
                        ADD COLUMN IF NOT EXISTS conversation_history JSONB NOT NULL DEFAULT '[]'::jsonb;
                        """
                    )
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_customer_profile_jsonb 
                        ON customer_profiles USING gin (profile_data);
                        """
                    )

                    # Append-only conversation storage. The legacy JSONB column
                    # remains intact for backward compatibility and migration.
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS conversation_sessions (
                            session_id VARCHAR(255) PRIMARY KEY,
                            customer_id VARCHAR(255),
                            started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            ended_at TIMESTAMP WITH TIME ZONE,
                            status VARCHAR(50) NOT NULL DEFAULT 'active',
                            session_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                        );
                        """
                    )
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS conversation_messages (
                            message_id VARCHAR(255) PRIMARY KEY,
                            session_id VARCHAR(255) NOT NULL REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,
                            sequence INTEGER NOT NULL,
                            role VARCHAR(32) NOT NULL,
                            content TEXT NOT NULL,
                            message_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(session_id, sequence)
                        );
                        """
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_conversation_messages_session_sequence "
                        "ON conversation_messages(session_id, sequence);"
                    )

                    # Non-destructive, idempotent migration of existing JSONB histories.
                    cur.execute(
                        """
                        SELECT cp.session_id, cp.conversation_history, cp.created_at, cp.updated_at
                        FROM customer_profiles cp
                        WHERE jsonb_array_length(cp.conversation_history) > 0
                          AND NOT EXISTS (
                              SELECT 1 FROM conversation_messages cm WHERE cm.session_id = cp.session_id
                          );
                        """
                    )
                    legacy_rows = cur.fetchall()
                    for legacy_session_id, legacy_history, created_at, updated_at in legacy_rows:
                        cur.execute(
                            """
                            INSERT INTO conversation_sessions (session_id, started_at, ended_at, status)
                            VALUES (%s, %s, %s, 'completed')
                            ON CONFLICT (session_id) DO NOTHING;
                            """,
                            (legacy_session_id, created_at, updated_at),
                        )
                        for index, message in enumerate(legacy_history or [], 1):
                            if not isinstance(message, dict):
                                continue
                            message_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{legacy_session_id}:{index}"))
                            cur.execute(
                                """
                                INSERT INTO conversation_messages
                                    (message_id, session_id, sequence, role, content, message_metadata)
                                VALUES (%s, %s, %s, %s, %s, %s)
                                ON CONFLICT DO NOTHING;
                                """,
                                (
                                    message_id,
                                    legacy_session_id,
                                    index,
                                    str(message.get("role", "unknown")),
                                    str(message.get("content", "")),
                                    Json({"migrated_from": "customer_profiles.conversation_history"}),
                                ),
                            )

                    # 2. Policy Documents Table (Full text, metadata, and raw PDF binary BYTEA)
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS policy_documents (
                            policy_id VARCHAR(255) PRIMARY KEY,
                            policy_name VARCHAR(255) NOT NULL,
                            insurer VARCHAR(255),
                            plan_type VARCHAR(100),
                            premium NUMERIC(12, 2),
                            sum_insured NUMERIC(14, 2),
                            document_text TEXT,
                            pdf_path VARCHAR(255),
                            pdf_data BYTEA,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        );
                        """
                    )
                    # Migration check to ensure pdf_data column exists on existing installations
                    cur.execute("ALTER TABLE policy_documents ADD COLUMN IF NOT EXISTS pdf_data BYTEA;")

                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS policy_ingestions (
                            ingestion_id VARCHAR(255) PRIMARY KEY,
                            policy_id VARCHAR(255) NOT NULL,
                            status VARCHAR(50) NOT NULL,
                            metadata JSONB NOT NULL,
                            document_text TEXT,
                            pdf_path VARCHAR(255),
                            pdf_data BYTEA,
                            error_context TEXT,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        );
                        """
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_policy_ingestions_policy_status "
                        "ON policy_ingestions(policy_id, status);"
                    )


                    # 3. Sent Email Logs Table
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS sent_email_logs (
                            id SERIAL PRIMARY KEY,
                            session_id VARCHAR(255),
                            recipient_email VARCHAR(255) NOT NULL,
                            policy_id VARCHAR(255),
                            policy_name VARCHAR(255),
                            status VARCHAR(50) NOT NULL,
                            error_message TEXT,
                            sent_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        );
                        """
                    )
            conn.close()
            self.enabled = True
            print(f"[Database] Connected to PostgreSQL '{self.dbname}' (tables: customer_profiles, policy_documents, sent_email_logs ready).")
            
            # Sync pre-existing policy documents into PostgreSQL
            self.sync_existing_markdown_documents()
            return True
        except Exception as e:
            print(f"[Database Warning] Could not connect to PostgreSQL: {e}")
            print("[Database Warning] Customer profiles will operate in-memory.")
            self.enabled = False
            return False

    def save_profile(
        self,
        session_id: str,
        profile_dict: Dict[str, Any],
        conversation_history: Optional[list] = None,
    ) -> bool:
        """Compatibility API that now appends messages instead of rewriting JSONB."""
        messages = []
        for index, message in enumerate(conversation_history or [], 1):
            if not isinstance(message, dict):
                continue
            messages.append({
                "message_id": message.get("message_id") or f"{session_id}:{index}",
                "sequence": int(message.get("sequence") or index),
                "role": str(message.get("role", "unknown")),
                "content": str(message.get("content", "")),
                "metadata": message.get("metadata") or {},
            })
        return self.persist_conversation_update(session_id, profile_dict, messages)

    def persist_conversation_update(
        self,
        session_id: str,
        profile_dict: Dict[str, Any],
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Atomically upsert a profile and append idempotent ordered messages."""
        if not self.enabled:
            return False
        try:
            conn = self._get_connection()
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO customer_profiles (session_id, profile_data, updated_at)
                        VALUES (%s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (session_id) DO UPDATE SET
                            profile_data = EXCLUDED.profile_data,
                            updated_at = CURRENT_TIMESTAMP;
                        """,
                        (session_id, Json(profile_dict)),
                    )
                    cur.execute(
                        """
                        INSERT INTO conversation_sessions (session_id, status)
                        VALUES (%s, 'active')
                        ON CONFLICT (session_id) DO NOTHING;
                        """,
                        (session_id,),
                    )
                    for message in messages or []:
                        cur.execute(
                            """
                            INSERT INTO conversation_messages
                                (message_id, session_id, sequence, role, content, message_metadata)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT DO NOTHING;
                            """,
                            (
                                message["message_id"],
                                session_id,
                                int(message["sequence"]),
                                str(message["role"]),
                                str(message["content"]),
                                Json(message.get("metadata") or {}),
                            ),
                        )
            conn.close()
            return True
        except Exception as e:
            logger.error(
                "Failed to persist conversation update",
                extra={
                    "session_id": session_id,
                    "message_count": len(messages or []),
                    "error_type": type(e).__name__,
                },
            )
            return False

    def get_conversation_history(self, session_id: str) -> Optional[list]:
        """Retrieve the ordered conversation messages for a call session."""
        if not self.enabled:
            return None

        try:
            conn = self._get_connection()
            history = None
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT role, content, message_id, sequence, created_at, message_metadata
                        FROM conversation_messages
                        WHERE session_id = %s
                        ORDER BY sequence ASC, created_at ASC;
                        """,
                        (session_id,),
                    )
                    rows = cur.fetchall()
                    if rows:
                        history = [
                            {
                                "role": row[0],
                                "content": row[1],
                                "message_id": row[2],
                                "sequence": row[3],
                                "created_at": row[4].isoformat() if row[4] else None,
                                "metadata": row[5] or {},
                            }
                            for row in rows
                        ]
                    else:
                        cur.execute(
                            "SELECT conversation_history FROM customer_profiles WHERE session_id = %s;",
                            (session_id,),
                        )
                        row = cur.fetchone()
                        if row:
                            history = row[0] or []
            conn.close()
            return history
        except Exception as e:
            print(f"[Database Error] Failed to retrieve conversation for session '{session_id}': {e}")
            return None

    def end_conversation_session(self, session_id: str, status: str = "completed") -> bool:
        if not self.enabled:
            return False
        try:
            conn = self._get_connection()
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE conversation_sessions
                        SET ended_at = CURRENT_TIMESTAMP, status = %s
                        WHERE session_id = %s;
                        """,
                        (status, session_id),
                    )
            conn.close()
            return True
        except Exception:
            logger.exception("Failed to close conversation session", extra={"session_id": session_id})
            return False

    def close(self) -> None:
        """Compatibility hook; connections are currently opened per operation."""
        return None

    def get_profile(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the JSONB customer profile dictionary for a given session ID.
        """
        if not self.enabled:
            return None

        try:
            conn = self._get_connection()
            profile_data = None
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT profile_data FROM customer_profiles WHERE session_id = %s;",
                        (session_id,),
                    )
                    row = cur.fetchone()
                    if row:
                        profile_data = row[0]
            conn.close()
            return profile_data
        except Exception as e:
            print(f"[Database Error] Failed to retrieve profile for session '{session_id}': {e}")
            return None

    def save_policy_document(
        self,
        policy_id: str,
        policy_name: str,
        document_text: str,
        insurer: Optional[str] = None,
        plan_type: Optional[str] = None,
        premium: Optional[float] = None,
        sum_insured: Optional[float] = None,
        pdf_path: Optional[str] = None,
        pdf_bytes: Optional[bytes] = None,
    ) -> bool:
        """
        Stores or updates a complete policy document text, raw PDF binary bytes (BYTEA), and metadata in PostgreSQL `policy_documents`.
        """
        if not self.enabled:
            return False

        binary_data = psycopg2.Binary(pdf_bytes) if pdf_bytes else None
        try:
            conn = self._get_connection()
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO policy_documents
                        (policy_id, policy_name, insurer, plan_type, premium, sum_insured, document_text, pdf_path, pdf_data, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (policy_id) DO UPDATE SET
                            policy_name = EXCLUDED.policy_name,
                            insurer = EXCLUDED.insurer,
                            plan_type = EXCLUDED.plan_type,
                            premium = EXCLUDED.premium,
                            sum_insured = EXCLUDED.sum_insured,
                            document_text = EXCLUDED.document_text,
                            pdf_path = EXCLUDED.pdf_path,
                            pdf_data = COALESCE(EXCLUDED.pdf_data, policy_documents.pdf_data),
                            updated_at = CURRENT_TIMESTAMP;
                        """,
                        (policy_id, policy_name, insurer, plan_type, premium, sum_insured, document_text, pdf_path, binary_data),
                    )
            conn.close()
            return True
        except Exception as e:
            print(f"[Database Error] Failed to save policy document '{policy_id}': {e}")
            return False

    def stage_policy_document(
        self,
        ingestion_id: str,
        policy_id: str,
        metadata: Dict[str, Any],
        document_text: str,
        pdf_path: Optional[str] = None,
        pdf_bytes: Optional[bytes] = None,
    ) -> bool:
        """Store a retry-safe policy version without exposing it as active."""
        if not self.enabled:
            return False
        binary_data = psycopg2.Binary(pdf_bytes) if pdf_bytes else None
        try:
            conn = self._get_connection()
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO policy_ingestions
                            (ingestion_id, policy_id, status, metadata, document_text, pdf_path, pdf_data, error_context, updated_at)
                        VALUES (%s, %s, 'indexed', %s, %s, %s, %s, NULL, CURRENT_TIMESTAMP)
                        ON CONFLICT (ingestion_id) DO UPDATE SET
                            policy_id = EXCLUDED.policy_id,
                            status = 'indexed',
                            metadata = EXCLUDED.metadata,
                            document_text = EXCLUDED.document_text,
                            pdf_path = EXCLUDED.pdf_path,
                            pdf_data = COALESCE(EXCLUDED.pdf_data, policy_ingestions.pdf_data),
                            error_context = NULL,
                            updated_at = CURRENT_TIMESTAMP;
                        """,
                        (ingestion_id, policy_id, Json(metadata), document_text, pdf_path, binary_data),
                    )
            conn.close()
            return True
        except Exception:
            logger.exception(
                "Failed to stage policy document",
                extra={"ingestion_id": ingestion_id, "policy_id": policy_id},
            )
            return False

    def activate_staged_policy_document(self, ingestion_id: str) -> bool:
        """Publish a staged PostgreSQL policy version in one transaction."""
        if not self.enabled:
            return False
        try:
            conn = self._get_connection()
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT policy_id, metadata, document_text, pdf_path, pdf_data
                        FROM policy_ingestions
                        WHERE ingestion_id = %s AND status = 'indexed'
                        FOR UPDATE;
                        """,
                        (ingestion_id,),
                    )
                    row = cur.fetchone()
                    if not row:
                        raise RuntimeError("Staged policy ingestion was not found")
                    policy_id, metadata, document_text, pdf_path, pdf_data = row
                    cur.execute(
                        """
                        INSERT INTO policy_documents
                            (policy_id, policy_name, insurer, plan_type, premium, sum_insured, document_text, pdf_path, pdf_data, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (policy_id) DO UPDATE SET
                            policy_name = EXCLUDED.policy_name,
                            insurer = EXCLUDED.insurer,
                            plan_type = EXCLUDED.plan_type,
                            premium = EXCLUDED.premium,
                            sum_insured = EXCLUDED.sum_insured,
                            document_text = EXCLUDED.document_text,
                            pdf_path = EXCLUDED.pdf_path,
                            pdf_data = COALESCE(EXCLUDED.pdf_data, policy_documents.pdf_data),
                            updated_at = CURRENT_TIMESTAMP;
                        """,
                        (
                            policy_id,
                            metadata.get("policy_name"),
                            metadata.get("insurer"),
                            metadata.get("plan_type"),
                            metadata.get("premium"),
                            metadata.get("sum_insured"),
                            document_text,
                            pdf_path,
                            pdf_data,
                        ),
                    )
                    cur.execute(
                        """
                        UPDATE policy_ingestions
                        SET status = 'active', error_context = NULL, updated_at = CURRENT_TIMESTAMP
                        WHERE ingestion_id = %s;
                        """,
                        (ingestion_id,),
                    )
            conn.close()
            return True
        except Exception:
            logger.exception("Failed to activate policy ingestion", extra={"ingestion_id": ingestion_id})
            return False

    def mark_policy_ingestion_failed(self, ingestion_id: str, error_context: str) -> bool:
        if not self.enabled:
            return False
        try:
            conn = self._get_connection()
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE policy_ingestions
                        SET status = 'failed', error_context = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE ingestion_id = %s;
                        """,
                        (error_context[:2000], ingestion_id),
                    )
            conn.close()
            return True
        except Exception:
            logger.exception("Failed to mark policy ingestion failed", extra={"ingestion_id": ingestion_id})
            return False

    def get_policy_document(self, policy_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves complete policy document text, raw PDF bytes (BYTEA), and metadata from PostgreSQL `policy_documents`.
        """
        if not self.enabled:
            return None

        try:
            conn = self._get_connection()
            result = None
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT policy_id, policy_name, insurer, plan_type, premium, sum_insured, document_text, pdf_path, pdf_data
                        FROM policy_documents
                        WHERE policy_id = %s OR LOWER(policy_name) LIKE %s;
                        """,
                        (policy_id, f"%{policy_id.lower()}%"),
                    )
                    row = cur.fetchone()
                    if row:
                        result = {
                            "policy_id": row[0],
                            "policy_name": row[1],
                            "insurer": row[2],
                            "plan_type": row[3],
                            "premium": float(row[4]) if row[4] is not None else 0.0,
                            "sum_insured": float(row[5]) if row[5] is not None else 0.0,
                            "document_text": row[6],
                            "pdf_path": row[7],
                            "pdf_data": bytes(row[8]) if row[8] is not None else None,
                        }
            conn.close()
            return result
        except Exception as e:
            print(f"[Database Error] Failed to retrieve policy document '{policy_id}': {e}")
            return None

    def delete_policy_document(self, policy_id: str) -> bool:
        """Compensating delete used only when a brand-new activation rolls back."""
        if not self.enabled:
            return False
        try:
            conn = self._get_connection()
            with conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM policy_documents WHERE policy_id = %s;", (policy_id,))
            conn.close()
            return True
        except Exception:
            logger.exception("Failed to roll back policy document", extra={"policy_id": policy_id})
            return False

    def log_sent_email(
        self,
        session_id: str,
        recipient_email: str,
        policy_id: str,
        policy_name: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Logs sent email status to PostgreSQL `sent_email_logs`.
        """
        if not self.enabled:
            return False

        try:
            conn = self._get_connection()
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO sent_email_logs (session_id, recipient_email, policy_id, policy_name, status, error_message)
                        VALUES (%s, %s, %s, %s, %s, %s);
                        """,
                        (session_id, recipient_email, policy_id, policy_name, status, error_message),
                    )
            conn.close()
            return True
        except Exception as e:
            print(f"[Database Error] Failed to log sent email: {e}")
            return False

    def sync_existing_markdown_documents(self) -> None:
        """
        Scans Data/*.md files and policy catalog to populate policy_documents in PostgreSQL.
        """
        import os
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        data_dir = os.path.join(base_dir, "Data")
        catalog_path = os.path.join(base_dir, "recommendation", "policy_catalog.json")

        if not os.path.exists(catalog_path):
            return

        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                catalog = json.load(f)
        except Exception:
            return

        for entry in catalog:
            if entry.get("_ingestion_status", "active") != "active":
                continue
            p_id = entry.get("policy_id")
            p_name = entry.get("policy_name")
            if not p_id or not p_name:
                continue

            # Check if document and raw PDF binary are already present
            existing = self.get_policy_document(p_id)
            if existing and existing.get("document_text") and existing.get("pdf_data"):
                continue


            # Look for matching .md file in Data directory
            clean_name = p_name.replace(" ", "_")
            md_filename = f"{clean_name}_Policy_Document.md"
            md_path = os.path.join(data_dir, md_filename)

            text_content = ""
            if os.path.exists(md_path):
                try:
                    with open(md_path, "r", encoding="utf-8") as f:
                        text_content = f.read()
                except Exception:
                    text_content = ""

            if not text_content:
                text_content = (
                    f"Policy Document: {p_name}\n"
                    f"Insurer: {entry.get('insurer')}\n"
                    f"Plan Type: {entry.get('plan_type')}\n"
                    f"Premium: ₹{entry.get('premium', 0):,.2f}/year\n"
                    f"Sum Insured: ₹{entry.get('sum_insured', 0):,.2f}\n\n"
                    f"Coverage Details:\n"
                    f"- Smoker Allowed: {entry.get('smoker_allowed')}\n"
                    f"- Diabetes Covered: {entry.get('covers_diabetes')}\n"
                    f"- Hypertension Covered: {entry.get('covers_hypertension')}\n"
                )

            pdf_filename = f"{clean_name}_Policy_Document.pdf"
            pdf_path = os.path.join(data_dir, pdf_filename)
            if not os.path.exists(pdf_path):
                # Search for alternative matching PDF in Data directory
                for f in os.listdir(data_dir):
                    if f.endswith(".pdf"):
                        # Check partial keyword match (e.g. Individual_Health_Shield)
                        key_part = clean_name.replace("SecureLife_", "").replace("ApexCare_", "").replace("TrustShield_", "")
                        if key_part in f:
                            pdf_path = os.path.join(data_dir, f)
                            break
                        
            pdf_bytes = None
            if pdf_path and os.path.exists(pdf_path):
                try:
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                except Exception:
                    pdf_bytes = None
            else:
                pdf_path = None


            self.save_policy_document(
                policy_id=p_id,
                policy_name=p_name,
                document_text=text_content,
                insurer=entry.get("insurer"),
                plan_type=entry.get("plan_type"),
                premium=entry.get("premium"),
                sum_insured=entry.get("sum_insured"),
                pdf_path=pdf_path,
                pdf_bytes=pdf_bytes,
            )
