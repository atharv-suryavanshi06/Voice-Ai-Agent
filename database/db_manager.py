"""
db_manager.py

Manages PostgreSQL connection and JSONB operations for persisting customer profiles.
Performs automatic table creation, GIN index initialization, and JSONB upserts.
"""

import hashlib
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from core import config

logger = logging.getLogger(__name__)

_EXACT_ID_SYNC_VERSION = "v1"


def _exact_id_sync_backup_id(policy_id: str) -> str:
    digest = hashlib.sha256(policy_id.encode("utf-8")).hexdigest()
    return f"exact-id-markdown-sync-{_EXACT_ID_SYNC_VERSION}:{digest}"


def _optional_bytes(value: Any) -> Optional[bytes]:
    return bytes(value) if value is not None else None


def _paths_equal(left: Optional[str], right: Optional[str]) -> bool:
    if not left or not right:
        return left == right
    import os
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))


def _document_matches_source(
    existing: Optional[Dict[str, Any]],
    entry: Dict[str, Any],
    document_text: str,
    pdf_path: Optional[str],
    pdf_bytes: Optional[bytes],
) -> bool:
    """Return True when an exact-source sync would be a no-op."""
    if not existing:
        return False

    numeric_fields = ("premium", "sum_insured")
    text_fields = ("policy_id", "policy_name", "insurer", "plan_type")
    if any(str(existing.get(field) or "") != str(entry.get(field) or "") for field in text_fields):
        return False
    for field in numeric_fields:
        if float(existing.get(field) or 0.0) != float(entry.get(field) or 0.0):
            return False
    return (
        (existing.get("document_text") or "") == document_text
        and _paths_equal(existing.get("pdf_path"), pdf_path)
        and _optional_bytes(existing.get("pdf_data")) == _optional_bytes(pdf_bytes)
    )

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
        self._verified_policy_document_ids: set[str] = set()

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
            if not self.sync_existing_markdown_documents():
                logger.error(
                    "Policy document synchronization was incomplete; unverified email document content and attachments will fail closed"
                )
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

    def is_policy_document_verified(self, policy_id: str) -> bool:
        """Whether exact-ID synchronization/ingestion verified this row this run."""
        return str(policy_id) in getattr(self, "_verified_policy_document_ids", set())

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
        replace_pdf: bool = False,
        backup_id: Optional[str] = None,
        backup_reason: Optional[str] = None,
    ) -> bool:
        """
        Stores or updates a complete policy document text, raw PDF binary bytes (BYTEA), and metadata in PostgreSQL `policy_documents`.

        ``replace_pdf`` is intentionally opt-in for backward compatibility. It
        lets an exact-source repair clear a previously misassigned attachment
        when the verified source has no PDF. When ``backup_id`` is provided,
        the previous row is copied to ``policy_ingestions`` in the same
        transaction before it is replaced. A stable backup ID makes retries
        idempotent without adding another database table.
        """
        if not self.enabled:
            return False

        binary_data = psycopg2.Binary(pdf_bytes) if pdf_bytes else None
        try:
            conn = self._get_connection()
            with conn:
                with conn.cursor() as cur:
                    if backup_id:
                        cur.execute(
                            """
                            INSERT INTO policy_ingestions
                                (ingestion_id, policy_id, status, metadata, document_text,
                                 pdf_path, pdf_data, error_context, updated_at)
                            SELECT
                                %s,
                                policy_id,
                                'backup',
                                jsonb_build_object(
                                    'policy_name', policy_name,
                                    'insurer', insurer,
                                    'plan_type', plan_type,
                                    'premium', premium,
                                    'sum_insured', sum_insured,
                                    'backup_reason', %s
                                ),
                                document_text,
                                pdf_path,
                                pdf_data,
                                %s,
                                CURRENT_TIMESTAMP
                            FROM policy_documents
                            WHERE policy_id = %s
                            ON CONFLICT (ingestion_id) DO NOTHING;
                            """,
                            (
                                backup_id,
                                backup_reason or "policy document replacement",
                                backup_reason,
                                policy_id,
                            ),
                        )
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
                            pdf_data = CASE
                                WHEN %s THEN EXCLUDED.pdf_data
                                ELSE COALESCE(EXCLUDED.pdf_data, policy_documents.pdf_data)
                            END,
                            updated_at = CURRENT_TIMESTAMP;
                        """,
                        (
                            policy_id,
                            policy_name,
                            insurer,
                            plan_type,
                            premium,
                            sum_insured,
                            document_text,
                            pdf_path,
                            binary_data,
                            replace_pdf,
                        ),
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
            verified_ids = getattr(self, "_verified_policy_document_ids", set())
            verified_ids.add(str(policy_id))
            self._verified_policy_document_ids = verified_ids
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

    def _get_verified_ingestion_pdf(self, policy_id: str) -> Optional[Dict[str, Any]]:
        """Return the newest attachment recorded by an exact-ID activation.

        Startup Markdown synchronization must not infer attachment ownership
        from a similar product name. An active ingestion is safe provenance
        because its extracted metadata and stored document use the same exact
        policy ID.
        """
        if not self.enabled:
            return None

        conn = self._get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT pdf_path, pdf_data
                        FROM policy_ingestions
                        WHERE policy_id = %s
                          AND status = 'active'
                          AND (pdf_path IS NOT NULL OR pdf_data IS NOT NULL)
                        ORDER BY updated_at DESC, created_at DESC
                        LIMIT 1;
                        """,
                        (policy_id,),
                    )
                    row = cur.fetchone()
            if not row:
                return None
            return {
                "pdf_path": row[0],
                "pdf_data": _optional_bytes(row[1]),
            }
        finally:
            conn.close()

    def _fetch_policy_document_exact(self, policy_id: str) -> Optional[Dict[str, Any]]:
        """Fetch an exact-ID row, allowing synchronization errors to propagate."""
        conn = self._get_connection()
        try:
            result = None
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT policy_id, policy_name, insurer, plan_type, premium, sum_insured, document_text, pdf_path, pdf_data
                        FROM policy_documents
                        WHERE policy_id = %s
                        LIMIT 1;
                        """,
                        (policy_id,),
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
            return result
        finally:
            conn.close()

    def get_policy_document(self, policy_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves complete policy document text, raw PDF bytes (BYTEA), and metadata from PostgreSQL `policy_documents`.

        ``policy_id`` is canonical. Policy names are deliberately not accepted
        as a fallback because marketing names are not guaranteed to be unique.
        """
        if not self.enabled:
            return None

        try:
            return self._fetch_policy_document_exact(policy_id)
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

    def sync_existing_markdown_documents(
        self,
        data_dir: Optional[str] = None,
        catalog_path: Optional[str] = None,
    ) -> bool:
        """Synchronize canonical Markdown documents using exact policy IDs.

        All sources and attachments are validated/read before the first write,
        so a missing, duplicate, unknown, malformed, or unreadable Markdown
        file cannot cause a partial filename-based repair. A PDF is accepted
        only when it has the exact verified Markdown stem or is retained from
        an active exact-ID ingestion record.
        """
        self._verified_policy_document_ids = set()

        import os
        from pathlib import Path

        from ingestion.source_locator import (
            MarkdownSourceValidationError,
            resolve_markdown_policy_sources,
        )

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        resolved_data_dir = Path(data_dir or os.path.join(base_dir, "Data"))
        resolved_catalog_path = Path(
            catalog_path or os.path.join(base_dir, "recommendation", "policy_catalog.json")
        )

        if not resolved_catalog_path.exists():
            logger.warning(
                "Skipping Markdown policy synchronization because the catalog is missing",
                extra={"catalog_path": str(resolved_catalog_path)},
            )
            return False

        try:
            with resolved_catalog_path.open("r", encoding="utf-8") as handle:
                raw_catalog = json.load(handle)
            if not isinstance(raw_catalog, list):
                raise ValueError("Policy catalog root must be a list")
            catalog = [
                entry for entry in raw_catalog
                if isinstance(entry, dict)
                and entry.get("_ingestion_status", "active") == "active"
            ]
            sources = resolve_markdown_policy_sources(resolved_data_dir, catalog)

            # Prepare every desired row before applying the first change.
            prepared = []
            for entry in catalog:
                policy_id = str(entry.get("policy_id") or "").strip()
                policy_name = str(entry.get("policy_name") or "").strip()
                if not policy_id or not policy_name:
                    raise ValueError("Every active catalog entry needs policy_id and policy_name")

                source = sources[policy_id]
                same_stem_pdf = source.path.with_suffix(".pdf")
                pdf_bytes = None
                if same_stem_pdf.is_file():
                    pdf_bytes = same_stem_pdf.read_bytes()
                    verified_pdf_path: Optional[str] = str(same_stem_pdf.resolve())
                else:
                    verified_pdf = self._get_verified_ingestion_pdf(policy_id)
                    verified_pdf_path = None
                    if verified_pdf:
                        recorded_path = verified_pdf.get("pdf_path")
                        pdf_bytes = _optional_bytes(verified_pdf.get("pdf_data"))
                        if recorded_path:
                            recorded_file = Path(recorded_path)
                            if pdf_bytes is not None or recorded_file.is_file():
                                verified_pdf_path = str(recorded_file.resolve())
                                if pdf_bytes is None:
                                    pdf_bytes = recorded_file.read_bytes()

                existing = self._fetch_policy_document_exact(policy_id)
                prepared.append(
                    {
                        "entry": entry,
                        "source_text": source.text,
                        "pdf_path": verified_pdf_path,
                        "pdf_bytes": pdf_bytes,
                        "existing": existing,
                    }
                )
        except (MarkdownSourceValidationError, OSError, ValueError, KeyError) as exc:
            logger.error("Exact-ID Markdown synchronization aborted before writes: %s", exc)
            return False
        except Exception:
            logger.exception("Exact-ID Markdown synchronization preparation failed before writes")
            return False

        verified_policy_ids: set[str] = set()
        for item in prepared:
            entry = item["entry"]
            policy_id = str(entry["policy_id"])
            if _document_matches_source(
                item["existing"],
                entry,
                item["source_text"],
                item["pdf_path"],
                item["pdf_bytes"],
            ):
                verified_policy_ids.add(policy_id)
                continue

            saved = self.save_policy_document(
                policy_id=policy_id,
                policy_name=str(entry["policy_name"]),
                document_text=item["source_text"],
                insurer=entry.get("insurer"),
                plan_type=entry.get("plan_type"),
                premium=entry.get("premium"),
                sum_insured=entry.get("sum_insured"),
                pdf_path=item["pdf_path"],
                pdf_bytes=item["pdf_bytes"],
                replace_pdf=True,
                backup_id=(
                    _exact_id_sync_backup_id(policy_id)
                    if item["existing"] is not None
                    else None
                ),
                backup_reason=f"before exact-ID Markdown synchronization {_EXACT_ID_SYNC_VERSION}",
            )
            if not saved:
                logger.error(
                    "Exact-ID Markdown synchronization could not save policy",
                    extra={"policy_id": policy_id},
                )
            else:
                verified_policy_ids.add(policy_id)

        self._verified_policy_document_ids = verified_policy_ids
        return len(verified_policy_ids) == len(prepared)
