-- Non-destructive schema migration. The application performs the idempotent
-- JSONB-to-message data copy during PostgresDBManager.init_db().

CREATE TABLE IF NOT EXISTS conversation_sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    customer_id VARCHAR(255),
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    session_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

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

CREATE INDEX IF NOT EXISTS idx_conversation_messages_session_sequence
ON conversation_messages(session_id, sequence);

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

CREATE INDEX IF NOT EXISTS idx_policy_ingestions_policy_status
ON policy_ingestions(policy_id, status);
