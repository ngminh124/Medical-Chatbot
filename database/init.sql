-- ============================================================
-- Medical RAG Chatbot - Database Initialization Script
-- ============================================================
-- Description: Khởi tạo cơ sở dữ liệu PostgreSQL cho dự án
--              Vietnamese Medical RAG Chatbot.
-- Database:    medical_rag_db
-- ============================================================

-- =========================
-- 1. Extensions
-- =========================
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =========================
-- 2. Custom ENUM Types
-- =========================
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'message_role') THEN
        CREATE TYPE message_role AS ENUM ('user', 'assistant');
    END IF;
END
$$;

-- =========================
-- 3. Tables
-- =========================

-- ----- users -----
CREATE TABLE IF NOT EXISTS users (
    id                          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email                       TEXT NOT NULL UNIQUE,
    email_verified              BOOLEAN NOT NULL DEFAULT FALSE,
    verification_token          TEXT,
    verification_token_time     TIMESTAMP WITH TIME ZONE,
    password                    TEXT NOT NULL,
    reset_password_token        TEXT,
    reset_password_token_time   TIMESTAMP WITH TIME ZONE,
    phone                       TEXT,
    name                        TEXT NOT NULL,
    type                        TEXT,
    status                      TEXT NOT NULL DEFAULT 'active',
    created_at                  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- ----- threads (phiên trò chuyện) -----
CREATE TABLE IF NOT EXISTS threads (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL
                        REFERENCES users(id) ON DELETE CASCADE,
    title           VARCHAR(512) NOT NULL DEFAULT 'Cuộc trò chuyện mới',
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- ----- messages -----
CREATE TABLE IF NOT EXISTS messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id       UUID NOT NULL
                        REFERENCES threads(id) ON DELETE CASCADE,
    role            message_role NOT NULL,
    content         TEXT NOT NULL,
    metadata        JSONB DEFAULT '{}'::JSONB,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- ----- feedbacks -----
CREATE TABLE IF NOT EXISTS feedbacks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id      UUID NOT NULL
                        REFERENCES messages(id) ON DELETE CASCADE,
    rating          INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    comment         TEXT,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- ----- documents (tài liệu y tế gốc) -----
CREATE TABLE IF NOT EXISTS documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title           TEXT NOT NULL,
    content         TEXT NOT NULL,
    source_url      TEXT,
    metadata        JSONB DEFAULT '{}'::JSONB,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- ----- chunks (đoạn văn bản đã cắt) -----
CREATE TABLE IF NOT EXISTS chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL
                        REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    metadata        JSONB DEFAULT '{}'::JSONB,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- =========================
-- 4. Indexes
-- =========================

-- users
CREATE INDEX IF NOT EXISTS idx_users_email          ON users (email);
CREATE INDEX IF NOT EXISTS idx_users_status         ON users (status);

-- threads
CREATE INDEX IF NOT EXISTS idx_threads_user_id      ON threads (user_id);
CREATE INDEX IF NOT EXISTS idx_threads_created_at   ON threads (created_at DESC);

-- messages
CREATE INDEX IF NOT EXISTS idx_messages_thread_id   ON messages (thread_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at  ON messages (created_at);

-- feedbacks
CREATE INDEX IF NOT EXISTS idx_feedbacks_message_id ON feedbacks (message_id);

-- documents
CREATE INDEX IF NOT EXISTS idx_documents_title      ON documents (title);

-- chunks
CREATE INDEX IF NOT EXISTS idx_chunks_document_id   ON chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index   ON chunks (document_id, chunk_index);

-- =========================
-- 5. Trigger: auto-update updated_at
-- =========================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
DECLARE
    tbl TEXT;
BEGIN
    FOREACH tbl IN ARRAY ARRAY['users', 'threads', 'documents']
    LOOP
        EXECUTE format(
            'CREATE OR REPLACE TRIGGER trg_%s_updated_at
             BEFORE UPDATE ON %I
             FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();',
            tbl, tbl
        );
    END LOOP;
END
$$;

-- ============================================================
-- Done – Database is ready 🚀
-- ============================================================
