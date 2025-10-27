-- 004_chat_token_hash.sql
ALTER TABLE chat_sessions ADD COLUMN token_hash TEXT;
CREATE INDEX IF NOT EXISTS idx_chat_sessions_status ON chat_sessions(status);
