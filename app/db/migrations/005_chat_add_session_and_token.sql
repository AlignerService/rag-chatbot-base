-- 005_chat_add_session_and_token.sql
-- Tilføj manglende kolonner og indeks. Brug CREATE INDEX IF NOT EXISTS,
-- og undgå referencer til ukendte kolonnenavne for at holde det idempotent.

-- Forsøg at tilføje session_id; hvis den allerede findes, vil nogle SQLite-versioner kaste fejl.
-- Hvis din migrator afbryder på fejl, så læs næste punkt (vi har en “safe” version i punkt 2).
ALTER TABLE chat_sessions ADD COLUMN session_id TEXT;

-- token_hash dækkes af 004, men hvis 004 ikke kørte, hjælper denne linje.
ALTER TABLE chat_sessions ADD COLUMN token_hash TEXT;

CREATE INDEX IF NOT EXISTS idx_chat_sessions_session_id ON chat_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_status ON chat_sessions(status);
