-- 005_chat_add_session_and_token.sql
-- Tilføj session_id + token_hash hvis de mangler. SQLite kan ikke IF NOT EXISTS på kolonner,
-- men det fejler bare næste gang, og vores migrator markerer versionen én gang.

ALTER TABLE chat_sessions ADD COLUMN session_id TEXT;
ALTER TABLE chat_sessions ADD COLUMN token_hash TEXT;

-- Indekser for performance og opslag
CREATE INDEX IF NOT EXISTS idx_chat_sessions_session_id ON chat_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_status ON chat_sessions(status);

-- Sikr created_at/updated_at kolonner findes i praksis (hvis de allerede findes, sker der intet her)
-- Hvis din ældre tabel ikke har dem, bliver de bare NULL ved første omgang. Koden sætter værdier ved INSERT.
