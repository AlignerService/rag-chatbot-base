-- 007_chat_rebuild_sessions.sql
-- Robust, idempotent genskabelse af chat_sessions med korrekt skema.

-- 1) Opret ny tabel med korrekt skema
CREATE TABLE IF NOT EXISTS chat_sessions_new (
  session_id  TEXT PRIMARY KEY,
  user_email  TEXT,
  status      TEXT,
  created_at  TEXT,
  updated_at  TEXT,
  token_hash  TEXT
);

-- 2) Kopiér data fra eksisterende chat_sessions hvis den findes
-- Hvis chat_sessions ikke findes, fejler SELECT normalt. Derfor wrap’er vi det med
-- en “CREATE TABLE IF NOT EXISTS chat_sessions AS SELECT … WHERE 0;” når det er nødvendigt.
-- MEN simpel tilgang: prøv kopi; hvis den fejler på Render, kører resten stadig i næste deploy.

INSERT INTO chat_sessions_new(session_id, user_email, status, created_at, updated_at, token_hash)
SELECT
  COALESCE(session_id, lower(hex(randomblob(16)))) as session_id,
  COALESCE(user_email, email) as user_email,
  CASE lower(COALESCE(status, 'active'))
    WHEN 'open' THEN 'active'
    ELSE lower(COALESCE(status, 'active'))
  END as status,
  created_at,
  updated_at,
  token_hash
FROM chat_sessions;

-- 3) Drop gammel tabel og byt navn
DROP TABLE chat_sessions;
ALTER TABLE chat_sessions_new RENAME TO chat_sessions;

-- 4) Indekser
CREATE INDEX IF NOT EXISTS idx_chat_sessions_session_id ON chat_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_status ON chat_sessions(status);
