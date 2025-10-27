-- 008_chat_migrate_from_legacy.sql
-- Migrér fra ældre skema (id, contact_email, ...) til nyt skema (session_id, user_email, ...)

-- 1) Ny tabel med korrekt skema
CREATE TABLE IF NOT EXISTS chat_sessions_new (
  session_id  TEXT PRIMARY KEY,
  user_email  TEXT,
  status      TEXT,
  created_at  TEXT,
  updated_at  TEXT,
  token_hash  TEXT
);

-- 2) Kopiér eksisterende rækker fra det gamle skema
-- Vi mapper: id -> session_id, contact_email -> user_email, created_at -> created_at/updated_at
-- og normaliserer status 'open' -> 'active'
INSERT INTO chat_sessions_new(session_id, user_email, status, created_at, updated_at, token_hash)
SELECT
  id AS session_id,
  contact_email AS user_email,
  CASE lower(COALESCE(status, 'active'))
    WHEN 'open' THEN 'active'
    ELSE lower(COALESCE(status, 'active'))
  END AS status,
  created_at,
  created_at AS updated_at,
  token_hash
FROM chat_sessions;

-- 3) Byt tabeller
DROP TABLE chat_sessions;
ALTER TABLE chat_sessions_new RENAME TO chat_sessions;

-- 4) Indekser
CREATE INDEX IF NOT EXISTS idx_chat_sessions_session_id ON chat_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_status ON chat_sessions(status);
