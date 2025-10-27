-- 006_chat_safe_add_session.sql
-- Idempotent “safe” migration uden ALTER-fejl-trap:
-- 1) Opret en ny tabel med det rigtige skema, hvis den ikke findes.
CREATE TABLE IF NOT EXISTS chat_sessions_new (
  session_id  TEXT PRIMARY KEY,
  user_email  TEXT,
  status      TEXT,
  created_at  TEXT,
  updated_at  TEXT,
  token_hash  TEXT
);

-- 2) Kopiér rækker fra gamle chat_sessions ind i den nye.
-- Vi kan ikke gætte alle kolonnenavne i den gamle, så vi bruger COALESCE og NULL’er.
-- Mangler en kolonne i kildetabellen, ender den som NULL (helt fint).
INSERT INTO chat_sessions_new(session_id, user_email, status, created_at, updated_at, token_hash)
SELECT
  -- hvis gammel kolonne 'session_id' findes, brug den; ellers generér en ny id
  COALESCE(session_id, lower(hex(randomblob(16)))) as session_id,
  -- prøv user_email, ellers email
  COALESCE(user_email, email) as user_email,
  -- normalisér status
  CASE lower(COALESCE(status, 'active'))
    WHEN 'open' THEN 'active'
    ELSE lower(COALESCE(status, 'active'))
  END as status,
  created_at,
  updated_at,
  token_hash
FROM chat_sessions;

-- 3) Erstat gamle tabel atomisk(ish).
DROP TABLE chat_sessions;
ALTER TABLE chat_sessions_new RENAME TO chat_sessions;

-- 4) Indekser
CREATE INDEX IF NOT EXISTS idx_chat_sessions_session_id ON chat_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_status ON chat_sessions(status);
