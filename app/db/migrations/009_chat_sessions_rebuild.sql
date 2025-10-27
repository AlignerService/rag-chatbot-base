-- 009_chat_sessions_rebuild.sql
-- Opgraderer chat_sessions fra gammelt skema:
--   id, created_at, clinic_id, contact_email, status, token_hash
-- til nyt skema:
--   session_id (PK), user_email, status, created_at, updated_at, token_hash

PRAGMA foreign_keys = OFF;

BEGIN;

-- 1) Byg ny tabel med det rigtige skema
CREATE TABLE IF NOT EXISTS chat_sessions_new (
  session_id  TEXT PRIMARY KEY,
  user_email  TEXT,
  status      TEXT,
  created_at  TEXT,
  updated_at  TEXT,
  token_hash  TEXT
);

-- 2) Kopiér data fra gammelt skema til nyt
--   - session_id  <- id
--   - user_email  <- contact_email
--   - status: normaliser 'open' -> 'active', ellers lowercase af eksisterende værdi eller 'active'
--   - updated_at  <- created_at (first fill; app’en opdaterer senere)
INSERT INTO chat_sessions_new (session_id, user_email, status, created_at, updated_at, token_hash)
SELECT
  id                                AS session_id,
  contact_email                     AS user_email,
  CASE lower(COALESCE(status,'active'))
    WHEN 'open' THEN 'active'
    ELSE lower(COALESCE(status,'active'))
  END                               AS status,
  created_at                        AS created_at,
  created_at                        AS updated_at,
  token_hash                        AS token_hash
FROM chat_sessions;

-- 3) Swap tabellerne
DROP TABLE chat_sessions;
ALTER TABLE chat_sessions_new RENAME TO chat_sessions;

-- 4) Indekser til det nye skema
CREATE INDEX IF NOT EXISTS idx_chat_sessions_session_id ON chat_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_status     ON chat_sessions(status);

COMMIT;

-- 5) Sikr at chat_messages findes med det forventede skema
-- (ikke destruktivt: opretter kun hvis den ikke findes)
CREATE TABLE IF NOT EXISTS chat_messages (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id  TEXT,
  role        TEXT,
  content     TEXT,
  created_at  TEXT
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id);

-- 6) Backfill created_at i chat_messages hvis der ligger gamle rækker uden stempel
UPDATE chat_messages
SET created_at = COALESCE(created_at, datetime('now'))
WHERE created_at IS NULL;
