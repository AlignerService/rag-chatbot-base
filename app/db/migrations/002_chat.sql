-- 002_chat.sql
CREATE TABLE IF NOT EXISTS chat_sessions (
  id            TEXT PRIMARY KEY,
  created_at    TEXT NOT NULL DEFAULT (datetime('now')),
  clinic_id     TEXT,
  contact_email TEXT,
  status        TEXT NOT NULL DEFAULT 'open',   -- open|closed
  token_hash    TEXT NOT NULL                   -- sha256(token)
);

CREATE TABLE IF NOT EXISTS chat_messages (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id    TEXT NOT NULL,
  role          TEXT NOT NULL,                  -- user|assistant|system
  content       TEXT NOT NULL,
  created_at    TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, created_at);
