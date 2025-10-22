-- Track kørte migrationer
CREATE TABLE IF NOT EXISTS schema_migrations (
  version TEXT PRIMARY KEY,
  applied_at TEXT NOT NULL
);

-- chat_sessions: “patient-light” state for samtaler
CREATE TABLE IF NOT EXISTS chat_sessions (
  session_id TEXT PRIMARY KEY,
  user_email TEXT,
  brand TEXT,             -- fx ClearCorrect, Invisalign, ...
  case_type TEXT,         -- open_bite, deep_bite, rotations, ...
  adult_child TEXT,       -- adult|child|unknown
  habits TEXT,            -- JSON: {"tongue_thrust":true, "bruxism":false}
  resources TEXT,         -- JSON: {"photos":"yes","xray":"no","ios":"unknown"}
  status TEXT DEFAULT 'active',  -- active|handover|closed
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

-- chat_messages: samtalelog + bot-svar (råt)
CREATE TABLE IF NOT EXISTS chat_messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  role TEXT NOT NULL,         -- user|assistant|system
  content TEXT NOT NULL,      -- brugerbesked el. bot-output
  confidence REAL,            -- NULL for user/system
  sources TEXT,               -- JSON-liste med kilde-id’er
  created_at TEXT NOT NULL
);

-- chat_answers: kuraterbare svar (det du vil reviewe)
CREATE TABLE IF NOT EXISTS chat_answers (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  question TEXT NOT NULL,
  answer_md TEXT NOT NULL,    -- “chat_clinical” markdown
  confidence REAL NOT NULL,
  sources_json TEXT,          -- JSON-liste med labels/id’er
  needs_review INTEGER NOT NULL DEFAULT 1,  -- 1=vis i konsol
  created_at TEXT NOT NULL
);

-- nyttige indeks
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_answers_session ON chat_answers(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_answers_review ON chat_answers(needs_review);

-- marker migration som kørt
INSERT OR IGNORE INTO schema_migrations(version, applied_at)
VALUES('003_add_chat_tables', strftime('%Y-%m-%dT%H:%M:%fZ','now'));
