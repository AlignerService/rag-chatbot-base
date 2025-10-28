-- 010_create_moderation_tables.sql

BEGIN;

CREATE TABLE IF NOT EXISTS chat_events (
  id            TEXT PRIMARY KEY,
  session_id    TEXT NOT NULL,
  role          TEXT NOT NULL CHECK (role IN ('user','assistant')),
  text          TEXT NOT NULL,
  created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chat_events_session_role_created
  ON chat_events (session_id, role, created_at);

CREATE TABLE IF NOT EXISTS answers_for_review (
  id               TEXT PRIMARY KEY,
  source_event_id  TEXT NOT NULL,
  proposed_text    TEXT NOT NULL,
  status           TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','approved','rejected')),
  reviewer         TEXT,
  reviewed_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_answers_for_review_status_created
  ON answers_for_review (status);

CREATE TABLE IF NOT EXISTS approved_qa (
  id             TEXT PRIMARY KEY,
  answer_text    TEXT NOT NULL,
  source_event_id TEXT,
  approved_at    TEXT NOT NULL
);

COMMIT;
