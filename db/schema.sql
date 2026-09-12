-- Kivi Semantic Memory — Database Schema

-- Immutable dictation history log
CREATE TABLE IF NOT EXISTS history (
    id TEXT PRIMARY KEY,
    raw_text TEXT,
    formatted_text TEXT NOT NULL,
    canonical_text TEXT NOT NULL,
    app TEXT,
    timestamp TEXT NOT NULL,
    day_index INTEGER,
    style_applied TEXT,
    metadata TEXT,
    superseded_by TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_history_app ON history(app);
CREATE INDEX IF NOT EXISTS idx_history_timestamp ON history(timestamp);
CREATE INDEX IF NOT EXISTS idx_history_day ON history(day_index);

-- Bi-temporal semantic memory
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK(kind IN ('factual', 'episodic', 'preference')),
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    value TEXT NOT NULL,
    qualifier TEXT,
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    source_record_ids TEXT NOT NULL,
    confidence REAL DEFAULT 0.8,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_memories_subject ON memories(subject);
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_memories_valid ON memories(valid_from, valid_to);
CREATE INDEX IF NOT EXISTS idx_memories_predicate ON memories(predicate);

-- Embedding vector store
CREATE TABLE IF NOT EXISTS embeddings (
    id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL CHECK(source_type IN ('history', 'memory')),
    source_id TEXT NOT NULL,
    embedding BLOB NOT NULL,
    text_content TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_embeddings_source ON embeddings(source_type, source_id);

-- User-taught dictionary corrections
CREATE TABLE IF NOT EXISTS dictionary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_form TEXT NOT NULL UNIQUE,
    corrected_form TEXT NOT NULL,
    occurrence_count INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Auto-suggestion candidates for dictionary
CREATE TABLE IF NOT EXISTS dictionary_suggestions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_form TEXT NOT NULL,
    corrected_form TEXT NOT NULL,
    occurrence_count INTEGER DEFAULT 1,
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'accepted', 'dismissed')),
    created_at TEXT DEFAULT (datetime('now'))
);

-- Query audit log for inspectability
CREATE TABLE IF NOT EXISTS query_logs (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    tool_name TEXT,
    tool_args TEXT,
    router_reasoning TEXT,
    retrieved_memory_ids TEXT,
    retrieved_context TEXT,
    generated_answer TEXT,
    verification_result TEXT,
    verification_detail TEXT,
    latency_ms REAL,
    model_calls INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Processed history tracking for idempotent pipeline runs
CREATE TABLE IF NOT EXISTS processed_history (
    history_id TEXT PRIMARY KEY,
    processed_at TEXT DEFAULT (datetime('now'))
);

