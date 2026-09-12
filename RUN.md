# Kivi Semantic Memory — Run Guide

Complete step-by-step guide for the reviewing agent or any evaluator.

## Prerequisites

- Python 3.10+ (tested with 3.13)
- pip
- A Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

## Quick Start

### 1. Environment Setup

```bash
cd kivi

# Create .env from template
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

**Optional model overrides** — add to `.env` if you want to use Gemini 3.x models:
```
EXTRACTION_MODEL=gemini-3.5-flash-lite
GENERATION_MODEL=gemini-3.8-flash
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Initialize Database

```bash
python -m db.migrate
```

You should see:
```
[OK] Database migrated successfully.
  dictionary: 0 rows
  dictionary_suggestions: 0 rows
  embeddings: 0 rows
  history: 0 rows
  memories: 0 rows
  query_logs: 0 rows
```

### 4. Ingest Corpus

```bash
python -m ingestion.pipeline --corpus data/corpus.jsonl
```

This runs the full pipeline:
1. Loads 500 corpus records
2. Auto-populates the dictionary from name correction patterns
3. Inserts history records with dictionary correction
4. Runs LLM extraction on each record (this is the slow step — ~15-30 min at free-tier rate limits)
5. Generates embeddings for all history + memory records

Progress is printed per batch. You can safely interrupt and re-run — the pipeline is idempotent.

### 5. Start Server

```bash
python server.py
```

Opens at: **http://127.0.0.1:8000**

### 6. Try Hey Kivi

Open the browser. Try these questions:

| Question | Expected Type |
|----------|--------------|
| "Who is Devon Marsh and what's their role?" | Single fact lookup |
| "What's my preference for Slack messages?" | Preference recall |
| "What is Atlas called now?" | Knowledge update (rename) |
| "Where do I sit right now?" | Temporal reasoning |
| "Did I talk to Priya Raman about the onboarding flow?" | Episodic recall |
| "What's my performance review rating?" | Abstention (not in corpus) |

### 7. Run Evaluation

```bash
python -m eval.run_eval
```

Results are printed to console and saved to `eval/results/eval_results.json`.

The eval runs all 39 questions through the full pipeline and scores each with an LLM judge.

### 8. Spot-Check Aggregation

```bash
python -m eval.spot_check --query "List everyone I talk to on Slack"
```

This prints the full pipeline output: routing decision, retrieved results, generated answer, and verification.

## Import a Different Corpus

1. Prepare a JSONL file with the same schema as `data/corpus.jsonl`:
   ```json
   {"id": "rec_001", "timestamp": "ISO 8601", "day_index": 0, "app": "Slack", "style_applied": "casual", "raw_asr_output": "...", "llm_formatted_output": "...", "category": "...", "entities": [], "ground_truth": null, "word_count": 10}
   ```

2. Reset the database:
   ```bash
   # Windows
   del data\kivi.db

   # Mac/Linux
   rm data/kivi.db
   ```

3. Re-run from step 3:
   ```bash
   python -m db.migrate
   python -m ingestion.pipeline --corpus path/to/your/corpus.jsonl
   ```

## Inspect

- **Query logs**: `GET /api/query-logs` or check `query_logs` table in SQLite
- **Memory state**: `GET /api/memories` or browse in the UI
- **Stats**: `GET /api/stats`
- **Database**: `sqlite3 data/kivi.db ".tables"` then `SELECT * FROM memories LIMIT 10;`

## Project Structure

```
kivi/
├── config.py              # Model pins, weights, rate limits
├── server.py              # FastAPI server (all endpoints)
├── llm.py                 # Gemini SDK wrapper with retry
├── requirements.txt
├── README.md
├── RUN.md                 # This file
├── .env.example           # Template for API key
├── data/
│   ├── corpus.jsonl       # Input corpus
│   ├── eval_questions.jsonl # Eval questions
│   └── kivi.db            # SQLite database (generated)
├── db/
│   ├── schema.sql         # All tables
│   ├── database.py        # Connection helpers
│   └── migrate.py         # Schema migration
├── ingestion/
│   ├── dictionary.py      # Term correction
│   ├── extractor.py       # LLM memory extraction
│   ├── embedder.py        # Embedding generation
│   └── pipeline.py        # Orchestrator
├── query/
│   ├── router.py          # LLM function-calling router
│   ├── tools.py           # Deterministic tool implementations
│   ├── bm25.py            # BM25 scorer
│   ├── generator.py       # Answer generation
│   └── verifier.py        # Grounding verification
├── eval/
│   ├── run_eval.py        # Evaluation harness
│   ├── spot_check.py      # Manual spot-check
│   └── results/           # Generated eval output
└── static/
    ├── index.html
    ├── style.css
    └── app.js
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `GEMINI_API_KEY not set` | Add your key to `.env` |
| Rate limit errors during ingestion | Pipeline retries automatically. If persistent, increase `INTER_BATCH_DELAY` in config.py |
| Unicode errors on Windows | We use ASCII-safe output. If you see encoding errors, set `PYTHONIOENCODING=utf-8` |
| Port 8000 in use | Set `PORT=8001` in `.env` |
| Empty memories after ingestion | Check that the corpus path is correct and records were loaded |
