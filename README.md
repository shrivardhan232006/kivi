# Kivi Semantic Memory — Hey Kivi v2

A semantic memory system for [Kivi](https://heykivi.ai), a voice-first dictation product. Ingests dictation history, extracts structured memories using LLM calls, and answers questions about the user's people, projects, preferences, and past events.

## Architecture

```
raw ASR → dictionary correction → canonical text (History) → LLM extraction → Memories
                                                                                  ↓
User question → LLM function-calling router → deterministic tool → LLM generation → verification → answer
```

> 📖 For the complete architectural specification and pipeline documentation, see [PIPELINE.md](PIPELINE.md).

### Three-Layer Boundary

| Layer | Nature | Job |
|-------|--------|-----|
| **Dictionary** | Deterministic | Exact term corrections (ASR misspellings & name replacements) |
| **History** | Raw log, immutable | Permanent record of what was said |
| **Semantic Memory** | LLM-extracted, confidence-scored | Structured understanding derived from History |

### Memory Schema (Bi-temporal)

```
(subject, predicate, value, qualifier, valid_from, valid_to, source_record_ids, confidence)
```

When a fact changes, the old record is marked `valid_to` rather than deleted — enabling temporal reasoning ("what was X called before?").

### Query Router

Uses Gemini's native **function-calling API** to route questions to five deterministic tools:

| Tool | Implementation | Use Case |
|------|---------------|----------|
| `lookup_fact(subject, predicate)` | SQL exact lookup + rename/alias resolution | "Who is Devon Marsh?", "Who's our Northwind contact?" |
| `aggregate(filter_field, filter_value)` | Case-insensitive `SELECT DISTINCT` | "List everyone I talk to on Slack" |
| `fuzzy_search(query, top_k)` | Semantic + BM25 + timeline recency fusion | "Did I talk to Priya about the onboarding flow?" |
| `get_valid_at(subject, predicate, timestamp)` | Bi-temporal SQL | "What was Atlas called before?", "Where do I sit now?" |
| `update_dictionary(raw_form, corrected_form)` | Dynamic dictionary & memory updater | "Its Priya Reddy not raman" |

**Key invariant**: Tools are SQL-backed and deterministic. The LLM decides *which* tool and *what arguments* — never executes retrieval itself. `aggregate` always runs `SELECT DISTINCT`, never similarity ranking.

**Fast-path bypass**: If the question is a direct factual lookup for an exact entity name from memory, `lookup_fact` is called directly without an LLM routing call.

### Retrieval Signals (fuzzy_search)

Three signals fused with configurable weights:
1. **Semantic similarity** (cosine, 0.45) — paraphrase matching via embeddings
2. **Lexical precision** (BM25, 0.35) — exact names, jargon, ASR-garbled entities
3. **Temporal recency** (exponential decay, 0.20) — timeline-anchored decay (half-life 7 days)

### Verification Pass

Every generated answer goes through a grounding check: an LLM verifies that each claim in the answer appears in the source context. Unsupported claims are flagged; heavily unsupported answers fall back to abstention.

## Model Pins

| Role | Model | Rationale |
|------|-------|-----------|
| Extraction, Verification, Routing | `gemini-3.5-flash-lite` | High volume, fast latency, lowest cost |
| Answer Generation | `gemini-3.5-flash-lite` | Context-grounded synthesis |
| Embeddings | `gemini-embedding-2` | Multilingual, semantic vector representations |

All configurable via `.env`. Temperature: 0.1 for extraction/verification, 0.2 for generation.

## Limitations

- **Entity-graph traversal**: Multi-hop relationship queries (e.g., "who does my manager's manager report to?") are out of scope. At 500-record corpus scale, insufficient depth to justify the complexity.
- **Embedding-only similarity can miss aggregation members**: This is why `aggregate` uses SQL filters, not similarity search.
- **ASR garbling**: Dictionary correction helps but can't fix all misspellings. The BM25 signal in fuzzy_search helps catch near-matches.
- **Extraction quality**: LLM extraction is imperfect — some facts may be missed or over-extracted from ambiguous dictations. The confidence score and verification pass mitigate but don't eliminate this.
- **Rate limits**: At free-tier Gemini API limits, full corpus ingestion takes ~15-30 minutes due to rate limiting. Batch sizes and delays are configurable.
- **No real-time dictation**: This is a batch-ingestion demo, not a live streaming integration.

## Eval Results

Run `python -m eval.run_eval` after ingestion to see real results. Results include real failures — a results section with zero failures would be a red flag.

## AI Assistance

This codebase was built with AI coding assistance (Antigravity IDE with Claude Opus). All architectural decisions, prompt engineering, and evaluation design were human-directed. The AI generated code, documentation, and test infrastructure under human review.

## Tech Stack

- **Backend**: Python 3.13, FastAPI, SQLite (WAL mode)
- **LLM**: Google Gemini via `google-genai` SDK
- **Frontend**: Vanilla HTML/CSS/JavaScript (dark theme)
- **Embeddings**: Stored as numpy arrays in SQLite BLOB columns
- **BM25**: Inline ~90-line implementation (no search engine dependency)
