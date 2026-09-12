# Kivi Memory — Complete Pipeline & Working Architecture

This document provides a comprehensive, end-to-end breakdown of how **Kivi** is architected, how data flows from voice dictations into structured memory, and how queries are routed, retrieved, generated, and verified.

---

## 1. High-Level Architectural Philosophy

Kivi transforms spoken dictations into an inspectable, reliable, and grounded personal memory workspace. Unlike conventional RAG chatbots that blindly chunk text and suffer from hallucinations, lost context, and incomplete listings, Kivi operates on five core principles:

1. **Deterministic Retrieval via Fixed SQL Tools**: The LLM chooses *which tool* to call and with *what arguments*, but the retrieval itself is executed by deterministic, reproducible SQL queries and three-signal algorithms.
2. **Bi-Temporal Memory Tracking**: Every memory tracks both transaction time (`created_at`) and valid time (`valid_from`, `valid_to`). Facts that change over time (such as office moves or project renames) supersede older records without deleting history.
3. **No Top-K Dropout for Aggregations**: Queries asking for complete sets (e.g. "Who are all the people I talk to on Slack?") use `SELECT DISTINCT` SQL aggregations rather than vector similarity top-K ranking, guaranteeing zero omissions.
4. **Three-Signal Search Fusion**: For fuzzy and episodic questions, Kivi fuses **Semantic Embeddings** (45%), **BM25 Lexical Matching** (35%), and **Timeline Recency Decay** (20%).
5. **Adversarial Grounding Verification**: Every generated answer undergoes an automated verification pass before reaching the user. Any claim not strictly supported by the source dictation context is flagged or abstained from.

---

## 2. End-to-End Pipeline Diagram

```
                              [ User Dictation ]
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ 1. Dictionary Correction  │ (ASR noise repair, deduplication)
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ 2. History Storage (SQL)  │ (Immutable dictation archive)
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ 3. LLM Memory Extractor   │ (Factual, Episodic, Preference)
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ 4. Conflict Resolution    │ (Bi-temporal invalidation / update)
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ 5. Dual Vector Store      │ (History & Memory embeddings)
                        └───────────────────────────┘
                                      │
======================================│======================================
                          QUERY & INFERENCE PIPELINE
======================================│======================================
                                      │
                                [ User Query ]
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ Contextualization Layer   │ (Rewrites follow-up questions)
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ Fast-Path Check           │ (Instant bypass for direct facts)
                        └───────┬───────────┬───────┘
                                │ Bypass    │ Fallback / Complex
                                ▼           ▼
                        ┌──────────┐   ┌───────────────────────────┐
                        │ Direct   │   │ LLM Function Router       │
                        │ Lookup   │   └─────────────┬─────────────┘
                        └────┬─────┘                 │
                             │   ┌───────────────────┴───────────────────┐
                             │   │                   │                   │
                             ▼   ▼                   ▼                   ▼
                     ┌───────────────┐       ┌───────────────┐   ┌───────────────┐
                     │ lookup_fact() │       │  aggregate()  │   │ get_valid_at()│
                     └───────┬───────┘       └───────┬───────┘   └───────┬───────┘
                             │                       │                   │
                             └───────────────┬───────┴───────────────────┘
                                             │ (If 0 results: Fallback)
                                             ▼
                                     ┌───────────────┐
                                     │ fuzzy_search()│ (Semantic + BM25 + Recency)
                                     └───────┬───────┘
                                             │
                                             ▼
                        ┌───────────────────────────┐
                        │ Context Formatter         │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ LLM Answer Generator      │ (Drafts persona-aligned response)
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ Grounding Verifier Pass   │ (Flags hallucinations / unsupported)
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        [ Verified Answer + Provenance Links ]
```

---

## 3. Pipeline Breakdown: Ingestion & Storage

The ingestion pipeline converts raw audio transcriptions into structured, queryable knowledge across four layers.

### Step 3.1: Vocabulary & Dictionary Normalization (`ingestion/dictionary.py`)
- **Speech-to-Text Noise Correction**: Automatic Speech Recognition (ASR) frequently distorts proper nouns (e.g. hearing "raman" when referring to "Priya Reddy", or "orien" for "Orion").
- **Deterministic Case-Insensitive Matching**: Regex whole-word boundaries (`\b`) match raw forms and replace them with canonical spellings. Entries are applied longest-first to prevent partial collisions.
- **Double-Token Deduplication**: An automated cleanup pass (`_deduplicate_consecutive_tokens`) strips adjacent duplicate words created by replacement cascades (e.g. turning `"Priya Priya Reddy"` into `"Priya Reddy"`).
- **Auto-Suggestion Engine**: Frequent corrections that appear consistently across dictations are suggested to the user for one-click addition to the dictionary.

### Step 3.2: Immutable History Log (`history` table)
- Dictations are permanently archived in SQLite with full metadata:
  - `id`: Unique record identifier (e.g. `rec_0001`).
  - `raw_text`: Exact raw transcription output.
  - `formatted_text`: Cleaned dictation text.
  - `canonical_text`: Post-dictionary corrected text.
  - `app`: Originating application (`Slack`, `Gmail`, `Notes`, `Linear`, `Messages`).
  - `timestamp`: ISO-8601 timestamp.

### Step 3.3: LLM Memory Extraction (`ingestion/extractor.py`)
Dictations are processed through `gemini-3.5-flash-lite` in structured JSON batches. The model extracts atomic memories classified into three categories:
1. **Factual**: Durable facts about people, roles, relationships, projects, organizations, and desk locations.
2. **Episodic**: Specific interactions or occurrences (who met with whom, about what topic, and in what context).
3. **Preference**: Communication styles, tool workflows, and formatting rules.

Every memory object defines:
- `subject`: Entity name or `"user"` for self-references.
- `predicate`: Attribute or action (e.g. `role`, `location`, `discussed`, `preference_for`).
- `value`: The core claim.
- `qualifier`: Contextual scope (e.g. `"for Slack messages"` vs `"for notes"`). This prevents conflicting preferences from falsely overwriting one another.
- `confidence`: Confidence score (0.0 to 1.0).

### Step 3.4: Bi-Temporal Conflict Resolution
When a new memory arrives with the same `subject` and `predicate`:
- **Same Scope, Different Value**: If qualifiers match and the value has changed (e.g. moving from `desk 412` to `desk 604`), the older record's `valid_to` is stamped with the new record's timestamp. The historical fact is preserved, while marking the new fact as the currently active truth.
- **Duplicate Value**: If the claim is already known, the new record ID is appended to `source_record_ids` for provenance without creating redundant memory rows.
- **Different Qualifier**: If qualifiers differ (e.g. short messages for Slack vs detailed paragraphs for Notes), both coexist as distinct scoped truths.

### Step 3.5: Dual Vector Embeddings (`ingestion/embedder.py`)
Embeddings are computed using `gemini-embedding-2` for both:
1. **History records**: Full canonical text of raw dictations.
2. **Memory records**: Structured string representation (`subject | predicate | value | qualifier`).
Vectors are stored as float32 binary blobs in SQLite, indexed by `source_type` and `source_id`.

---

## 4. Pipeline Breakdown: Query Routing & Retrieval

When a user submits a question via the web interface or API, the query pipeline executes in distinct stages:

### Step 4.1: Conversation Contextualization (`query/router.py`)
If a conversation has prior turns, follow-up queries (e.g. *"What about her?"* or *"Where was that?"*) are passed with recent dialogue turns to Gemini to produce a self-contained search query containing the explicit entities and constraints.

### Step 4.2: Fast-Path Evaluation
For simple, unambiguous factual inquiries (e.g. *"Who is Devon Marsh and what's their role?"*), Kivi detects the direct inquiry pattern and exact entity match, bypassing the LLM router to execute `lookup_fact` in under 2ms.
- Any complex, temporal (*"before"*, *"when"*), aggregation (*"all"*, *"list"*), episodic (*"did I talk to"*), or correction questions automatically bypass this shortcut and invoke the full LLM router.

### Step 4.3: LLM Function-Calling Router
The user query is evaluated by Gemini with 5 tool declarations:
1. **`lookup_fact(subject, predicate)`**:
   - Searches memories for facts about a person, project, or topic.
   - Automatically handles dictionary corrections, self-references (`"my desk"` -> `subject="user"`), rename aliases (e.g. finding that Atlas was renamed to Orion), and qualifiers.
2. **`aggregate(filter_field, filter_value)`**:
   - Executes deterministic SQL `SELECT DISTINCT` queries.
   - Filters on `app`, `kind`, `predicate`, or `subject` using case-insensitive comparisons.
   - Returns the complete set with zero dropouts.
3. **`get_valid_at(subject, predicate, timestamp)`**:
   - Bi-temporal SQL resolution.
   - Resolves what was true at a specific timestamp, what is true right now (`timestamp="now"`), or tracks the full history of changes (`timestamp="all"`).
4. **`fuzzy_search(query, top_k)`**:
   - Three-signal fusion search across memories and history records.
   - Fuses semantic similarity (45%), BM25 lexical relevance (35%), and temporal recency (20%).
5. **`update_dictionary(raw_form, corrected_form)`**:
   - Direct user instruction tool (e.g. *"its Priya Reddy not raman"*).
   - Inserts dictionary rules and updates active memory records across the database.

### Step 4.4: Deterministic Fallback Mechanism
If a structured tool returns 0 results, Kivi automatically falls back to `fuzzy_search` over both memories and raw dictations. This guarantees that unforeseen query phrasing never causes an empty response if relevant history exists.

---

## 5. Pipeline Breakdown: Generation & Verification

### Step 5.1: Structured Context Assembly (`query/generator.py`)
Retrieved memories and history records are assembled into a readable context block including:
- Memory Kind, Subject, Predicate, and Value.
- Qualifiers and Validity Windows (`valid_from`, `valid_to`).
- Provenance details: source record IDs, originating app, and timestamps.

### Step 5.2: LLM Synthesis (`generate_answer`)
Using `gemini-3.5-flash-lite`, the model synthesizes a concise, grounded answer adhering to strict prompt rules:
- Rely strictly on provided context.
- When answering temporal questions, state what changed and when.
- When information is missing, abstain clearly: *"I don't have that information in your dictation history."*

### Step 5.3: Adversarial Grounding Verification (`query/verifier.py`)
Before returning to the user, the answer, original query, and retrieved context are submitted to an independent verification pass:
- The verifier checks whether every claim made in the generated answer is strictly supported by the context.
- Unsupported claims are flagged.
- If multiple ungrounded claims are detected, the response automatically falls back to safe abstention.
- System actions (dictionary updates) and clear abstentions bypass redundant verification calls.

### Step 5.4: Audit Logging (`query_logs` table)
Every interaction is logged in SQLite with:
- Raw query and contextualized query.
- Router reasoning, chosen tool, and tool arguments.
- Retrieved memory IDs and context snippet.
- Final answer and verification status (`pass` or `fail`).
- End-to-end latency in milliseconds and total LLM API calls.

---

## 6. Database Schema Summary

The database is powered by SQLite with Write-Ahead Logging (WAL) enabled:

| Table | Purpose | Key Columns |
|---|---|---|
| `history` | Immutable log of all dictation records | `id`, `raw_text`, `canonical_text`, `app`, `timestamp`, `day_index` |
| `memories` | Bi-temporal structured knowledge | `id`, `kind`, `subject`, `predicate`, `value`, `qualifier`, `valid_from`, `valid_to`, `source_record_ids`, `confidence` |
| `embeddings` | Binary vector store | `id`, `source_type`, `source_id`, `embedding` (BLOB), `text_content` |
| `dictionary` | User-taught spelling & name corrections | `raw_form`, `corrected_form`, `occurrence_count` |
| `dictionary_suggestions` | Candidates for automated dictionary expansion | `raw_form`, `corrected_form`, `occurrence_count`, `status` |
| `query_logs` | Full audit trail of every query & verification | `id`, `question`, `tool_name`, `tool_args`, `router_reasoning`, `generated_answer`, `verification_result`, `latency_ms`, `model_calls` |
| `processed_history` | Idempotent tracker for ingestion runs | `history_id`, `processed_at` |

---

## 7. Bug Fixes & System Hardening Completed

During the comprehensive codebase review, the following bugs were diagnosed and resolved:

1. **GenAI Automatic Function Calling (AFC) & Non-Text Warnings**:
   - Added `automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)` to both `call_llm` and `call_llm_with_tools`.
   - Replaced direct `response.text` access on candidate responses containing function calls with safe text part concatenation, eliminating SDK warnings on stderr.

2. **Router Fast-Path Over-Interception**:
   - Fixed `_fast_path_check` in `query/router.py` which was intercepting questions that merely mentioned an entity name (e.g. episodic questions like *"Did I talk to Tom Okafor about roadmap slide?"*, or temporal questions like *"Before the rename, what was Orion called?"*).
   - Constrained fast-path exclusively to direct factual queries (`"Who is X?"`, `"What is X's role?"`).

3. **Rename & Reverse-Alias Memory Resolution**:
   - Updated `lookup_fact` and `get_valid_at` in `query/tools.py` to search `(predicate LIKE '%rename%' AND value LIKE ?)` in addition to `subject`. When asked what Orion was called before, the system now successfully discovers `Atlas renamed_to Orion`.

4. **Self-Reference & "My X" Query Normalization**:
   - Updated `lookup_fact` and `get_valid_at` so queries referencing `"my desk"` or `"my role"` automatically map `subject="user"` and resolve the attribute from predicate, qualifier, or value.

5. **Qualifier & Reverse Value Search in Facts**:
   - Updated `lookup_fact` to search `qualifier` and `value` when `subject` alone yields 0 results. Queries like *"Who's our Northwind contact?"* now cleanly match `Tom Okafor (qualifier: Northwind account)`.

6. **Case-Sensitivity in Aggregations & History**:
   - Replaced case-sensitive equality `h.app = ?` with `LOWER(h.app) = LOWER(?)` in `aggregate()` and `server.py`, ensuring filtering for `"slack"` returns all records matching `"Slack"`.
   - Added normalization for `kind` filtering (`factual`, `episodic`, `preference`).

7. **Timeline-Anchored Recency Scoring**:
   - Replaced `now = datetime.now()` in `fuzzy_search` with an anchor relative to the latest timestamp in the user's history. This prevents recency scores from decaying to zero when running against historical datasets in future calendar years.

8. **Accurate Metrics & Parameter Reporting**:
   - Fixed `route_query` in `query/router.py` to report dynamic `model_calls` (accounting for contextualization) and include `contextualized_query`.

9. **FastAPI Lifespan Migration & Deprecation Cleanup**:
   - Replaced deprecated `@app.on_event("startup")` in `server.py` with the modern `lifespan` async context manager.
   - Updated the dictionary delete route to `@app.delete("/api/dictionary/{raw_form:path}")` to safely handle multi-word entries with spaces.

10. **Qualifier Scope Conflict Resolution**:
    - Normalized qualifier comparisons in `resolve_conflicts` (`ingestion/extractor.py`) so memories scoped to different contexts (e.g. Slack vs Notes) do not accidentally supersede one another.

---

## 8. Verifying the Pipeline

To verify the pipeline end-to-end:

```bash
# 1. Check API statistics
python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/api/stats').read().decode())"

# 2. Test a direct question via CLI
python -m eval.spot_check --query "Who's our Northwind contact?"

# 3. Test temporal tracking
python -m eval.spot_check --query "Before the rename, what was Orion called?"

# 4. Run automated LLM-as-judge evaluation
python -m eval.run_eval
```
