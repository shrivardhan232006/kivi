"""Deterministic tool implementations for the query router.

Each tool is SQL-backed and deterministic:
- aggregate() -> SELECT DISTINCT, never similarity ranking
- lookup_fact() -> structured SQL lookup
- get_valid_at() -> bi-temporal SQL resolution
- fuzzy_search() -> three-signal fusion (semantic + BM25 + recency)
"""

import json
import math
import numpy as np
from datetime import datetime
from db.database import fetchall, fetchone
from ingestion.embedder import semantic_search, load_all_embeddings, cosine_similarity, generate_embeddings
from query.bm25 import BM25
import config


def aggregate(filter_field: str, filter_value: str) -> dict:
    """List ALL items matching a filter. Pure SQL — complete set, no ranking.

    filter_field: 'app', 'kind', 'predicate', 'subject'
    filter_value: the value to match
    """
    clean_val = filter_value.strip()

    if filter_field == 'app':
        # Join with history to filter by app (case-insensitive)
        rows = fetchall(
            """SELECT DISTINCT m.id, m.subject, m.predicate, m.value, m.kind, m.qualifier, m.source_record_ids
               FROM memories m, history h
               WHERE m.valid_to IS NULL
               AND m.source_record_ids LIKE '%' || h.id || '%'
               AND LOWER(h.app) = LOWER(?)
               ORDER BY m.subject""",
            (clean_val,)
        )
    elif filter_field == 'kind':
        # Normalize kind (e.g. facts -> factual, preferences -> preference)
        norm_kind = clean_val.lower()
        if norm_kind.startswith('fact'):
            norm_kind = 'factual'
        elif norm_kind.startswith('pref'):
            norm_kind = 'preference'
        elif norm_kind.startswith('episod'):
            norm_kind = 'episodic'

        rows = fetchall(
            """SELECT DISTINCT id, subject, predicate, value, kind, qualifier, source_record_ids
               FROM memories WHERE LOWER(kind) = LOWER(?) AND valid_to IS NULL
               ORDER BY subject""",
            (norm_kind,)
        )
    elif filter_field == 'predicate':
        rows = fetchall(
            """SELECT DISTINCT id, subject, predicate, value, kind, qualifier, source_record_ids
               FROM memories WHERE LOWER(predicate) LIKE LOWER(?) AND valid_to IS NULL
               ORDER BY subject""",
            (f"%{clean_val}%",)
        )
    elif filter_field == 'subject':
        rows = fetchall(
            """SELECT DISTINCT id, subject, predicate, value, kind, qualifier, source_record_ids
               FROM memories WHERE LOWER(subject) LIKE LOWER(?) AND valid_to IS NULL
               ORDER BY predicate""",
            (f"%{clean_val}%",)
        )
    else:
        rows = []

    results = [dict(r) for r in rows]

    # Also fetch source history records for provenance and context
    for mem in results:
        try:
            source_ids = json.loads(mem.get('source_record_ids', '[]'))
            if source_ids:
                placeholders = ','.join(['?'] * len(source_ids))
                sources = fetchall(
                    f"SELECT id, canonical_text, app, timestamp FROM history WHERE id IN ({placeholders})",
                    tuple(source_ids)
                )
                mem['source_records'] = [dict(s) for s in sources]
        except (json.JSONDecodeError, TypeError):
            mem['source_records'] = []

    return {
        'tool': 'aggregate',
        'filter_field': filter_field,
        'filter_value': filter_value,
        'count': len(results),
        'results': results,
    }


def lookup_fact(subject: str, predicate: str | None = None) -> dict:
    """Look up facts about a specific subject. Structured SQL lookup."""
    from ingestion.dictionary import apply_corrections

    orig_subject = subject or ""
    sub_clean = orig_subject.strip()

    # Normalize self-references and 'my <thing>'
    if sub_clean.lower() in ('i', 'me', 'my', 'myself', 'self'):
        subject = 'user'
    elif sub_clean.lower().startswith('my '):
        remainder = sub_clean[3:].strip()
        if not predicate:
            predicate = remainder
        subject = 'user'
    else:
        subject = sub_clean

    # Apply dictionary corrections (e.g. Raman -> Priya Reddy)
    corrected_sub = apply_corrections(subject)

    rows = []
    if predicate:
        pred_clean = predicate.strip()
        # Look up matching subject, rename alias, and predicate/qualifier/value
        rows = fetchall(
            """SELECT * FROM memories
               WHERE (subject LIKE ? OR subject LIKE ? OR (predicate LIKE '%rename%' AND value LIKE ?))
               AND (predicate LIKE ? OR qualifier LIKE ? OR value LIKE ?)
               AND valid_to IS NULL
               ORDER BY confidence DESC""",
            (f"%{subject}%", f"%{corrected_sub}%", f"%{subject}%",
             f"%{pred_clean}%", f"%{pred_clean}%", f"%{pred_clean}%")
        )

    # Fallback 1: match subject or rename alias
    if not rows:
        rows = fetchall(
            """SELECT * FROM memories
               WHERE (subject LIKE ? OR subject LIKE ? OR (predicate LIKE '%rename%' AND value LIKE ?))
               AND valid_to IS NULL
               ORDER BY confidence DESC""",
            (f"%{subject}%", f"%{corrected_sub}%", f"%{subject}%")
        )

    # Fallback 2: search value or qualifier across memories (e.g. "Northwind" in qualifier)
    if not rows and subject != 'user' and len(subject) >= 3:
        rows = fetchall(
            """SELECT * FROM memories
               WHERE (value LIKE ? OR qualifier LIKE ?)
               AND valid_to IS NULL
               ORDER BY confidence DESC""",
            (f"%{subject}%", f"%{subject}%")
        )

    results = [dict(r) for r in rows]

    # Also fetch source history records for provenance
    for mem in results:
        try:
            source_ids = json.loads(mem.get('source_record_ids', '[]'))
            if source_ids:
                placeholders = ','.join(['?'] * len(source_ids))
                sources = fetchall(
                    f"SELECT id, canonical_text, app, timestamp FROM history WHERE id IN ({placeholders})",
                    tuple(source_ids)
                )
                mem['source_records'] = [dict(s) for s in sources]
        except (json.JSONDecodeError, TypeError):
            mem['source_records'] = []

    return {
        'tool': 'lookup_fact',
        'subject': subject,
        'predicate': predicate,
        'count': len(results),
        'results': results,
    }


def get_valid_at(subject: str, predicate: str | None = None, timestamp: str = "now") -> dict:
    """Resolve what was true at a specific point in time. Bi-temporal SQL."""
    from ingestion.dictionary import apply_corrections

    orig_subject = subject or ""
    sub_clean = orig_subject.strip()

    if sub_clean.lower() in ('i', 'me', 'my', 'myself', 'self'):
        subject = 'user'
    elif sub_clean.lower().startswith('my '):
        remainder = sub_clean[3:].strip()
        if not predicate:
            predicate = remainder
        subject = 'user'
    else:
        subject = sub_clean

    corrected_sub = apply_corrections(subject)

    rows = []
    if timestamp == "now":
        # Current state only
        if predicate:
            pred_clean = predicate.strip()
            rows = fetchall(
                """SELECT * FROM memories
                   WHERE (subject LIKE ? OR subject LIKE ? OR (predicate LIKE '%rename%' AND value LIKE ?))
                   AND (predicate LIKE ? OR qualifier LIKE ? OR value LIKE ?)
                   AND valid_to IS NULL
                   ORDER BY valid_from DESC""",
                (f"%{subject}%", f"%{corrected_sub}%", f"%{subject}%",
                 f"%{pred_clean}%", f"%{pred_clean}%", f"%{pred_clean}%")
            )
        if not rows:
            rows = fetchall(
                """SELECT * FROM memories
                   WHERE (subject LIKE ? OR subject LIKE ? OR (predicate LIKE '%rename%' AND value LIKE ?))
                   AND valid_to IS NULL
                   ORDER BY valid_from DESC""",
                (f"%{subject}%", f"%{corrected_sub}%", f"%{subject}%")
            )
        if not rows and subject != 'user' and len(subject) >= 3:
            rows = fetchall(
                """SELECT * FROM memories
                   WHERE (value LIKE ? OR qualifier LIKE ?)
                   AND valid_to IS NULL
                   ORDER BY valid_from DESC""",
                (f"%{subject}%", f"%{subject}%")
            )
    elif timestamp == "all":
        # Full history
        if predicate:
            pred_clean = predicate.strip()
            rows = fetchall(
                """SELECT * FROM memories
                   WHERE (subject LIKE ? OR subject LIKE ? OR (predicate LIKE '%rename%' AND value LIKE ?))
                   AND (predicate LIKE ? OR qualifier LIKE ? OR value LIKE ?)
                   ORDER BY valid_from ASC""",
                (f"%{subject}%", f"%{corrected_sub}%", f"%{subject}%",
                 f"%{pred_clean}%", f"%{pred_clean}%", f"%{pred_clean}%")
            )
        if not rows:
            rows = fetchall(
                """SELECT * FROM memories
                   WHERE (subject LIKE ? OR subject LIKE ? OR (predicate LIKE '%rename%' AND value LIKE ?))
                   ORDER BY valid_from ASC""",
                (f"%{subject}%", f"%{corrected_sub}%", f"%{subject}%")
            )
        if not rows and subject != 'user' and len(subject) >= 3:
            rows = fetchall(
                """SELECT * FROM memories
                   WHERE (value LIKE ? OR qualifier LIKE ?)
                   ORDER BY valid_from ASC""",
                (f"%{subject}%", f"%{subject}%")
            )
    else:
        # Specific timestamp
        if predicate:
            pred_clean = predicate.strip()
            rows = fetchall(
                """SELECT * FROM memories
                   WHERE (subject LIKE ? OR subject LIKE ? OR (predicate LIKE '%rename%' AND value LIKE ?))
                   AND (predicate LIKE ? OR qualifier LIKE ? OR value LIKE ?)
                   AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)
                   ORDER BY valid_from DESC""",
                (f"%{subject}%", f"%{corrected_sub}%", f"%{subject}%",
                 f"%{pred_clean}%", f"%{pred_clean}%", f"%{pred_clean}%",
                 timestamp, timestamp)
            )
        if not rows:
            rows = fetchall(
                """SELECT * FROM memories
                   WHERE (subject LIKE ? OR subject LIKE ? OR (predicate LIKE '%rename%' AND value LIKE ?))
                   AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)
                   ORDER BY valid_from DESC""",
                (f"%{subject}%", f"%{corrected_sub}%", f"%{subject}%",
                 timestamp, timestamp)
            )

    results = [dict(r) for r in rows]

    # Fetch provenance
    for mem in results:
        try:
            source_ids = json.loads(mem.get('source_record_ids', '[]'))
            if source_ids:
                placeholders = ','.join(['?'] * len(source_ids))
                sources = fetchall(
                    f"SELECT id, canonical_text, app, timestamp FROM history WHERE id IN ({placeholders})",
                    tuple(source_ids)
                )
                mem['source_records'] = [dict(s) for s in sources]
        except (json.JSONDecodeError, TypeError):
            mem['source_records'] = []

    return {
        'tool': 'get_valid_at',
        'subject': subject,
        'predicate': predicate,
        'timestamp': timestamp,
        'count': len(results),
        'results': results,
    }


def fuzzy_search(query: str, top_k: int = 10) -> dict:
    """Multi-signal fuzzy search combining semantic, BM25, and recency.

    This is the fallback for episodic/multi-session/ambiguous queries.
    """
    # Build candidate pool from both memories and history
    all_memories = fetchall(
        "SELECT id, kind, subject, predicate, value, qualifier, valid_from, source_record_ids, confidence FROM memories"
    )
    all_history = fetchall(
        "SELECT id, canonical_text, app, timestamp, day_index FROM history"
    )

    if not all_memories and not all_history:
        return {'tool': 'fuzzy_search', 'query': query, 'count': 0, 'results': []}

    # --- Signal 1: Semantic similarity ---
    semantic_results = semantic_search(query, top_k=top_k * 2)
    semantic_scores: dict[str, float] = {}
    for r in semantic_results:
        key = f"{r['source_type']}:{r['source_id']}"
        semantic_scores[key] = r['score']

    # --- Signal 2: BM25 lexical scoring ---
    bm25 = BM25()
    bm25_docs = []
    for mem in all_memories:
        text = f"{mem['subject']} {mem['predicate']} {mem['value']}"
        if mem['qualifier']:
            text += f" {mem['qualifier']}"
        bm25_docs.append({'id': f"memory:{mem['id']}", 'text': text})
    for hist in all_history:
        bm25_docs.append({'id': f"history:{hist['id']}", 'text': hist['canonical_text']})

    bm25.index(bm25_docs)
    bm25_results = bm25.score(query)
    bm25_scores: dict[str, float] = {}
    max_bm25 = max((s for _, _, s in bm25_results), default=1.0) or 1.0
    for doc_id, _, score in bm25_results:
        bm25_scores[doc_id] = score / max_bm25  # Normalize to [0,1]

    # --- Signal 3: Temporal recency (anchored to timeline reference) ---
    hist_times = []
    for h in all_history:
        try:
            hist_times.append(datetime.fromisoformat(h['timestamp'].replace('Z', '+00:00')).replace(tzinfo=None))
        except (ValueError, TypeError):
            pass
    ref_time = max(hist_times) if hist_times else datetime.now()

    recency_scores: dict[str, float] = {}
    half_life = config.RECENCY_HALF_LIFE_DAYS

    for mem in all_memories:
        try:
            mem_time = datetime.fromisoformat(mem['valid_from'].replace('Z', '+00:00')).replace(tzinfo=None)
            days_ago = max(0.0, (ref_time - mem_time).total_seconds() / 86400)
            recency_scores[f"memory:{mem['id']}"] = math.exp(-0.693 * days_ago / half_life)
        except (ValueError, TypeError):
            recency_scores[f"memory:{mem['id']}"] = 0.5

    for hist in all_history:
        try:
            hist_time = datetime.fromisoformat(hist['timestamp'].replace('Z', '+00:00')).replace(tzinfo=None)
            days_ago = max(0.0, (ref_time - hist_time).total_seconds() / 86400)
            recency_scores[f"history:{hist['id']}"] = math.exp(-0.693 * days_ago / half_life)
        except (ValueError, TypeError):
            recency_scores[f"history:{hist['id']}"] = 0.5

    # --- Fusion ---
    all_keys = set()
    all_keys.update(semantic_scores.keys())
    all_keys.update(bm25_scores.keys())

    fused: list[tuple[str, float]] = []
    for key in all_keys:
        sem = semantic_scores.get(key, 0.0)
        lex = bm25_scores.get(key, 0.0)
        rec = recency_scores.get(key, 0.5)
        score = (config.WEIGHT_SEMANTIC * sem +
                 config.WEIGHT_LEXICAL * lex +
                 config.WEIGHT_RECENCY * rec)
        fused.append((key, score))

    fused.sort(key=lambda x: x[1], reverse=True)
    top_results = fused[:top_k]

    # Resolve to full records
    results = []
    for key, score in top_results:
        source_type, source_id = key.split(':', 1)
        if source_type == 'memory':
            mem = fetchone("SELECT * FROM memories WHERE id = ?", (source_id,))
            if mem:
                record = dict(mem)
                record['_score'] = score
                record['_source_type'] = 'memory'
                # Fetch provenance
                try:
                    src_ids = json.loads(mem['source_record_ids'])
                    if src_ids:
                        placeholders = ','.join(['?'] * len(src_ids))
                        sources = fetchall(
                            f"SELECT id, canonical_text, app, timestamp FROM history WHERE id IN ({placeholders})",
                            tuple(src_ids)
                        )
                        record['source_records'] = [dict(s) for s in sources]
                except (json.JSONDecodeError, TypeError):
                    record['source_records'] = []
                results.append(record)
        else:
            hist = fetchone("SELECT * FROM history WHERE id = ?", (source_id,))
            if hist:
                record = dict(hist)
                record['_score'] = score
                record['_source_type'] = 'history'
                results.append(record)

    return {
        'tool': 'fuzzy_search',
        'query': query,
        'count': len(results),
        'results': results,
    }


def update_dictionary(raw_form: str, corrected_form: str) -> dict:
    """Add or update a vocabulary correction rule in the dictionary and update memories."""
    from ingestion.dictionary import add_entry, _deduplicate_consecutive_tokens
    import re
    from db.database import execute, commit

    raw_clean = raw_form.strip().lower()
    corrected_clean = corrected_form.strip()

    # 1. Update dictionary table
    add_entry(raw_clean, corrected_clean)

    # 2. Update memory records matching raw form in subject or value
    pattern = rf'\b{re.escape(raw_clean)}\b'
    memories = fetchall("SELECT id, subject, value FROM memories WHERE valid_to IS NULL")
    updated_count = 0
    for m in memories:
        sub = m['subject'] or ''
        val = m['value'] or ''
        new_sub = re.sub(pattern, corrected_clean, sub, flags=re.IGNORECASE)
        new_val = re.sub(pattern, corrected_clean, val, flags=re.IGNORECASE)
        # Deduplicate consecutive tokens to prevent "Priya Priya Reddy"
        new_sub = _deduplicate_consecutive_tokens(new_sub)
        new_val = _deduplicate_consecutive_tokens(new_val)
        if new_sub != sub or new_val != val:
            execute("UPDATE memories SET subject = ?, value = ? WHERE id = ?", (new_sub, new_val, m['id']))
            updated_count += 1

    if updated_count > 0:
        commit()

    return {
        'tool': 'update_dictionary',
        'raw_form': raw_clean,
        'corrected_form': corrected_clean,
        'memories_updated': updated_count,
        'count': 1,
        'results': [{
            'action': 'dictionary_updated',
            'raw_form': raw_clean,
            'corrected_form': corrected_clean,
            'memories_updated': updated_count,
        }],
    }
