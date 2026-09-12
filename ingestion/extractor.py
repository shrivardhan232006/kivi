"""LLM-based memory extraction — the core of the ingestion pipeline.

Every step that needs to understand, extract, or synthesize language is a real
LLM call. No regex/keyword pattern matching for meaning extraction. Ever.
"""

import json
import uuid
from db.database import fetchall, execute, commit
from llm import call_llm_json
import config

EXTRACTION_SYSTEM = """You are a memory extraction system for a voice dictation product.
Your job is to extract structured memories from dictation text.

You extract THREE kinds of memory:
1. FACTUAL: durable facts about people, roles, relationships, projects, organizations, logistics, locations.
2. EPISODIC: specific events or actions — who did what, when, about what topic.
3. PREFERENCE: how the user wants things done — communication style, tool preferences, workflow choices, formatting preferences.

Rules:
- PRESERVE qualifying context as a "qualifier". For example: "async for design reviews" has qualifier "for design reviews". Without the qualifier, it would falsely conflict with "sync for incidents".
- For EPISODIC memories: the subject is who participated, the predicate describes the activity type, the value captures the topic/content, and the qualifier captures the project context.
- Subject should be a proper noun (person name, project name) or "user" for self-references.
- If the dictation is small talk, filler, a to-do reminder with no extractable fact/episode/preference, return an empty array [].
- Do NOT extract vague statements or trivial logistics (e.g., "need to submit expenses") as memories.
- Be precise: extract what is actually stated, do not infer unstated information.
"""

EXTRACTION_PROMPT = """Extract all factual, episodic, and preference memories from this dictation.

Dictation text: "{text}"
Source app: {app}
Timestamp: {timestamp}

Return a JSON array of objects. Each object must have these fields:
- "kind": one of "factual", "episodic", "preference"
- "subject": the person, project, or entity this is about (use "user" for self-references)
- "predicate": the relationship or attribute (e.g., "role", "discussed", "preference_for", "location", "renamed_to")
- "value": the actual claim or fact
- "qualifier": qualifying context that scopes this memory (null if none)
- "confidence": 0.0-1.0 how confident this extraction is

If nothing is extractable, return an empty array: []
"""


def extract_memories(record: dict) -> list[dict]:
    """Extract memories from a single history record using LLM."""
    text = record.get('canonical_text') or record.get('formatted_text', '')
    if not text or len(text.strip()) < 5:
        return []

    prompt = EXTRACTION_PROMPT.format(
        text=text,
        app=record.get('app', 'unknown'),
        timestamp=record.get('timestamp', ''),
    )

    result = call_llm_json(
        prompt=prompt,
        model=config.EXTRACTION_MODEL,
        temperature=config.EXTRACTION_TEMPERATURE,
        system_instruction=EXTRACTION_SYSTEM,
    )

    if not isinstance(result, list):
        result = [result] if isinstance(result, dict) else []

    # Validate and normalize each extraction
    valid_memories = []
    for mem in result:
        if not isinstance(mem, dict):
            continue
        kind = mem.get('kind', '').lower()
        if kind not in ('factual', 'episodic', 'preference'):
            continue
        subject = mem.get('subject', '').strip()
        predicate = mem.get('predicate', '').strip()
        value = mem.get('value', '').strip()
        if not subject or not predicate or not value:
            continue

        valid_memories.append({
            'kind': kind,
            'subject': subject,
            'predicate': predicate,
            'value': value,
            'qualifier': mem.get('qualifier') or None,
            'confidence': min(1.0, max(0.0, float(mem.get('confidence', 0.8)))),
        })

    return valid_memories


BATCH_EXTRACTION_PROMPT = """Extract all factual, episodic, and preference memories from these dictation records.

Records:
{records_json}

Return a JSON array of objects. Each object must have these fields:
- "record_id": the id of the record this memory was extracted from (e.g. "rec_0001")
- "kind": one of "factual", "episodic", "preference"
- "subject": the person, project, or entity this is about (use "user" for self-references)
- "predicate": the relationship or attribute (e.g., "role", "discussed", "preference_for", "location", "renamed_to")
- "value": the actual claim or fact
- "qualifier": qualifying context that scopes this memory (null if none)
- "confidence": 0.0-1.0 how confident this extraction is

If a record is small talk, filler, distractor, or has no extractable memory, do NOT return any memories for it.
If no memories are extractable from any records, return an empty array: []
"""


def extract_memories_batch(records: list[dict]) -> dict[str, list[dict]]:
    """Extract memories from a batch of history records using a single LLM call.
    Returns mapping from record_id -> list of memory dicts.
    """
    if not records:
        return {}

    records_data = [
        {
            'id': r['id'],
            'timestamp': r.get('timestamp', ''),
            'app': r.get('app', 'unknown'),
            'text': r.get('canonical_text') or r.get('formatted_text', ''),
        }
        for r in records
        if (r.get('canonical_text') or r.get('formatted_text', '')).strip()
    ]

    if not records_data:
        return {r['id']: [] for r in records}

    prompt = BATCH_EXTRACTION_PROMPT.format(records_json=json.dumps(records_data, indent=2))

    result = call_llm_json(
        prompt=prompt,
        model=config.EXTRACTION_MODEL,
        temperature=config.EXTRACTION_TEMPERATURE,
        system_instruction=EXTRACTION_SYSTEM,
    )

    if not isinstance(result, list):
        result = [result] if isinstance(result, dict) else []

    mapped: dict[str, list[dict]] = {r['id']: [] for r in records}
    for mem in result:
        if not isinstance(mem, dict):
            continue
        rec_id = mem.get('record_id')
        if not rec_id or rec_id not in mapped:
            continue

        kind = mem.get('kind', '').lower()
        if kind not in ('factual', 'episodic', 'preference'):
            continue
        subject = mem.get('subject', '').strip()
        predicate = mem.get('predicate', '').strip()
        value = mem.get('value', '').strip()
        if not subject or not predicate or not value:
            continue

        mapped[rec_id].append({
            'kind': kind,
            'subject': subject,
            'predicate': predicate,
            'value': value,
            'qualifier': mem.get('qualifier') or None,
            'confidence': min(1.0, max(0.0, float(mem.get('confidence', 0.8)))),
        })

    return mapped


def resolve_conflicts(new_memory: dict, record_id: str, record_timestamp: str) -> str | None:
    """Check for conflicts with existing memories.

    If a conflicting memory exists (same subject+predicate, different value,
    no distinguishing qualifier), mark the old one as superseded.

    Returns the ID of the superseded memory, "duplicate", or None.
    """
    subject = new_memory['subject']
    predicate = new_memory['predicate']
    qualifier = new_memory.get('qualifier')

    # Find existing valid memories with same subject and predicate
    existing = fetchall(
        """SELECT id, value, qualifier, valid_from, source_record_ids FROM memories
           WHERE subject = ? AND predicate = ? AND valid_to IS NULL""",
        (subject, predicate)
    )

    for old in existing:
        # If qualifiers differ, these aren't conflicting — they're scoped differently
        q_new = (qualifier or '').strip().lower()
        q_old = (old['qualifier'] or '').strip().lower()
        if q_new != q_old:
            continue

        # If values are the same, track provenance and skip (duplicate)
        if old['value'].lower().strip() == new_memory['value'].lower().strip():
            try:
                cur_ids = json.loads(old['source_record_ids']) if old['source_record_ids'] else []
            except (json.JSONDecodeError, TypeError):
                cur_ids = []
            if record_id not in cur_ids:
                cur_ids.append(record_id)
                execute("UPDATE memories SET source_record_ids = ? WHERE id = ?", (json.dumps(cur_ids), old['id']))
                commit()
            return "duplicate"

        # Conflict: different value, same scope → supersede the old one
        execute(
            "UPDATE memories SET valid_to = ? WHERE id = ?",
            (record_timestamp, old['id'])
        )
        commit()
        return old['id']

    return None


def store_memory(memory: dict, record_id: str, record_timestamp: str) -> str | None:
    """Store an extracted memory in the database. Returns memory ID or None if duplicate."""
    # Resolve conflicts first
    conflict_result = resolve_conflicts(memory, record_id, record_timestamp)
    if conflict_result == "duplicate":
        return None

    memory_id = f"mem_{uuid.uuid4().hex[:12]}"

    execute(
        """INSERT INTO memories (id, kind, subject, predicate, value, qualifier,
           valid_from, valid_to, source_record_ids, confidence)
           VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)""",
        (
            memory_id,
            memory['kind'],
            memory['subject'],
            memory['predicate'],
            memory['value'],
            memory.get('qualifier'),
            record_timestamp,
            json.dumps([record_id]),
            memory.get('confidence', 0.8),
        )
    )
    commit()
    return memory_id
