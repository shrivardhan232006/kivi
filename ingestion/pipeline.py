"""Ingestion pipeline orchestrator.

Reads corpus.jsonl -> dictionary correction -> insert history ->
LLM extraction -> conflict resolution -> embedding generation.
"""

import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import execute, fetchall, fetchone, commit
from db.migrate import migrate
from ingestion.dictionary import apply_corrections, load_dictionary, auto_populate_from_corpus
from ingestion.extractor import extract_memories, extract_memories_batch, store_memory
from ingestion.embedder import batch_embed_and_store
import config


def load_corpus(corpus_path: str) -> list[dict]:
    """Load corpus JSONL file."""
    records = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def ingest_history(records: list[dict], dictionary: list[dict]) -> int:
    """Insert history records with dictionary correction. Returns count inserted."""
    inserted = 0
    for rec in records:
        # Skip if already ingested
        existing = fetchone("SELECT id FROM history WHERE id = ?", (rec['id'],))
        if existing:
            continue

        formatted = rec.get('llm_formatted_output', rec.get('raw_asr_output', ''))
        canonical = apply_corrections(formatted, dictionary)

        execute(
            """INSERT INTO history (id, raw_text, formatted_text, canonical_text,
               app, timestamp, day_index, style_applied, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rec['id'],
                rec.get('raw_asr_output', ''),
                formatted,
                canonical,
                rec.get('app', ''),
                rec.get('timestamp', ''),
                rec.get('day_index'),
                rec.get('style_applied', ''),
                json.dumps({k: v for k, v in rec.items()
                           if k not in ('id', 'raw_asr_output', 'llm_formatted_output',
                                       'app', 'timestamp', 'day_index', 'style_applied')}),
            )
        )
        inserted += 1

    commit()
    return inserted


def run_extraction(batch_size: int = None) -> tuple[int, int]:
    """Run LLM extraction on all history records not yet processed.
    Returns (memories_created, records_processed)."""
    batch_size = batch_size or config.EXTRACTION_BATCH_SIZE

    # Get history records that haven't been extracted yet
    all_history = fetchall("SELECT * FROM history ORDER BY timestamp ASC")

    # Load from processed_history table
    already_processed = set(r['history_id'] for r in fetchall("SELECT history_id FROM processed_history"))

    # Also include records from existing memories
    existing_sources = fetchall("SELECT DISTINCT source_record_ids FROM memories")
    for row in existing_sources:
        try:
            ids = json.loads(row['source_record_ids'])
            already_processed.update(ids)
        except (json.JSONDecodeError, TypeError):
            pass

    to_process = [r for r in all_history if r['id'] not in already_processed]

    if not to_process:
        print(f"  All {len(all_history)} records already processed.", flush=True)
        return 0, 0

    total_memories = 0
    records_done = 0

    for i in range(0, len(to_process), batch_size):
        batch = to_process[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(to_process) + batch_size - 1) // batch_size

        batch_dicts = [dict(r) for r in batch]

        # Batch extraction via LLM
        batch_memories = 0
        try:
            extracted_map = extract_memories_batch(batch_dicts)
        except Exception as e:
            print(f"  Batch extraction failed ({e}), falling back to individual...", flush=True)
            extracted_map = {}

        for rec in batch_dicts:
            memories = extracted_map.get(rec['id'])
            if memories is None:  # was not in batch or failed
                memories = extract_memories(rec)

            for mem in memories:
                mem_id = store_memory(mem, rec['id'], rec['timestamp'])
                if mem_id:
                    total_memories += 1
                    batch_memories += 1

            execute("INSERT OR IGNORE INTO processed_history (history_id) VALUES (?)", (rec['id'],))
            records_done += 1

        commit()
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} records) -> {batch_memories} memories ({total_memories} new, {records_done}/{len(to_process)} done)...", flush=True)

        # Delay between batches for rate limiting
        if i + batch_size < len(to_process):
            time.sleep(config.INTER_BATCH_DELAY)

    return total_memories, records_done


def run_embedding() -> int:
    """Generate embeddings for all history and memory records. Returns count created."""
    existing_embeddings = set(
        (r['source_type'], r['source_id'])
        for r in fetchall("SELECT source_type, source_id FROM embeddings")
    )

    items = []

    # History embeddings
    history_records = fetchall("SELECT id, canonical_text FROM history")
    for rec in history_records:
        if ('history', rec['id']) not in existing_embeddings:
            items.append({
                'source_type': 'history',
                'source_id': rec['id'],
                'text': rec['canonical_text'],
            })

    # Memory embeddings
    memories = fetchall("SELECT id, subject, predicate, value, qualifier FROM memories")
    for mem in memories:
        if ('memory', mem['id']) not in existing_embeddings:
            text_parts = [mem['subject'], mem['predicate'], mem['value']]
            if mem['qualifier']:
                text_parts.append(mem['qualifier'])
            items.append({
                'source_type': 'memory',
                'source_id': mem['id'],
                'text': ' | '.join(text_parts),
            })

    if not items:
        print("  All items already have embeddings.", flush=True)
        return 0

    print(f"  Embedding {len(items)} items...", flush=True)
    return batch_embed_and_store(items)


def run_pipeline(corpus_path: str):
    """Full ingestion pipeline."""
    start = time.time()

    print("=" * 60)
    print("KIVI SEMANTIC MEMORY - Ingestion Pipeline")
    print("=" * 60)

    # Step 0: Migrate DB
    print("\n[1/5] Database migration...")
    migrate()

    # Step 1: Load corpus
    print(f"\n[2/5] Loading corpus from {corpus_path}...")
    records = load_corpus(corpus_path)
    print(f"  Loaded {len(records)} records")

    # Step 2: Auto-populate dictionary from corpus patterns
    print("\n[3/5] Dictionary setup...")
    added = auto_populate_from_corpus(records)
    dictionary = load_dictionary()
    print(f"  Dictionary: {len(dictionary)} entries ({added} auto-populated from corpus)")

    # Step 3: Ingest history with dictionary correction
    print("\n[4/5] Ingesting history records...")
    inserted = ingest_history(records, dictionary)
    total_history = fetchone("SELECT COUNT(*) as c FROM history")['c']
    print(f"  Inserted {inserted} new records (total: {total_history})")

    # Step 4: LLM extraction
    print("\n[5/5] LLM memory extraction...")
    memories_created, records_processed = run_extraction()
    total_memories = fetchone("SELECT COUNT(*) as c FROM memories")['c']
    print(f"  Created {memories_created} memories from {records_processed} records (total: {total_memories})")

    # Memory breakdown
    for kind in ('factual', 'episodic', 'preference'):
        count = fetchone("SELECT COUNT(*) as c FROM memories WHERE kind = ?", (kind,))['c']
        print(f"    {kind}: {count}")

    # Step 5: Embedding generation
    print("\n[6/6] Generating embeddings...")
    emb_count = run_embedding()
    total_embs = fetchone("SELECT COUNT(*) as c FROM embeddings")['c']
    print(f"  Created {emb_count} new embeddings (total: {total_embs})")

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"  History: {total_history} records")
    print(f"  Memories: {total_memories} extracted")
    print(f"  Embeddings: {total_embs} vectors")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kivi ingestion pipeline")
    parser.add_argument("--corpus", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "corpus.jsonl"))
    args = parser.parse_args()
    run_pipeline(args.corpus)
