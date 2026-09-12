"""Embedding generation for history records and memories."""

import json
import uuid
import time
import numpy as np
from db.database import execute, fetchall, commit
from llm import generate_embeddings


def embed_and_store(source_type: str, source_id: str, text: str) -> str | None:
    """Generate embedding for a text and store it. Returns embedding ID."""
    if not text or len(text.strip()) < 3:
        return None

    # Check if already embedded
    existing = fetchall(
        "SELECT id FROM embeddings WHERE source_type = ? AND source_id = ?",
        (source_type, source_id)
    )
    if existing:
        return existing[0]['id']

    embeddings = generate_embeddings([text])
    if not embeddings:
        return None

    emb_id = f"emb_{uuid.uuid4().hex[:12]}"
    emb_blob = np.array(embeddings[0], dtype=np.float32).tobytes()

    execute(
        "INSERT INTO embeddings (id, source_type, source_id, embedding, text_content) VALUES (?, ?, ?, ?, ?)",
        (emb_id, source_type, source_id, emb_blob, text)
    )
    commit()
    return emb_id


def batch_embed_and_store(items: list[dict], chunk_size: int = 50) -> int:
    """Batch embed and store. Each item: {source_type, source_id, text}.
    Embeds and commits in chunks for resilience.
    Returns count of embeddings created."""
    # Filter out already-embedded and empty texts
    existing = set(
        (r['source_type'], r['source_id'])
        for r in fetchall("SELECT source_type, source_id FROM embeddings")
    )

    to_embed = [
        item for item in items
        if item.get('text') and len(item['text'].strip()) >= 3
        and (item['source_type'], item['source_id']) not in existing
    ]

    if not to_embed:
        print("  All items already embedded.", flush=True)
        return 0

    total_created = 0
    total_chunks = (len(to_embed) + chunk_size - 1) // chunk_size

    for i in range(0, len(to_embed), chunk_size):
        chunk = to_embed[i:i + chunk_size]
        chunk_num = i // chunk_size + 1
        texts = [item['text'] for item in chunk]

        embeddings = generate_embeddings(texts)
        if len(embeddings) != len(chunk):
            print(f"  Warning: chunk {chunk_num} got {len(embeddings)}/{len(chunk)} embeddings", flush=True)

        for item, emb in zip(chunk, embeddings):
            emb_id = f"emb_{uuid.uuid4().hex[:12]}"
            emb_blob = np.array(emb, dtype=np.float32).tobytes()
            execute(
                "INSERT INTO embeddings (id, source_type, source_id, embedding, text_content) VALUES (?, ?, ?, ?, ?)",
                (emb_id, item['source_type'], item['source_id'], emb_blob, item['text'])
            )
            total_created += 1

        commit()
        print(f"  Embedded chunk {chunk_num}/{total_chunks} ({total_created}/{len(to_embed)})...", flush=True)
        time.sleep(1.0)

    return total_created


def load_all_embeddings(source_type: str | None = None) -> list[dict]:
    """Load all embeddings from DB. Returns [{source_type, source_id, embedding(np), text_content}]."""
    if source_type:
        rows = fetchall(
            "SELECT source_type, source_id, embedding, text_content FROM embeddings WHERE source_type = ?",
            (source_type,)
        )
    else:
        rows = fetchall("SELECT source_type, source_id, embedding, text_content FROM embeddings")

    result = []
    for row in rows:
        emb = np.frombuffer(row['embedding'], dtype=np.float32)
        result.append({
            'source_type': row['source_type'],
            'source_id': row['source_id'],
            'embedding': emb,
            'text_content': row['text_content'],
        })
    return result


def cosine_similarity(query_emb: np.ndarray, corpus_embs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and all corpus embeddings."""
    if len(corpus_embs) == 0:
        return np.array([])
    # Normalize
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    corpus_norms = corpus_embs / (np.linalg.norm(corpus_embs, axis=1, keepdims=True) + 1e-10)
    return corpus_norms @ query_norm


def semantic_search(query_text: str, top_k: int = 20, source_type: str | None = None) -> list[dict]:
    """Search embeddings by semantic similarity. Returns [{source_type, source_id, text_content, score}]."""
    query_embs = generate_embeddings([query_text])
    if not query_embs:
        return []

    query_emb = np.array(query_embs[0], dtype=np.float32)
    all_embs = load_all_embeddings(source_type)
    if not all_embs:
        return []

    corpus_matrix = np.stack([e['embedding'] for e in all_embs])
    scores = cosine_similarity(query_emb, corpus_matrix)

    # Get top-K
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            'source_type': all_embs[idx]['source_type'],
            'source_id': all_embs[idx]['source_id'],
            'text_content': all_embs[idx]['text_content'],
            'score': float(scores[idx]),
        })
    return results
