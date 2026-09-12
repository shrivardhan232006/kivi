"""Kivi Memory — FastAPI Server.

Serves the Hey Kivi API and the web UI.
"""

import json
import os
import sys
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db.database import fetchall, fetchone, execute, commit, get_db
from db.migrate import migrate
from query.router import route_query
from query.generator import generate_answer, format_context
from query.verifier import verify_answer

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    migrate()
    yield

app = FastAPI(title="Kivi Memory", version="1.0.0", lifespan=lifespan)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Session history buffer for server-side conversation state fallback
_session_history: dict[str, list[dict]] = {}

@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# --- Models ---
class AskRequest(BaseModel):
    question: str
    history: list[dict] | None = None

class ForgetRequest(BaseModel):
    text: str

class DictionaryEntry(BaseModel):
    raw_form: str
    corrected_form: str


# --- UI ---
@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Kivi Memory</h1><p>Static files not found.</p>"


# --- Hey Kivi Ask ---
@app.post("/api/ask")
async def ask(req: AskRequest, request: Request):
    start = time.time()
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Question is required")

    client_ip = request.client.host if request.client else "default"

    # Use client-provided history if present; otherwise fallback to server session buffer
    if req.history:
        history = req.history
    else:
        history = _session_history.get(client_ip, [])

    # Step 1: Route query to appropriate tool (with conversation context if available)
    route_result = route_query(question, conversation_history=history)
    tool_name = route_result['tool_name']
    tool_args = route_result['tool_args']
    tool_result = route_result['results']
    model_calls = route_result.get('model_calls', 0)

    # Step 2: Generate answer from retrieved context and conversation history
    gen_result = generate_answer(question, tool_result, conversation_history=history)
    answer = gen_result['answer']
    model_calls += gen_result['model_calls']

    # Step 3: Verify answer
    context_text = format_context(tool_result)
    ver_result = verify_answer(question, answer, context_text, conversation_history=history)
    final_answer = ver_result['verified_answer']
    model_calls += 1  # verification call

    # Update server session buffer
    if client_ip not in _session_history:
        _session_history[client_ip] = []
    _session_history[client_ip].append({'role': 'user', 'content': question})
    _session_history[client_ip].append({'role': 'assistant', 'content': final_answer})
    if len(_session_history[client_ip]) > 8:
        _session_history[client_ip] = _session_history[client_ip][-8:]

    latency_ms = (time.time() - start) * 1000

    # Log the query
    log_id = f"qlog_{uuid.uuid4().hex[:12]}"
    execute(
        """INSERT INTO query_logs (id, question, tool_name, tool_args, router_reasoning,
           retrieved_memory_ids, retrieved_context, generated_answer,
           verification_result, verification_detail, latency_ms, model_calls)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            log_id,
            question,
            tool_name,
            json.dumps(tool_args),
            route_result.get('reasoning', ''),
            json.dumps(gen_result.get('source_record_ids', [])),
            context_text[:5000],  # Truncate for storage
            final_answer,
            'pass' if ver_result['passed'] else 'fail',
            json.dumps(ver_result.get('flagged_claims', [])),
            latency_ms,
            model_calls,
        )
    )
    commit()

    return {
        'answer': final_answer,
        'tool_name': tool_name,
        'tool_args': tool_args,
        'reasoning': route_result.get('reasoning', ''),
        'source_record_ids': gen_result.get('source_record_ids', []),
        'verification': {
            'passed': ver_result['passed'],
            'flagged_claims': ver_result.get('flagged_claims', []),
        },
        'fast_path': route_result.get('fast_path', False),
        'latency_ms': round(latency_ms, 1),
        'model_calls': model_calls,
    }


@app.post("/api/clear-chat")
async def clear_chat_endpoint(request: Request):
    client_ip = request.client.host if request.client else "default"
    _session_history[client_ip] = []
    return {"status": "ok"}


# --- Memory Browser ---
@app.get("/api/memories")
async def list_memories(kind: str = None, subject: str = None):
    """Get memories grouped by subject."""
    conditions = ["valid_to IS NULL"]
    params = []
    if kind:
        conditions.append("LOWER(kind) = LOWER(?)")
        params.append(kind)
    if subject:
        conditions.append("subject LIKE ?")
        params.append(f"%{subject}%")

    where = " AND ".join(conditions)
    memories = fetchall(
        f"""SELECT * FROM memories WHERE {where} ORDER BY subject, kind, valid_from DESC""",
        tuple(params)
    )

    # Group by subject
    groups = {}
    for mem in memories:
        subj = mem['subject']
        if subj not in groups:
            groups[subj] = {
                'subject': subj,
                'items': [],
                'kinds': set(),
                'last_updated': mem['valid_from'],
                'count': 0,
            }
        groups[subj]['items'].append(dict(mem))
        groups[subj]['kinds'].add(mem['kind'])
        groups[subj]['count'] += 1
        if mem['valid_from'] > groups[subj]['last_updated']:
            groups[subj]['last_updated'] = mem['valid_from']

    # Convert sets to lists for JSON
    result = []
    for g in sorted(groups.values(), key=lambda x: x['last_updated'], reverse=True):
        g['kinds'] = list(g['kinds'])
        result.append(g)

    return {'groups': result, 'total': len(memories)}


@app.get("/api/memories/{memory_id}")
async def get_memory(memory_id: str):
    mem = fetchone("SELECT * FROM memories WHERE id = ?", (memory_id,))
    if not mem:
        raise HTTPException(404, "Memory not found")
    result = dict(mem)
    # Fetch provenance
    try:
        source_ids = json.loads(mem['source_record_ids'])
        if source_ids:
            placeholders = ','.join(['?'] * len(source_ids))
            sources = fetchall(
                f"SELECT * FROM history WHERE id IN ({placeholders})",
                tuple(source_ids)
            )
            result['source_records'] = [dict(s) for s in sources]
    except (json.JSONDecodeError, TypeError):
        result['source_records'] = []
    return result


@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: str):
    mem = fetchone("SELECT id FROM memories WHERE id = ?", (memory_id,))
    if not mem:
        raise HTTPException(404, "Memory not found")
    execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    execute("DELETE FROM embeddings WHERE source_type = 'memory' AND source_id = ?", (memory_id,))
    commit()
    return {'deleted': memory_id}


@app.post("/api/forget")
async def forget(req: ForgetRequest):
    """Conversational 'forget that' — delete memories matching text."""
    text = req.text.strip().lower()
    if not text:
        raise HTTPException(400, "Text is required")

    # Find matching memories
    memories = fetchall(
        """SELECT id, subject, value FROM memories
           WHERE (LOWER(subject) LIKE ? OR LOWER(value) LIKE ?)
           AND valid_to IS NULL""",
        (f"%{text}%", f"%{text}%")
    )

    deleted = []
    for mem in memories:
        execute("DELETE FROM memories WHERE id = ?", (mem['id'],))
        execute("DELETE FROM embeddings WHERE source_type = 'memory' AND source_id = ?", (mem['id'],))
        deleted.append({'id': mem['id'], 'subject': mem['subject'], 'value': mem['value']})

    commit()
    return {'deleted': deleted, 'count': len(deleted)}


# --- History ---
@app.get("/api/history")
async def list_history(app_filter: str = None, search: str = None, page: int = 1, per_page: int = 50):
    conditions = []
    params = []
    if app_filter:
        conditions.append("LOWER(app) = LOWER(?)")
        params.append(app_filter)
    if search:
        conditions.append("canonical_text LIKE ?")
        params.append(f"%{search}%")

    where = " AND ".join(conditions) if conditions else "1=1"
    total = fetchone(f"SELECT COUNT(*) as c FROM history WHERE {where}", tuple(params))['c']
    offset = (page - 1) * per_page

    records = fetchall(
        f"SELECT id, canonical_text, app, timestamp, day_index FROM history WHERE {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
        tuple(params) + (per_page, offset)
    )
    return {
        'records': [dict(r) for r in records],
        'total': total,
        'page': page,
        'pages': (total + per_page - 1) // per_page,
    }


@app.get("/api/history/{record_id}")
async def get_history_record(record_id: str):
    rec = fetchone("SELECT * FROM history WHERE id = ?", (record_id,))
    if not rec:
        raise HTTPException(404, "Record not found")
    result = dict(rec)
    # Find memories derived from this record
    memories = fetchall(
        "SELECT * FROM memories WHERE source_record_ids LIKE ?",
        (f"%{record_id}%",)
    )
    result['derived_memories'] = [dict(m) for m in memories]
    return result


# --- Dictionary ---
@app.get("/api/dictionary")
async def list_dictionary():
    entries = fetchall("SELECT * FROM dictionary ORDER BY raw_form")
    return {'entries': [dict(e) for e in entries]}


@app.post("/api/dictionary")
async def add_dictionary_entry(entry: DictionaryEntry):
    from ingestion.dictionary import add_entry
    add_entry(entry.raw_form, entry.corrected_form)
    return {'added': entry.raw_form, 'corrected_to': entry.corrected_form}


@app.delete("/api/dictionary/{raw_form:path}")
async def delete_dictionary_entry(raw_form: str):
    from ingestion.dictionary import delete_entry
    delete_entry(raw_form)
    return {'deleted': raw_form}


@app.get("/api/dictionary/suggestions")
async def list_suggestions():
    suggestions = fetchall(
        "SELECT * FROM dictionary_suggestions WHERE status = 'pending' ORDER BY occurrence_count DESC"
    )
    return {'suggestions': [dict(s) for s in suggestions]}


@app.post("/api/dictionary/suggestions/{suggestion_id}/{action}")
async def handle_suggestion(suggestion_id: int, action: str):
    if action not in ('accept', 'dismiss'):
        raise HTTPException(400, "Action must be 'accept' or 'dismiss'")

    suggestion = fetchone("SELECT * FROM dictionary_suggestions WHERE id = ?", (suggestion_id,))
    if not suggestion:
        raise HTTPException(404, "Suggestion not found")

    if action == 'accept':
        from ingestion.dictionary import add_entry
        add_entry(suggestion['raw_form'], suggestion['corrected_form'])
        execute("UPDATE dictionary_suggestions SET status = 'accepted' WHERE id = ?", (suggestion_id,))
    else:
        execute("UPDATE dictionary_suggestions SET status = 'dismissed' WHERE id = ?", (suggestion_id,))

    commit()
    return {'suggestion_id': suggestion_id, 'action': action}


# --- Stats ---
@app.get("/api/stats")
async def get_stats():
    return {
        'history_count': fetchone("SELECT COUNT(*) as c FROM history")['c'],
        'memory_count': fetchone("SELECT COUNT(*) as c FROM memories")['c'],
        'memory_valid_count': fetchone("SELECT COUNT(*) as c FROM memories WHERE valid_to IS NULL")['c'],
        'embedding_count': fetchone("SELECT COUNT(*) as c FROM embeddings")['c'],
        'dictionary_count': fetchone("SELECT COUNT(*) as c FROM dictionary")['c'],
        'pending_suggestions': fetchone("SELECT COUNT(*) as c FROM dictionary_suggestions WHERE status = 'pending'")['c'],
        'query_log_count': fetchone("SELECT COUNT(*) as c FROM query_logs")['c'],
        'memory_by_kind': {
            'factual': fetchone("SELECT COUNT(*) as c FROM memories WHERE kind = 'factual' AND valid_to IS NULL")['c'],
            'episodic': fetchone("SELECT COUNT(*) as c FROM memories WHERE kind = 'episodic' AND valid_to IS NULL")['c'],
            'preference': fetchone("SELECT COUNT(*) as c FROM memories WHERE kind = 'preference' AND valid_to IS NULL")['c'],
        },
        'apps': [r['app'] for r in fetchall("SELECT DISTINCT app FROM history ORDER BY app")],
    }


# --- Query Logs ---
@app.get("/api/query-logs")
async def list_query_logs(limit: int = 50):
    logs = fetchall(
        "SELECT * FROM query_logs ORDER BY created_at DESC LIMIT ?",
        (limit,)
    )
    return {'logs': [dict(l) for l in logs]}


if __name__ == "__main__":
    import uvicorn
    import config as cfg
    print(f"Starting Kivi Memory server at http://{cfg.HOST}:{cfg.PORT}")
    uvicorn.run(app, host=cfg.HOST, port=cfg.PORT)
