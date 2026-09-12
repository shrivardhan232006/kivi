"""LLM function-calling router.

The LLM sees the user's question and decides which deterministic tool to invoke
and with what arguments. Tools themselves are fixed and SQL-backed.

Fast-path bypass: if the question contains an exact entity name from the
memories table, skip the LLM and call lookup_fact() directly.
"""

import json
import time
from db.database import fetchall
from llm import call_llm_with_tools
from query.tools import aggregate, lookup_fact, get_valid_at, fuzzy_search, update_dictionary
import config

# Tool declarations for Gemini function calling
TOOL_DECLARATIONS = [
    {
        "name": "lookup_fact",
        "description": (
            "Look up specific facts about a person, project, or topic. "
            "Use for questions like 'who is X', 'what is X's role', "
            "'what do you know about X', 'what's my preference for Y'. "
            "For user's own preferences, role, or info, use subject 'user'."
        ),
        "parameters": {
            "subject": {
                "type": "STRING",
                "description": "The person, project, or entity to look up (use 'user' for self-references like I/me/my)",
                "required": True,
            },
            "predicate": {
                "type": "STRING",
                "description": "The specific attribute to look up (e.g., 'role', 'preference_for', 'location', 'communication_style'). Leave empty to get all facts about the subject.",
                "required": False,
            },
        },
    },
    {
        "name": "aggregate",
        "description": (
            "List ALL items matching a filter. Use for 'list everyone', "
            "'who are all the people', 'what are all my projects', "
            "'who do I talk to on Slack'. Returns the COMPLETE set, not a ranked subset."
        ),
        "parameters": {
            "filter_field": {
                "type": "STRING",
                "description": "Field to filter on: 'app' (source application), 'kind' (factual/episodic/preference), 'predicate' (attribute type), 'subject' (entity name)",
                "enum": ["app", "kind", "predicate", "subject"],
                "required": True,
            },
            "filter_value": {
                "type": "STRING",
                "description": "Value to match (e.g., 'Slack', 'factual', 'role')",
                "required": True,
            },
        },
    },
    {
        "name": "fuzzy_search",
        "description": (
            "Semantic search across all memories and history. Use for "
            "episodic recall ('did I talk to X about Y', 'what happened with Z'), "
            "multi-session synthesis ('summarize what's been happening with X'), "
            "or any question that doesn't fit exact lookup or aggregation."
        ),
        "parameters": {
            "query": {
                "type": "STRING",
                "description": "Natural language search query",
                "required": True,
            },
            "top_k": {
                "type": "INTEGER",
                "description": "Max results to return (default 10)",
                "required": False,
            },
        },
    },
    {
        "name": "get_valid_at",
        "description": (
            "Resolve what was true at a specific point in time. Use for "
            "'what was X called before', 'where did I sit before the move', "
            "'what changed', 'what is X called now' (renames/updates). "
            "Queries the bi-temporal schema to show history of changes."
        ),
        "parameters": {
            "subject": {
                "type": "STRING",
                "description": "The entity to check",
                "required": True,
            },
            "predicate": {
                "type": "STRING",
                "description": "The attribute to resolve (e.g., 'renamed_to', 'location', 'name')",
                "required": True,
            },
            "timestamp": {
                "type": "STRING",
                "description": "Use 'now' for current state, 'all' for full history, or an ISO 8601 timestamp",
                "required": True,
            },
        },
    },
    {
        "name": "update_dictionary",
        "description": (
            "Add or update a vocabulary correction rule in the dictionary, or correct a person's, project's, or entity's name. "
            "Use when the user makes a direct correction or modification like 'its Priya Reddy not raman', "
            "'her name is Priya Reddy not Raman', 'correct X to Y', 'change X to Y in dictionary', "
            "'kivi should write Y instead of X', 'add Y to dictionary'. "
            "raw_form is the incorrect/spoken form (e.g. 'raman' or 'Priya Raman'). "
            "corrected_form is the correct name/spelling (e.g. 'Priya Reddy')."
        ),
        "parameters": {
            "raw_form": {
                "type": "STRING",
                "description": "The incorrect, spoken, or old form to replace (e.g. 'raman' or 'Priya Raman')",
                "required": True,
            },
            "corrected_form": {
                "type": "STRING",
                "description": "The correct form to write (e.g. 'Priya Reddy')",
                "required": True,
            },
        },
    },
]

ROUTER_SYSTEM = """You are a query router for a personal memory system.
Given a user's question, decide which tool to call and with what arguments.
You have access to these tools:
- lookup_fact: for looking up specific facts about people/projects/preferences
- aggregate: for listing ALL items matching a filter (complete set, not top-K)
- fuzzy_search: for episodic recall, multi-session synthesis, or anything that doesn't fit exact lookup
- get_valid_at: for temporal questions about what was true at a specific time, or tracking changes/renames
- update_dictionary: for adding or modifying dictionary corrections or correcting entity names (e.g. 'its Priya Reddy not raman')

Choose the MOST APPROPRIATE tool. When in doubt, use fuzzy_search as fallback.
For preference questions, use lookup_fact with the relevant subject.
For corrections like 'it's Y not X', use update_dictionary with raw_form='X' and corrected_form='Y'.
"""


def _fast_path_check(question: str) -> dict | None:
    """Check if question is a simple, direct entity inquiry for fast-path bypass.
    
    Complex, episodic, temporal, synthesis, aggregation, or correction questions
    must never be bypassed and should always go to the LLM router.
    """
    import re

    question_clean = question.strip()
    question_lower = question_clean.lower()

    # Disqualify if the question contains any complex intent signals
    complex_signals_pattern = (
        r'\b(not|instead|change|changed|replace|replaced|correct|corrected|dictionary|'
        r'before|previously|earlier|used to|was called|what changed|when did|renam|rename|renamed|renaming|prior|'
        r'all|everyone|every|list|who are all|'
        r'summarize|summary|status|happening with|overview|draft|combine|combining|'
        r'preference|prefer|style|how should|'
        r'did|talk|discussed|discuss|meeting|call|coffee|happened|what did)\b'
    )
    if re.search(complex_signals_pattern, question_lower):
        return None

    # Must be a direct inquiry pattern (e.g. "Who is X?", "What is X's role?", "Tell me about X")
    direct_inquiry_pattern = r'^(who\s+is|who\'s|who\s+are|what\s+is|what\'s|tell\s+me\s+about)\b'
    if not re.search(direct_inquiry_pattern, question_lower) and len(question_clean.split()) > 7:
        return None

    # Load all known subjects
    subjects = fetchall("SELECT DISTINCT subject FROM memories WHERE valid_to IS NULL")
    if not subjects:
        return None

    for row in subjects:
        subject = row['subject']
        if subject and subject.lower() != 'user' and len(subject) >= 3:
            # Check whole-word / phrase match
            pattern = rf'\b{re.escape(subject.lower())}\b'
            if re.search(pattern, question_lower):
                return {
                    'tool_name': 'lookup_fact',
                    'tool_args': {'subject': subject},
                    'reasoning': f'Fast-path: direct entity lookup for "{subject}"',
                    'fast_path': True,
                }

    return None


def contextualize_query(question: str, conversation_history: list[dict] | None = None) -> str:
    """Rewrite a follow-up question into a standalone query using recent conversation turns."""
    if not conversation_history:
        return question

    history_turns = [t for t in conversation_history if t.get('content')]
    if not history_turns:
        return question

    history_text = "\n".join(
        f"{'User' if t.get('role') == 'user' else 'Hey Kivi'}: {t.get('content', '')[:250]}"
        for t in history_turns[-4:]
    )

    prompt = f"""Given this recent conversation between a user and their personal memory assistant:
{history_text}

User's latest message: "{question}"

If the latest message is a follow-up, refinement, or instruction that relies on the conversation above (e.g. pronouns like 'he/she/they/it/that', follow-up questions like 'what else?', or formatting/modification instructions like 'don't include dates' or 'make it shorter' or 'what about Tom?'), rewrite it into a single standalone search query that includes the relevant entities, topics, and constraints from the previous messages.
If the message is already completely standalone and introduces a new topic, return it unchanged.

Return ONLY the rewritten query text, with no preamble or explanation."""

    from llm import call_llm
    rewritten = call_llm(
        prompt=prompt,
        model=config.EXTRACTION_MODEL,
        temperature=0.0,
    )
    rewritten = (rewritten or '').strip().strip('"\'')
    return rewritten if rewritten and len(rewritten) > 2 else question


def route_query(question: str, conversation_history: list[dict] | None = None) -> dict:
    """Route a question to the appropriate tool via LLM function calling.

    Returns {tool_name, tool_args, reasoning, results, fast_path, contextualized_query}.
    """
    start = time.time()
    model_calls = 0

    # Contextualize query if conversation history is provided
    search_query = question
    if conversation_history:
        search_query = contextualize_query(question, conversation_history)
        if search_query != question:
            model_calls += 1

    # Fast-path check for direct entity lookups
    fast_result = _fast_path_check(search_query)
    if fast_result:
        # Execute the tool directly
        result = _execute_tool(fast_result['tool_name'], fast_result['tool_args'])
        # If fast-path found results, use them
        if result.get('count', 0) > 0:
            fast_result['results'] = result
            fast_result['latency_ms'] = (time.time() - start) * 1000
            fast_result['model_calls'] = model_calls
            fast_result['contextualized_query'] = search_query
            return fast_result
        # If no results from fast-path, fall through to LLM router

    # LLM function-calling router
    router_result = call_llm_with_tools(
        prompt=f"User question: {search_query}",
        tool_declarations=TOOL_DECLARATIONS,
        model=config.EXTRACTION_MODEL,
        temperature=config.EXTRACTION_TEMPERATURE,
        system_instruction=ROUTER_SYSTEM,
    )
    model_calls += 1

    tool_name = router_result.get('tool_name')
    tool_args = router_result.get('tool_args', {})
    reasoning = router_result.get('reasoning', '')

    if not tool_name:
        # LLM didn't call a tool — fallback to fuzzy_search
        tool_name = 'fuzzy_search'
        tool_args = {'query': search_query, 'top_k': 10}
        reasoning = 'LLM did not select a tool; falling back to fuzzy_search'

    # Execute the selected tool
    result = _execute_tool(tool_name, tool_args)

    # If structured tool returned 0 results, fall back to 3-signal fuzzy search
    if result.get('count', 0) == 0 and tool_name != 'fuzzy_search':
        fallback_result = _execute_tool('fuzzy_search', {'query': search_query, 'top_k': 10})
        if fallback_result.get('count', 0) > 0:
            tool_name = 'fuzzy_search'
            tool_args = {'query': search_query, 'top_k': 10}
            reasoning = (reasoning or '') + ' (fallback to fuzzy_search because initial tool returned 0 results)'
            result = fallback_result

    return {
        'tool_name': tool_name,
        'tool_args': tool_args,
        'reasoning': reasoning,
        'results': result,
        'fast_path': False,
        'latency_ms': (time.time() - start) * 1000,
        'model_calls': model_calls,
        'contextualized_query': search_query,
    }


def _execute_tool(tool_name: str, tool_args: dict) -> dict:
    """Execute a tool by name with given arguments."""
    if tool_name == 'lookup_fact':
        return lookup_fact(
            subject=tool_args.get('subject', ''),
            predicate=tool_args.get('predicate'),
        )
    elif tool_name == 'aggregate':
        return aggregate(
            filter_field=tool_args.get('filter_field', 'kind'),
            filter_value=tool_args.get('filter_value', 'factual'),
        )
    elif tool_name == 'fuzzy_search':
        return fuzzy_search(
            query=tool_args.get('query', ''),
            top_k=int(tool_args.get('top_k', 10)),
        )
    elif tool_name == 'get_valid_at':
        return get_valid_at(
            subject=tool_args.get('subject', ''),
            predicate=tool_args.get('predicate', ''),
            timestamp=tool_args.get('timestamp', 'now'),
        )
    elif tool_name == 'update_dictionary':
        return update_dictionary(
            raw_form=tool_args.get('raw_form', ''),
            corrected_form=tool_args.get('corrected_form', ''),
        )
    else:
        # Unknown tool — fallback
        return fuzzy_search(query=str(tool_args), top_k=10)
