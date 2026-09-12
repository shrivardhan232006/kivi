"""LLM answer generation — synthesizes a response from retrieved context.

Every generated answer is a real LLM call, never a string template with
values substituted in. This is the user-facing quality layer.
"""

import json
from llm import call_llm
import config

GENERATION_SYSTEM = """You are Hey Kivi, a personal memory assistant for a voice dictation product.
You answer questions about the user's dictation history, the people they work with,
their preferences, and their projects.

Rules:
- Answer using ONLY the provided context. Do not make up information.
- If the context doesn't contain enough information to answer, say clearly:
  "I don't have that information in your dictation history."
- Be specific and concise.
- When referencing people, use their full names.
- When a preference is asked about AND the user wants you to APPLY it (not just recite),
  demonstrate how it would be applied.
- When multiple records relate to the same topic over time, synthesize them into
  a coherent narrative showing how things evolved.
- Include relevant dates/timeframes when they add value.
- For temporal questions (renames, moves, changes), explain what changed and when.
"""


def format_context(tool_result: dict) -> str:
    """Format tool results into a readable context string for the LLM."""
    tool_name = tool_result.get('tool', 'unknown')
    results = tool_result.get('results', [])

    if not results:
        return "No relevant memories or history records found."

    lines = []
    if tool_name == 'aggregate':
        field = tool_result.get('filter_field', '')
        val = tool_result.get('filter_value', '')
        lines.append(f"Aggregate query results: All records in user history matching {field} = '{val}':")
    elif tool_name == 'update_dictionary':
        raw = tool_result.get('raw_form', '')
        corrected = tool_result.get('corrected_form', '')
        mem_count = tool_result.get('memories_updated', 0)
        lines.append(f"System Action: Dictionary updated successfully.")
        lines.append(f"Whenever the user says '{raw}', Kivi will now write '{corrected}'.")
        if mem_count > 0:
            lines.append(f"{mem_count} memory records in database updated with the new name '{corrected}'.")
        lines.append("Confirm this dictionary update clearly to the user.")

    for i, item in enumerate(results, 1):
        if 'kind' in item:
            # Memory record
            line = f"[Memory {i}] ({item.get('kind', 'unknown')})"
            line += f" Subject: {item.get('subject', '?')}"
            line += f" | {item.get('predicate', '?')}: {item.get('value', '?')}"
            if item.get('qualifier'):
                line += f" (qualifier: {item['qualifier']})"
            if item.get('valid_from'):
                line += f" | Valid from: {item['valid_from']}"
            if item.get('valid_to'):
                line += f" | Valid until: {item['valid_to']}"
            # Source provenance
            source_records = item.get('source_records', [])
            if source_records:
                for src in source_records:
                    line += f"\n  Source [{src.get('id', '?')}]: \"{src.get('canonical_text', '')}\""
                    line += f" (app: {src.get('app', '?')}, time: {src.get('timestamp', '?')})"
            lines.append(line)
        elif 'canonical_text' in item:
            # History record
            line = f"[History {i}] \"{item.get('canonical_text', '')}\""
            line += f" (app: {item.get('app', '?')}, time: {item.get('timestamp', '?')})"
            lines.append(line)

    return "\n".join(lines)


def generate_answer(question: str, tool_result: dict, conversation_history: list[dict] | None = None) -> dict:
    """Generate an answer from retrieved context using LLM.

    Returns {answer, source_record_ids, model_calls}.
    """
    if tool_result.get('tool') == 'update_dictionary':
        raw = tool_result.get('raw_form', '')
        corrected = tool_result.get('corrected_form', '')
        mem_count = tool_result.get('memories_updated', 0)
        msg = f"I've updated your dictionary: whenever you say **{raw}**, Kivi will now write **{corrected}**."
        if mem_count > 0:
            msg += f" I've also updated {mem_count} memory record(s) in your history to reflect this."
        return {
            'answer': msg,
            'source_record_ids': [],
            'model_calls': 0,
        }

    context = format_context(tool_result)

    history_str = ""
    if conversation_history:
        history_lines = []
        for turn in conversation_history[-4:]:
            role = "User" if turn.get('role') == 'user' else "Hey Kivi"
            history_lines.append(f"{role}: {turn.get('content', '')[:350]}")
        if history_lines:
            history_str = "Recent conversation:\n" + "\n".join(history_lines) + "\n\n"

    prompt = f"""{history_str}Context from user's memory:
{context}

User question: {question}

Answer the question based on the context and conversation history above."""

    answer = call_llm(
        prompt=prompt,
        model=config.GENERATION_MODEL,
        temperature=config.GENERATION_TEMPERATURE,
        system_instruction=GENERATION_SYSTEM,
    )

    # Extract source record IDs from the tool results
    source_ids = set()
    for item in tool_result.get('results', []):
        if 'source_record_ids' in item:
            try:
                ids = json.loads(item['source_record_ids']) if isinstance(item['source_record_ids'], str) else item['source_record_ids']
                source_ids.update(ids)
            except (json.JSONDecodeError, TypeError):
                pass
        if 'id' in item and item.get('_source_type') == 'history':
            source_ids.add(item['id'])
        for src in item.get('source_records', []):
            source_ids.add(src.get('id', ''))

    return {
        'answer': answer,
        'source_record_ids': list(source_ids),
        'model_calls': 1,
    }
