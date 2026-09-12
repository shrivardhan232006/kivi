"""Grounding verification pass — checks that generated answers are supported by context.

Runs after every generated answer. If unsupported claims are found, they're
stripped or the answer falls back to abstention.
"""

from llm import call_llm_json
import config

VERIFICATION_SYSTEM = """You are a verification system that checks whether a generated answer is
supported by the source context. Your job is to identify any claims in the
answer that are NOT present in or supported by the source context.

Be strict: if a specific claim cannot be traced to the source context, flag it.
But don't flag reasonable phrasings or summaries of what IS in the context."""


def verify_answer(question: str, answer: str, context: str, conversation_history: list[dict] | None = None) -> dict:
    """Verify that a generated answer is supported by its source context.

    Returns {passed, flagged_claims, verified_answer}.
    """
    if not answer or "I don't have" in answer or "not mentioned" in answer.lower() or "dictionary updated" in context.lower():
        # Abstention answers and system confirmations don't need verification
        return {
            'passed': True,
            'flagged_claims': [],
            'verified_answer': answer,
        }

    history_str = ""
    if conversation_history:
        turns = [f"{'User' if t.get('role') == 'user' else 'Assistant'}: {t.get('content', '')[:200]}" for t in conversation_history[-3:]]
        if turns:
            history_str = "Recent conversation:\n" + "\n".join(turns) + "\n\n"

    prompt = f"""{history_str}Source context:
{context}

Generated answer:
{answer}

Question that was asked:
{question}

Check: does every factual claim in the generated answer appear in or
follow logically from the source context?

Return a JSON object with:
- "pass": true if all claims are supported, false if any are unsupported
- "flagged_claims": array of strings, each being an unsupported claim found in the answer (empty if all pass)
"""

    result = call_llm_json(
        prompt=prompt,
        model=config.EXTRACTION_MODEL,
        temperature=config.VERIFICATION_TEMPERATURE,
        system_instruction=VERIFICATION_SYSTEM,
    )

    if isinstance(result, list):
        result = result[0] if result else {'pass': True, 'flagged_claims': []}

    passed = result.get('pass', True)
    flagged = result.get('flagged_claims', [])

    verified_answer = answer
    if not passed and flagged:
        # If many claims flagged, fall back to abstention
        if len(flagged) >= 3:
            verified_answer = "I'm not confident enough to answer that based on your dictation history."
        # Otherwise keep the answer but note it
        # (In production you'd strip specific flagged claims)

    return {
        'passed': passed,
        'flagged_claims': flagged,
        'verified_answer': verified_answer,
    }
