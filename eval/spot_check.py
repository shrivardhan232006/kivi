"""Manual spot-check tool for aggregation queries.

Runs a configurable query and prints the full retrieval pipeline output
to verify aggregation returns complete sets, not just top-K.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.migrate import migrate
from query.router import route_query
from query.generator import generate_answer, format_context
from query.verifier import verify_answer


def spot_check(question: str):
    """Run a single question through the full pipeline with verbose output."""
    migrate()

    print(f"\n{'='*60}")
    print(f"SPOT CHECK: {question}")
    print(f"{'='*60}")

    # Route
    print("\n--- ROUTING ---")
    route_result = route_query(question)
    print(f"Tool: {route_result['tool_name']}")
    print(f"Args: {json.dumps(route_result['tool_args'], indent=2)}")
    print(f"Reasoning: {route_result.get('reasoning', 'N/A')}")
    print(f"Fast-path: {route_result.get('fast_path', False)}")

    tool_result = route_result['results']
    print(f"\n--- RETRIEVAL ({tool_result.get('count', 0)} results) ---")
    for i, item in enumerate(tool_result.get('results', []), 1):
        if 'subject' in item:
            print(f"  [{i}] {item.get('kind','?')}: {item['subject']} | {item.get('predicate','')} = {item.get('value','')}")
            if item.get('qualifier'):
                print(f"       qualifier: {item['qualifier']}")
        elif 'canonical_text' in item:
            print(f"  [{i}] history: {item.get('canonical_text','')[:80]} ({item.get('app','')}, {item.get('timestamp','')})")

    # Generate
    print("\n--- GENERATION ---")
    gen_result = generate_answer(question, tool_result)
    print(f"Answer: {gen_result['answer']}")
    print(f"Sources: {gen_result.get('source_record_ids', [])}")

    # Verify
    print("\n--- VERIFICATION ---")
    context_text = format_context(tool_result)
    ver_result = verify_answer(question, gen_result['answer'], context_text)
    print(f"Passed: {ver_result['passed']}")
    if ver_result.get('flagged_claims'):
        print(f"Flagged: {ver_result['flagged_claims']}")
    print(f"Final answer: {ver_result['verified_answer']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Question to spot-check")
    args = parser.parse_args()
    spot_check(args.query)
