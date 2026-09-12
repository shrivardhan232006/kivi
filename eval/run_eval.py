"""Evaluation harness — LLM-as-judge scoring.

Scores by comparing generated answers against expected answers using
an LLM judge call, NOT by keyword/ID presence matching.

Scores by category (mapping to query types) with abstention tracked explicitly.
"""

import json
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import get_db
from db.migrate import migrate
from query.router import route_query
from query.generator import generate_answer, format_context
from query.verifier import verify_answer
from llm import call_llm_json
import config


def load_eval_questions(path: str) -> list[dict]:
    """Load eval questions JSONL."""
    questions = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def judge_answer(question: str, expected: str, generated: str) -> dict:
    """LLM-as-judge: compare generated answer against expected."""
    prompt = f"""You are judging whether a generated answer correctly answers a question.

Question: {question}
Expected answer: {expected}
Generated answer: {generated}

Evaluate:
1. Does the generated answer contain the key facts from the expected answer?
2. Is the generated answer factually correct (no hallucinated claims)?
3. Is the generated answer complete (doesn't miss key information)?

Return JSON:
{{
    "correct": true/false,
    "partial": true/false (some key facts present but incomplete),
    "reason": "brief explanation"
}}
"""
    result = call_llm_json(
        prompt=prompt,
        model=config.EXTRACTION_MODEL,
        temperature=0.1,
    )

    if isinstance(result, list):
        result = result[0] if result else {'correct': False, 'partial': False, 'reason': 'Judge returned empty'}

    return {
        'correct': result.get('correct', False),
        'partial': result.get('partial', False),
        'reason': result.get('reason', ''),
    }


def run_eval(questions_path: str, output_dir: str):
    """Run full evaluation."""
    start = time.time()

    print("=" * 60)
    print("KIVI SEMANTIC MEMORY - Evaluation")
    print("=" * 60)

    # Ensure DB is ready
    migrate()

    questions = load_eval_questions(questions_path)
    print(f"Loaded {len(questions)} eval questions")

    results = []
    category_stats: dict[str, dict] = {}

    for i, q in enumerate(questions, 1):
        q_start = time.time()
        qid = q['id']
        question = q['question']
        expected = q['expected_answer']
        qtype = q.get('type', 'unknown')

        print(f"\n[{i}/{len(questions)}] ({qtype}) {question[:60]}...")

        try:
            # Step 1: Route
            route_result = route_query(question)
            tool_name = route_result['tool_name']
            tool_result = route_result['results']
            model_calls = route_result.get('model_calls', 0)

            # Step 2: Generate
            gen_result = generate_answer(question, tool_result)
            answer = gen_result['answer']
            model_calls += gen_result['model_calls']

            # Step 3: Verify
            context_text = format_context(tool_result)
            ver_result = verify_answer(question, answer, context_text)
            final_answer = ver_result['verified_answer']
            model_calls += 1

            # Step 4: Judge
            judge = judge_answer(question, expected, final_answer)
            model_calls += 1

            q_latency = (time.time() - q_start) * 1000

            result = {
                'id': qid,
                'question': question,
                'type': qtype,
                'expected_answer': expected,
                'generated_answer': final_answer,
                'tool_name': tool_name,
                'tool_args': route_result.get('tool_args', {}),
                'correct': judge['correct'],
                'partial': judge.get('partial', False),
                'judge_reason': judge['reason'],
                'verification_passed': ver_result['passed'],
                'flagged_claims': ver_result.get('flagged_claims', []),
                'fast_path': route_result.get('fast_path', False),
                'latency_ms': q_latency,
                'model_calls': model_calls,
            }

            status = "PASS" if judge['correct'] else ("PARTIAL" if judge.get('partial') else "FAIL")
            print(f"  {status} | tool={tool_name} | {q_latency:.0f}ms | {model_calls} calls")
            if not judge['correct']:
                print(f"  Expected: {expected[:80]}")
                print(f"  Got:      {final_answer[:80]}")
                print(f"  Reason:   {judge['reason'][:80]}")

        except Exception as e:
            q_latency = (time.time() - q_start) * 1000
            result = {
                'id': qid,
                'question': question,
                'type': qtype,
                'expected_answer': expected,
                'generated_answer': f'ERROR: {str(e)}',
                'tool_name': 'error',
                'correct': False,
                'partial': False,
                'judge_reason': str(e),
                'latency_ms': q_latency,
                'model_calls': 0,
            }
            print(f"  ERROR: {str(e)[:100]}")

        results.append(result)

        # Track category stats
        if qtype not in category_stats:
            category_stats[qtype] = {'total': 0, 'correct': 0, 'partial': 0, 'failed': 0}
        category_stats[qtype]['total'] += 1
        if result['correct']:
            category_stats[qtype]['correct'] += 1
        elif result.get('partial'):
            category_stats[qtype]['partial'] += 1
        else:
            category_stats[qtype]['failed'] += 1

        time.sleep(1.0)

    # --- Summary ---
    elapsed = time.time() - start
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    partial = sum(1 for r in results if r.get('partial') and not r['correct'])
    failed = total - correct - partial

    total_latency = sum(r['latency_ms'] for r in results)
    avg_latency = total_latency / total if total else 0
    total_model_calls = sum(r.get('model_calls', 0) for r in results)

    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"Overall: {correct}/{total} correct ({100*correct/total:.1f}%)")
    print(f"  Partial: {partial} | Failed: {failed}")
    print(f"")
    print(f"By category:")
    for cat, stats in sorted(category_stats.items()):
        pct = 100 * stats['correct'] / stats['total'] if stats['total'] else 0
        print(f"  {cat:25s}: {stats['correct']}/{stats['total']} ({pct:.0f}%) "
              f"[partial: {stats['partial']}, failed: {stats['failed']}]")

    print(f"\nPerformance:")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Avg latency: {avg_latency:.0f}ms per question")
    print(f"  Total model calls: {total_model_calls}")
    print(f"  Avg model calls: {total_model_calls/total:.1f} per question")
    print(f"{'=' * 60}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'eval_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total': total,
                'correct': correct,
                'partial': partial,
                'failed': failed,
                'accuracy': correct / total if total else 0,
                'avg_latency_ms': avg_latency,
                'total_model_calls': total_model_calls,
                'elapsed_seconds': elapsed,
            },
            'by_category': category_stats,
            'results': results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {results_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kivi eval harness")
    parser.add_argument("--questions", default=os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "eval_questions.jsonl"))
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "eval", "results"))
    args = parser.parse_args()
    run_eval(args.questions, args.output)
