import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from evaluator.runner import EvalRunner, TestCase


def progress_cb(done: int, total: int, result) -> None:
    status = "✓" if result.passed else ("ERR" if result.error else "✗")
    score  = f"{result.composite_score:.3f}" if not result.error else "error"
    print(f"  [{done}/{total}] {status}  {result.test_case_id:<10}  score={score}")


def main():
    parser = argparse.ArgumentParser(description="EvalForge - conversational AI evaluator")
    parser.add_argument("--endpoint",      required=True,  help="Target endpoint URL")
    parser.add_argument("--endpoint-type", default="openai",
                        choices=["openai", "generic"], help="Endpoint protocol")
    parser.add_argument("--api-key",       default="",     help="API key / bearer token")
    parser.add_argument("--model",         default="gpt-4o-mini", help="Model name (openai mode)")
    parser.add_argument("--prompt-field",  default="message",     help="Prompt field (generic mode)")
    parser.add_argument("--response-field",default="response",    help="Response field (generic mode)")
    parser.add_argument("--suite",         default="tests/test_suite.json",
                        help="Path to test suite JSON")
    parser.add_argument("--output",        default="results/run.json",
                        help="Path to write results JSON")
    parser.add_argument("--timeout",       default=30, type=int)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  EvalForge  |  endpoint: {args.endpoint}")
    print(f"  Suite: {args.suite}  |  Model: {args.model}")
    print(f"{'='*60}\n")

    runner = EvalRunner(
        endpoint_url   = args.endpoint,
        endpoint_type  = args.endpoint_type,
        api_key        = args.api_key,
        model          = args.model,
        prompt_field   = args.prompt_field,
        response_field = args.response_field,
        timeout        = args.timeout,
    )

    with open(args.suite) as f:
        raw = json.load(f)
    test_cases = [TestCase(**tc) for tc in raw]

    print(f"Running {len(test_cases)} test cases...\n")
    summary = runner.run(test_cases, on_progress=progress_cb)
    runner.save_results(summary, args.output)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  RUN SUMMARY  (run_id={summary.run_id})")
    print(f"{'='*60}")
    print(f"  Total cases   : {summary.total_cases}")
    print(f"  Passed        : {summary.passed_cases}  ({summary.pass_rate*100:.1f}%)")
    print(f"  Failed        : {summary.failed_cases}")
    print(f"  Errors        : {summary.error_cases}")
    print(f"  Mean score    : {summary.mean_composite_score:.4f}")
    print(f"  Mean latency  : {summary.mean_latency_ms:.0f} ms")
    print(f"\n  Metric breakdown:")
    for name, data in summary.metric_breakdown.items():
        bar = "█" * int(data["mean_score"] * 20) + "░" * (20 - int(data["mean_score"] * 20))
        print(f"    {name:<28}  {bar}  {data['mean_score']:.3f}")
    print(f"\n  Full results → {args.output}")
    print()


if __name__ == "__main__":
    main()
