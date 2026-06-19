"""Entry point for the GrEStG autoformalization pipeline."""

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import os

import config
from llm import ClaudeClient, GeminiClient, OpenAIClient
from runner.formalizer import formalize
from runner.test_runner import run_tests, print_results, plot_results
from runner.visualizer import visualize_graph
from runner.benchmark import run_benchmark


def make_client(args):
    """Create the appropriate LLM client based on CLI arguments."""
    provider = getattr(args, "model", "claude")
    if provider == "gemini":
        return GeminiClient()
    if provider == "openai":
        return OpenAIClient()
    return ClaudeClient()


def cmd_formalize(args):
    """Run the LLM-based formalization step."""
    client = make_client(args)
    formalize(client)


SUITE_PATHS = {
    "train": config.TRAIN_SUITE_PATH,
    "test": config.TEST_SUITE_PATH,
}


def cmd_test(args):
    """Run a test suite against the generated formalization."""
    suite_path = SUITE_PATHS[args.suite]
    print(f"Suite: {args.suite} ({suite_path})")
    results = run_tests(suite_path)
    print_results(results)
    plot_results(results, os.path.join(config.RESULTS_DIR, f"{args.suite}_results.png"))

    with open(suite_path, "r", encoding="utf-8") as f:
        all_cases = json.load(f)

    case_data = {tc["id"]: tc for tc in all_cases}
    for case in results["cases"]:
        tc = case_data.get(case["id"])
        if tc:
            output_path = os.path.join(config.RESULTS_DIR, f"graph_{case['id']}.png")
            visualize_graph(tc["graph"], case["id"], output_path,
                            tc.get("acquirer_groups", []))


def cmd_benchmark(args):
    """Run formalize + test for multiple iterations and produce reports."""
    client = make_client(args)
    run_benchmark(args.runs, client)


def cmd_visualize(args):
    """Visualize the ownership graph for a specific test case."""
    for suite_name in ("train", "test"):
        with open(SUITE_PATHS[suite_name], "r", encoding="utf-8") as f:
            all_cases = json.load(f)

        for tc in all_cases:
            if tc["id"] == args.case:
                output_path = os.path.join(config.RESULTS_DIR, f"graph_{tc['id']}.png")
                visualize_graph(tc["graph"], tc["id"], output_path,
                                tc.get("acquirer_groups", []))
                return

    print(f"Error: case '{args.case}' not found in any suite "
          f"({', '.join(SUITE_PATHS.values())})")


def main():
    """Parse CLI arguments and dispatch to the appropriate command."""
    parser = argparse.ArgumentParser(
        description="GrEStG § 1 Abs 3 autoformalization pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    form_parser = subparsers.add_parser("formalize", help="Generate formalization via LLM")
    form_parser.add_argument("--model", choices=["claude", "gemini", "openai"], default="claude",
                             help="LLM provider to use (default: claude)")

    test_parser = subparsers.add_parser("test", help="Run a test suite against generated code")
    test_parser.add_argument("--suite", choices=["train", "test"], default="train",
                             help="Suite to run (default: train; 'test' is the held-out set "
                                  "reserved for final metrics)")

    bench_parser = subparsers.add_parser("benchmark", help="Run formalize+test N times and generate reports")
    bench_parser.add_argument("--runs", type=int, required=True, help="Number of runs")
    bench_parser.add_argument("--model", choices=["claude", "gemini", "openai"], default="claude",
                              help="LLM provider to use (default: claude)")

    vis_parser = subparsers.add_parser("visualize", help="Visualize a test case graph")
    vis_parser.add_argument("--case", required=True, help="Test case ID (e.g. tc_001)")

    args = parser.parse_args()

    if args.command == "formalize":
        cmd_formalize(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "visualize":
        cmd_visualize(args)


if __name__ == "__main__":
    main()
