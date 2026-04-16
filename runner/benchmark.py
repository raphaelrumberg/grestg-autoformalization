"""Benchmark runner: formalize + test over multiple runs, then produce reports."""

import json
import os
import shutil
import traceback
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from llm.base import LLMClient
from runner.formalizer import formalize, fix_code
from runner.test_runner import run_tests, print_results, plot_results

MAX_FIX_ATTEMPTS = 3


def _build_failed_cases_info(results: dict) -> list:
    """Extract info about failing test cases for the fix prompt."""
    with open(config.TEST_SUITE_PATH, "r", encoding="utf-8") as f:
        all_cases = {tc["id"]: tc for tc in json.load(f)}

    failed = []
    for c in results["cases"]:
        if not c["passed"]:
            tc = all_cases[c["id"]]
            actual_str = f"({c['actual_triggered']}, {c['actual_taxpayer']!r})" if c["error"] is None else f"ERROR: {c['error']}"
            failed.append({
                "id": c["id"],
                "description": c["description"],
                "graph": tc["graph"],
                "target_entity": tc["target_entity"],
                "acquirer_groups": tc.get("acquirer_groups", []),
                "expected": f"({c['expected_triggered']}, {c['expected_taxpayer']!r})",
                "actual": actual_str,
            })
    return failed


def _save_sub_run(run_dir: str, sub_idx: int, code: str, results: dict):
    """Save code and results for a correction sub-run."""
    sub_dir = os.path.join(run_dir, f"fix_{sub_idx}")
    os.makedirs(sub_dir, exist_ok=True)
    with open(os.path.join(sub_dir, "generated_code.py"), "w", encoding="utf-8") as f:
        f.write(code)
    plot_results(results, os.path.join(sub_dir, "test_results.png"))
    sub_report = {
        "fix_attempt": sub_idx,
        "passed": results["passed"],
        "failed": results["failed"],
        "total": results["total"],
        "pass_rate": results["passed"] / results["total"] if results["total"] > 0 else 0,
        "cases": results["cases"],
    }
    with open(os.path.join(sub_dir, "test_report.json"), "w", encoding="utf-8") as f:
        json.dump(sub_report, f, indent=2)
    return sub_dir


def run_benchmark(num_runs: int, client: LLMClient):
    """
    Run formalize -> test for num_runs iterations, saving per-run reports
    and an overall summary report at the end.

    When a run has failing tests, up to MAX_FIX_ATTEMPTS correction sub-runs
    are attempted where the LLM receives the failing cases and is asked to fix
    the code.
    """
    model = client.model_name()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_dir = os.path.join(config.RESULTS_DIR, f"benchmark_{model}_{timestamp}")
    os.makedirs(benchmark_dir, exist_ok=True)
    all_run_results = []

    print(f"Model: {model}")

    for i in range(1, num_runs + 1):
        print(f"\n{'='*60}")
        print(f"  Run {i}/{num_runs}  [{model}]")
        print(f"{'='*60}\n")

        run_dir = os.path.join(benchmark_dir, f"run_{i:03d}")
        os.makedirs(run_dir, exist_ok=True)

        try:
            # Formalize
            formalize(client)

            # Read generated code for archival
            with open(config.GENERATED_CODE_PATH, "r", encoding="utf-8") as f:
                generated_code = f.read()

            # Save generated code
            with open(os.path.join(run_dir, "generated_code.py"), "w", encoding="utf-8") as f:
                f.write(generated_code)

            # Test
            results = run_tests()
            print_results(results)

            # Save test results plot
            plot_results(results, os.path.join(run_dir, "test_results.png"))

            # Correction loop
            fix_attempts = 0
            while results["failed"] > 0 and fix_attempts < MAX_FIX_ATTEMPTS:
                fix_attempts += 1
                print(f"\n  --- Fix attempt {fix_attempts}/{MAX_FIX_ATTEMPTS} ---\n")

                failed_info = _build_failed_cases_info(results)
                generated_code = fix_code(client, generated_code, failed_info)

                results = run_tests()
                print_results(results)

                sub_dir = _save_sub_run(run_dir, fix_attempts, generated_code, results)
                print(f"  Fix {fix_attempts} saved to {sub_dir}/")

            # Save test results JSON
            run_report = {
                "run": i,
                "passed": results["passed"],
                "failed": results["failed"],
                "total": results["total"],
                "pass_rate": results["passed"] / results["total"] if results["total"] > 0 else 0,
                "fix_attempts": fix_attempts,
                "cases": results["cases"],
                "error": None,
            }

        except Exception:
            error_msg = traceback.format_exc()
            print(f"\nRun {i} FAILED with error:\n{error_msg}")

            # Save error info and generated code if available
            generated_code = ""
            if os.path.exists(config.GENERATED_CODE_PATH):
                with open(config.GENERATED_CODE_PATH, "r", encoding="utf-8") as f:
                    generated_code = f.read()
                with open(os.path.join(run_dir, "generated_code.py"), "w", encoding="utf-8") as f:
                    f.write(generated_code)

            with open(config.TEST_SUITE_PATH, "r", encoding="utf-8") as f:
                total_cases = len(json.load(f))

            run_report = {
                "run": i,
                "passed": 0,
                "failed": total_cases,
                "total": total_cases,
                "pass_rate": 0.0,
                "fix_attempts": 0,
                "cases": [],
                "error": error_msg,
            }

        with open(os.path.join(run_dir, "test_report.json"), "w", encoding="utf-8") as f:
            json.dump(run_report, f, indent=2)

        all_run_results.append(run_report)
        print(f"\nRun {i} report saved to {run_dir}/")

    # Generate overall report
    _generate_overall_report(all_run_results, benchmark_dir, num_runs, model)

    print(f"\nBenchmark complete. All results saved to {benchmark_dir}/")


def _generate_overall_report(all_runs: list, benchmark_dir: str, num_runs: int, model: str = ""):
    """Generate the overall summary report across all runs."""
    total_cases = all_runs[0]["total"] if all_runs else 0
    pass_rates = [r["pass_rate"] for r in all_runs]
    avg_pass_rate = sum(pass_rates) / len(pass_rates) if pass_rates else 0
    perfect_runs = sum(1 for r in all_runs if r["pass_rate"] == 1.0)

    # Per-case pass rate across runs
    case_pass_counts = {}
    for run in all_runs:
        for case in run["cases"]:
            cid = case["id"]
            if cid not in case_pass_counts:
                case_pass_counts[cid] = {"passed": 0, "total": 0, "description": case["description"]}
            case_pass_counts[cid]["total"] += 1
            if case["passed"]:
                case_pass_counts[cid]["passed"] += 1

    case_stats = []
    for cid in sorted(case_pass_counts.keys()):
        info = case_pass_counts[cid]
        case_stats.append({
            "id": cid,
            "description": info["description"],
            "pass_count": info["passed"],
            "total_runs": info["total"],
            "pass_rate": info["passed"] / info["total"] if info["total"] > 0 else 0,
        })

    # Summary JSON
    overall = {
        "model": model,
        "num_runs": num_runs,
        "total_cases_per_run": total_cases,
        "average_pass_rate": round(avg_pass_rate, 4),
        "min_pass_rate": round(min(pass_rates), 4) if pass_rates else 0,
        "max_pass_rate": round(max(pass_rates), 4) if pass_rates else 0,
        "perfect_runs": perfect_runs,
        "per_run_pass_rates": [round(r, 4) for r in pass_rates],
        "per_case_stats": case_stats,
    }

    with open(os.path.join(benchmark_dir, "overall_report.json"), "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  OVERALL BENCHMARK REPORT")
    print(f"  Model: {model}  |  Runs: {num_runs}")
    print(f"{'='*60}")
    print(f"Average pass rate:  {avg_pass_rate*100:.1f}%")
    print(f"Min pass rate:      {min(pass_rates)*100:.1f}%" if pass_rates else "")
    print(f"Max pass rate:      {max(pass_rates)*100:.1f}%" if pass_rates else "")
    print(f"Perfect runs:       {perfect_runs}/{num_runs}")
    print(f"\nPer-run pass rates:")
    for i, rate in enumerate(pass_rates, 1):
        bar = "#" * int(rate * 30)
        print(f"  Run {i:3d}: {rate*100:5.1f}%  |{bar:<30}|")

    print(f"\nPer-case reliability:")
    for cs in case_stats:
        bar = "#" * int(cs["pass_rate"] * 30)
        print(f"  {cs['id']}: {cs['pass_count']}/{cs['total_runs']} ({cs['pass_rate']*100:.0f}%)  |{bar:<30}|")

    # Plot overall results
    _plot_overall(overall, benchmark_dir)


def _plot_overall(overall: dict, benchmark_dir: str):
    """Create overall benchmark visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Pass rate per run (bar chart)
    ax = axes[0, 0]
    runs = list(range(1, overall["num_runs"] + 1))
    rates = [r * 100 for r in overall["per_run_pass_rates"]]
    colors = ["#2ecc71" if r == 100 else "#f39c12" if r >= 75 else "#e74c3c" for r in rates]
    ax.bar(runs, rates, color=colors)
    ax.set_xlabel("Run")
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("Pass Rate per Run")
    ax.set_ylim(0, 105)
    ax.axhline(y=100, color="#2ecc71", linestyle="--", alpha=0.5)
    ax.set_xticks(runs)

    # 2. Pass rate distribution (histogram)
    ax = axes[0, 1]
    ax.hist(rates, bins=max(5, overall["num_runs"] // 2), color="#3498db", edgecolor="white")
    ax.set_xlabel("Pass Rate (%)")
    ax.set_ylabel("Frequency")
    ax.set_title("Pass Rate Distribution")
    ax.axvline(x=overall["average_pass_rate"] * 100, color="#e74c3c", linestyle="--",
               label=f"Mean: {overall['average_pass_rate']*100:.1f}%")
    ax.legend()

    # 3. Per-case reliability (horizontal bar)
    ax = axes[1, 0]
    case_ids = [cs["id"] for cs in overall["per_case_stats"]]
    case_rates = [cs["pass_rate"] * 100 for cs in overall["per_case_stats"]]
    case_colors = ["#2ecc71" if r == 100 else "#f39c12" if r >= 75 else "#e74c3c" for r in case_rates]
    ax.barh(case_ids, case_rates, color=case_colors)
    ax.set_xlabel("Pass Rate (%)")
    ax.set_title("Per-Case Reliability")
    ax.set_xlim(0, 105)

    # 4. Summary pie chart
    ax = axes[1, 1]
    perfect = overall["perfect_runs"]
    imperfect = overall["num_runs"] - perfect
    if perfect > 0 or imperfect > 0:
        ax.pie(
            [perfect, imperfect],
            labels=["Perfect (100%)", "Imperfect"],
            colors=["#2ecc71", "#e74c3c"],
            autopct="%1.0f%%",
            startangle=90,
        )
    ax.set_title("Perfect Runs")

    plt.suptitle(
        f"Benchmark Report: {overall['model']} — {overall['num_runs']} Runs, "
        f"Avg Pass Rate {overall['average_pass_rate']*100:.1f}%",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    output_path = os.path.join(benchmark_dir, "overall_report.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Overall report plot saved to {output_path}")
