"""Benchmark runner: formalize + test over multiple runs, then produce reports."""

import json
import os
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


def _build_failed_cases_info(train_results: dict) -> list:
    """Extract info about failing TRAIN cases for the fix prompt."""
    assert train_results["suite_path"] == config.TRAIN_SUITE_PATH, (
        "fix loop must only ever receive TRAIN results, got "
        f"{train_results['suite_path']}"
    )
    with open(config.TRAIN_SUITE_PATH, "r", encoding="utf-8") as f:
        all_cases = {tc["id"]: tc for tc in json.load(f)}

    failed = []
    for c in train_results["cases"]:
        if not c["passed"]:
            tc = all_cases[c["id"]]
            actual_str = f"{c['actual_triggered']}" if c["error"] is None else f"ERROR: {c['error']}"
            failed.append({
                "id": c["id"],
                "description": c["description"],
                "graph": tc["graph"],
                "target_entity": tc["target_entity"],
                "acquirer_groups": tc.get("acquirer_groups", []),
                "expected": f"{c['expected_triggered']}",
                "actual": actual_str,
            })
    return failed


def _suite_summary(results: dict) -> dict:
    return {
        "suite_path": results["suite_path"],
        "passed": results["passed"],
        "failed": results["failed"],
        "total": results["total"],
        "pass_rate": results["passed"] / results["total"] if results["total"] > 0 else 0,
        "cases": results["cases"],
    }

def _save_sub_run(run_dir: str, sub_idx: int, code: str, train_results: dict,
                  reflection: str = ""):
    """Save code, TRAIN results, and reflection for a correction sub-run."""
    sub_dir = os.path.join(run_dir, f"fix_{sub_idx}")
    os.makedirs(sub_dir, exist_ok=True)
    with open(os.path.join(sub_dir, "generated_code.py"), "w", encoding="utf-8") as f:
        f.write(code)
    if reflection:
        with open(os.path.join(sub_dir, "reflection.md"), "w", encoding="utf-8") as f:
            f.write(reflection)
    plot_results(train_results, os.path.join(sub_dir, "train_results.png"))
    sub_report = {
        "fix_attempt": sub_idx,
        "reflection": reflection,
        "train": _suite_summary(train_results),
    }
    with open(os.path.join(sub_dir, "train_report.json"), "w", encoding="utf-8") as f:
        json.dump(sub_report, f, indent=2)
    return sub_dir


def run_benchmark(num_runs: int, client: LLMClient):
    """
    Run formalize -> fix-loop(TRAIN) -> held-out TEST for num_runs iterations,
    saving per-run reports and an overall summary report at the end.

    When a run has failing TRAIN tests, up to MAX_FIX_ATTEMPTS correction
    sub-runs are attempted where the LLM receives the failing TRAIN cases and
    is asked to fix the code. The held-out TEST suite is evaluated once after
    the loop terminates and is never shown to the LLM.
    """
    model = client.model_name()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_dir = os.path.join(config.RESULTS_DIR, f"benchmark_{model}_{timestamp}")
    os.makedirs(benchmark_dir, exist_ok=True)
    all_run_results = []

    print(f"Model: {model}")
    print(f"Fix loop suite (LLM-visible): {config.TRAIN_SUITE_PATH}")
    print(f"Held-out suite (final metric): {config.TEST_SUITE_PATH}")

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

            # Evaluate on TRAIN
            print(f"[TRAIN] evaluating {config.TRAIN_SUITE_PATH}")
            train_results = run_tests(config.TRAIN_SUITE_PATH)
            print_results(train_results)
            plot_results(train_results, os.path.join(run_dir, "train_results.png"))

            # Correction loop
            fix_attempts = 0
            while train_results["failed"] > 0 and fix_attempts < MAX_FIX_ATTEMPTS:
                fix_attempts += 1
                print(f"\n  --- Fix attempt {fix_attempts}/{MAX_FIX_ATTEMPTS} ---")

                failed_info = _build_failed_cases_info(train_results)
                print(f"  fix loop input: {len(failed_info)} failing TRAIN cases "
                      f"from {config.TRAIN_SUITE_PATH}\n")
                generated_code, reflection = fix_code(client, generated_code, failed_info)

                train_results = run_tests(config.TRAIN_SUITE_PATH)
                print_results(train_results)

                sub_dir = _save_sub_run(run_dir, fix_attempts, generated_code,
                                        train_results, reflection)
                print(f"  Fix {fix_attempts} saved to {sub_dir}/")

            print(f"\n[HELD-OUT TEST] evaluating {config.TEST_SUITE_PATH} once "
                  f"(excluded from the fix loop)")
            test_results = run_tests(config.TEST_SUITE_PATH)
            print_results(test_results)
            plot_results(test_results, os.path.join(run_dir, "test_results.png"))

            train_summary = _suite_summary(train_results)
            test_summary = _suite_summary(test_results)
            gap = train_summary["pass_rate"] - test_summary["pass_rate"]
            print(f"\nRun {i}: train {train_summary['pass_rate']*100:.1f}%  |  "
                  f"held-out test {test_summary['pass_rate']*100:.1f}%  |  "
                  f"generalization gap {gap*100:+.1f} pp")

            run_report = {
                "run": i,
                "fix_attempts": fix_attempts,
                "train": train_summary,
                "test": test_summary,
                "generalization_gap": gap,
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

            def _empty(suite_path):
                with open(suite_path, "r", encoding="utf-8") as f:
                    total = len(json.load(f))
                return {"suite_path": suite_path, "passed": 0, "failed": total,
                        "total": total, "pass_rate": 0.0, "cases": []}

            run_report = {
                "run": i,
                "fix_attempts": 0,
                "train": _empty(config.TRAIN_SUITE_PATH),
                "test": _empty(config.TEST_SUITE_PATH),
                "generalization_gap": 0.0,
                "error": error_msg,
            }

        with open(os.path.join(run_dir, "run_report.json"), "w", encoding="utf-8") as f:
            json.dump(run_report, f, indent=2)

        all_run_results.append(run_report)
        print(f"\nRun {i} report saved to {run_dir}/")

    # Generate overall report
    _generate_overall_report(all_run_results, benchmark_dir, num_runs, model)

    print(f"\nBenchmark complete. All results saved to {benchmark_dir}/")


def _per_case_stats(all_runs: list, suite_key: str) -> list:
    """Per-case pass rates across runs for one suite ('train' or 'test')."""
    counts = {}
    for run in all_runs:
        for case in run[suite_key]["cases"]:
            cid = case["id"]
            if cid not in counts:
                counts[cid] = {"passed": 0, "total": 0, "description": case["description"]}
            counts[cid]["total"] += 1
            if case["passed"]:
                counts[cid]["passed"] += 1
    return [{
        "id": cid,
        "description": counts[cid]["description"],
        "pass_count": counts[cid]["passed"],
        "total_runs": counts[cid]["total"],
        "pass_rate": counts[cid]["passed"] / counts[cid]["total"] if counts[cid]["total"] > 0 else 0,
    } for cid in sorted(counts.keys())]


def _generate_overall_report(all_runs: list, benchmark_dir: str, num_runs: int, model: str = ""):
    """Generate the overall summary report across all runs."""

    def _suite_block(suite_key):
        rates = [r[suite_key]["pass_rate"] for r in all_runs]
        return {
            "average_pass_rate": round(sum(rates) / len(rates), 4) if rates else 0,
            "min_pass_rate": round(min(rates), 4) if rates else 0,
            "max_pass_rate": round(max(rates), 4) if rates else 0,
            "perfect_runs": sum(1 for r in rates if r == 1.0),
            "per_run_pass_rates": [round(r, 4) for r in rates],
            "total_cases_per_run": all_runs[0][suite_key]["total"] if all_runs else 0,
            "per_case_stats": _per_case_stats(all_runs, suite_key),
        }

    gaps = [r["generalization_gap"] for r in all_runs]
    overall = {
        "model": model,
        "num_runs": num_runs,
        "train": _suite_block("train"),
        "test": _suite_block("test"),
        "average_generalization_gap": round(sum(gaps) / len(gaps), 4) if gaps else 0,
        "per_run_generalization_gaps": [round(g, 4) for g in gaps],
        "fix_attempts_per_run": [r["fix_attempts"] for r in all_runs],
    }

    with open(os.path.join(benchmark_dir, "overall_report.json"), "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    # Print summary
    train_b, test_b = overall["train"], overall["test"]
    print(f"\n{'='*60}")
    print(f"  OVERALL BENCHMARK REPORT")
    print(f"  Model: {model}  |  Runs: {num_runs}")
    print(f"{'='*60}")
    print(f"{'':24}{'TRAIN':>10}{'TEST (held-out)':>18}")
    print(f"{'Average pass rate:':<24}{train_b['average_pass_rate']*100:>9.1f}%"
          f"{test_b['average_pass_rate']*100:>17.1f}%")
    print(f"{'Min pass rate:':<24}{train_b['min_pass_rate']*100:>9.1f}%"
          f"{test_b['min_pass_rate']*100:>17.1f}%")
    print(f"{'Max pass rate:':<24}{train_b['max_pass_rate']*100:>9.1f}%"
          f"{test_b['max_pass_rate']*100:>17.1f}%")
    print(f"{'Perfect runs:':<24}{train_b['perfect_runs']:>9}/{num_runs}"
          f"{test_b['perfect_runs']:>13}/{num_runs}")
    print(f"\nAverage generalization gap (train - test): "
          f"{overall['average_generalization_gap']*100:+.1f} pp")
    print(f"\nPer-run (train | test | gap | fix attempts):")
    for idx, run in enumerate(all_runs, 1):
        tr = run["train"]["pass_rate"] * 100
        te = run["test"]["pass_rate"] * 100
        gap = run["generalization_gap"] * 100
        print(f"  Run {idx:3d}: {tr:5.1f}% | {te:5.1f}% | {gap:+6.1f} pp | "
              f"{run['fix_attempts']} fixes")

    for label, block in (("TRAIN", train_b), ("TEST (held-out)", test_b)):
        print(f"\nPer-case reliability — {label}:")
        for cs in block["per_case_stats"]:
            bar = "#" * int(cs["pass_rate"] * 30)
            print(f"  {cs['id']}: {cs['pass_count']}/{cs['total_runs']} "
                  f"({cs['pass_rate']*100:.0f}%)  |{bar:<30}|")

    # Plot overall results
    _plot_overall(overall, benchmark_dir)


def _plot_overall(overall: dict, benchmark_dir: str):
    """Create overall benchmark visualization (train vs held-out test)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    runs = list(range(1, overall["num_runs"] + 1))
    train_rates = [r * 100 for r in overall["train"]["per_run_pass_rates"]]
    test_rates = [r * 100 for r in overall["test"]["per_run_pass_rates"]]

    # 1. Train vs test pass rate per run (grouped bars)
    ax = axes[0, 0]
    width = 0.4
    ax.bar([r - width / 2 for r in runs], train_rates, width, label="Train",
           color="#3498db")
    ax.bar([r + width / 2 for r in runs], test_rates, width, label="Test (held-out)",
           color="#9b59b6")
    ax.set_xlabel("Run")
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("Pass Rate per Run: Train vs Held-out Test")
    ax.set_ylim(0, 105)
    ax.axhline(y=100, color="#2ecc71", linestyle="--", alpha=0.5)
    ax.set_xticks(runs)
    ax.legend()

    # 2. Generalization gap per run
    ax = axes[0, 1]
    gaps = [g * 100 for g in overall["per_run_generalization_gaps"]]
    colors = ["#e74c3c" if g > 0 else "#2ecc71" for g in gaps]
    ax.bar(runs, gaps, color=colors)
    ax.axhline(y=0, color="black", linewidth=0.8)
    avg_gap = overall["average_generalization_gap"] * 100
    ax.axhline(y=avg_gap, color="#e67e22", linestyle="--",
               label=f"Mean gap: {avg_gap:+.1f} pp")
    ax.set_xlabel("Run")
    ax.set_ylabel("Train − Test (pp)")
    ax.set_title("Generalization Gap per Run")
    ax.set_xticks(runs)
    ax.legend()

    # 3./4. Per-case reliability for each suite
    for ax, (label, block, color) in zip(
        (axes[1, 0], axes[1, 1]),
        (("Held-out Test", overall["test"], "#9b59b6"),
         ("Train", overall["train"], "#3498db")),
    ):
        case_ids = [cs["id"] for cs in block["per_case_stats"]]
        case_rates = [cs["pass_rate"] * 100 for cs in block["per_case_stats"]]
        case_colors = ["#2ecc71" if r == 100 else "#f39c12" if r >= 75 else "#e74c3c"
                       for r in case_rates]
        ax.barh(case_ids, case_rates, color=case_colors)
        ax.set_xlabel("Pass Rate (%)")
        ax.set_title(f"Per-Case Reliability — {label}")
        ax.set_xlim(0, 105)
        ax.tick_params(axis="y", labelsize=6)

    plt.suptitle(
        f"Benchmark Report: {overall['model']} — {overall['num_runs']} Runs | "
        f"Train {overall['train']['average_pass_rate']*100:.1f}% | "
        f"Held-out Test {overall['test']['average_pass_rate']*100:.1f}% | "
        f"Gap {overall['average_generalization_gap']*100:+.1f} pp",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    output_path = os.path.join(benchmark_dir, "overall_report.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Overall report plot saved to {output_path}")
