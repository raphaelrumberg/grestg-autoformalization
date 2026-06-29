import importlib.util
import json
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def run_tests(suite_path: str) -> dict:
    """
    Load the given test suite and run each case against the generated formalization.

    The generated code is always loaded fresh from disk via importlib to ensure
    the latest version is tested. A case passes iff the function's returned bool
    matches expected_triggered (§ 1 Abs 3 = whether the tax is triggered).

    Returns:
        A dict with keys: passed, failed, total, suite_path, cases.
    """
    with open(suite_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    # Load generated code fresh from disk
    spec = importlib.util.spec_from_file_location(
        "generated_code", config.GENERATED_CODE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    check_fn = module.is_grest_triggered

    results = []
    passed = 0
    failed = 0

    for case in cases:
        case_result = {
            "id": case["id"],
            "description": case["description"],
            "expected_triggered": case["expected_triggered"],
            "actual_triggered": None,
            "passed": False,
            "error": None,
        }
        try:
            triggered = check_fn(
                case["graph"],
                case["target_entity"],
                case.get("acquirer_groups", []),
            )
            case_result["actual_triggered"] = triggered
            case_result["passed"] = (triggered == case["expected_triggered"])
        except Exception:
            case_result["error"] = traceback.format_exc()

        if case_result["passed"]:
            passed += 1
        else:
            failed += 1
        results.append(case_result)

    return {"passed": passed, "failed": failed, "total": len(cases),
            "suite_path": suite_path, "cases": results}


def print_results(results: dict):
    """Print a simple aligned table of test results to stdout."""
    header = f"{'ID':<10} {'Exp/Act':<14} {'Status':<8} {'Error'}"
    print(header)
    print("-" * len(header))
    for c in results["cases"]:
        status = "PASS" if c["passed"] else "FAIL"
        error_snippet = (c["error"] or "").split("\n")[-2][:40] if c["error"] else ""

        if c["actual_triggered"] is not None:
            trig_str = f"{c['expected_triggered']}/{c['actual_triggered']}"
        else:
            trig_str = f"{c['expected_triggered']}/CRASH"

        print(f"{c['id']:<10} {trig_str:<14} {status:<8} {error_snippet}")
    print(f"\nTotal: {results['total']}  Passed: {results['passed']}  Failed: {results['failed']}")


def plot_results(results: dict, output_path: str):
    """
    Create a matplotlib figure with test results visualization.

    Left subplot: horizontal bar chart (one bar per case, green=pass / red=fail).
    Right subplot: pie chart of overall pass rate.
    A case passes iff the returned bool matches expected_triggered.

    Args:
        results: The dict returned by run_tests().
        output_path: File path to save the figure.
    """
    cases = results["cases"]
    ids = [c["id"] for c in cases]
    colors = ["#2ecc71" if c["passed"] else "#e74c3c" for c in cases]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(4, len(cases) * 0.8)))

    # Bar chart
    ax1.barh(ids, [1] * len(cases), color=colors)
    ax1.set_xlabel("Result")
    ax1.set_title("Test Cases")
    ax1.set_xlim(0, 1.2)
    ax1.set_xticks([])

    # Pie chart
    ax2.pie(
        [results["passed"], results["failed"]],
        labels=["Passed", "Failed"],
        colors=["#2ecc71", "#e74c3c"],
        autopct="%1.0f%%",
        startangle=90,
    )
    ax2.set_title("Pass Rate")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Results plot saved to {output_path}")
