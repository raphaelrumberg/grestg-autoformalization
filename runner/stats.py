"""Statistical analysis across completed benchmark directories.

  - 95% percentile-bootstrap CIs (Efron 1979) on run-level means
  - exact McNemar test (McNemar 1947; Dietterich 1998) on (case, run-index)
    paired held-out outcomes, pairwise across models
  - Wilcoxon signed-rank robustness check on per-case pass rates
  - Holm-Bonferroni correction over the model pairs
  - LaTeX tables + comparison figures

Usage:
    python -m runner.stats --dirs results/benchmark_A results/benchmark_B ... \
        [--log results/benchmark_full_log.txt] [--out results/analysis]
"""

import argparse
import itertools
import json
import math
import os
import re

import numpy as np
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 42  # fixed for reproducibility
ALPHA = 0.05


def load_benchmark(bench_dir: str) -> dict:
    """Load overall_report.json and per-run reports from a benchmark dir."""
    with open(os.path.join(bench_dir, "overall_report.json"), encoding="utf-8") as f:
        overall = json.load(f)

    runs = []
    for name in sorted(os.listdir(bench_dir)):
        if re.fullmatch(r"run_\d{3}", name):
            with open(os.path.join(bench_dir, name, "run_report.json"),
                      encoding="utf-8") as f:
                runs.append(json.load(f))

    case_ids = sorted({c["id"] for r in runs for c in r["test"]["cases"]})
    # case-outcome matrix [run][case] -> bool. A crashed run counts as failing every case, matching its recorded 0.0 pass rate
    matrix = {
        r["run"]: {cid: False for cid in case_ids} for r in runs
    }
    for r in runs:
        for c in r["test"]["cases"]:
            matrix[r["run"]][c["id"]] = bool(c["passed"])

    return {
        "dir": bench_dir,
        "model": overall["model"],
        "overall": overall,
        "runs": runs,
        "case_ids": case_ids,
        "test_matrix": matrix,
        "errored_runs": [r["run"] for r in runs if r.get("error")],
    }


def parse_initial_train_rates(log_path: str) -> dict:
    """Recover the initial train pass counts per model+run from the
    benchmark console log. Returns {model: {run_idx: passed}}.
    """
    with open(log_path, encoding="utf-8") as f:
        text = f.read()

    out = {}
    model = None
    run_idx = None
    expecting_train_total = False
    for line in text.splitlines():
        m = re.match(r"Model: (\S+)", line)
        if m:
            model = m.group(1)
            out.setdefault(model, {})
        m = re.match(r"\s*Run (\d+)/\d+", line)
        if m:
            run_idx = int(m.group(1))
        if re.match(r"\[TRAIN\] evaluating", line):
            # only the FIRST train eval of a run is the initial one
            expecting_train_total = run_idx not in out.get(model, {})
        m = re.match(r"Total: (\d+)\s+Passed: (\d+)\s+Failed: (\d+)", line)
        if m and expecting_train_total and model and run_idx is not None:
            out[model][run_idx] = int(m.group(2))
            expecting_train_total = False
    return out

def bootstrap_ci(values: list, b: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED):
    """95% percentile-bootstrap CI for the mean of `values`."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = rng.choice(arr, size=(b, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n) * 2
    return min(1.0, p)


def paired_case_run_outcomes(bench_a: dict, bench_b: dict):
    """Pair held-out outcomes by (case, run-index) across two models."""
    runs = sorted(set(bench_a["test_matrix"]) & set(bench_b["test_matrix"]))
    cases = sorted(set(bench_a["case_ids"]) & set(bench_b["case_ids"]))
    a, b = [], []
    for r in runs:
        for cid in cases:
            a.append(bench_a["test_matrix"][r][cid])
            b.append(bench_b["test_matrix"][r][cid])
    return np.array(a), np.array(b), runs, cases


def per_case_rates(bench: dict) -> dict:
    """Per held-out case: pass fraction across runs."""
    runs = sorted(bench["test_matrix"])
    return {
        cid: float(np.mean([bench["test_matrix"][r][cid] for r in runs]))
        for cid in bench["case_ids"]
    }


def load_case_kinds(suite_path: str = "test_suite/test.json") -> dict:
    """Map held-out case id -> kind (statutory mechanism taxonomy)."""
    try:
        with open(suite_path, encoding="utf-8") as f:
            return {tc["id"]: tc.get("kind", "?") for tc in json.load(f)}
    except OSError:
        return {}


def per_kind_rates(bench: dict, kinds: dict) -> dict:
    """Mean held-out pass rate per statutory kind (across cases and runs)."""
    rates = per_case_rates(bench)
    by_kind = {}
    for cid, rate in rates.items():
        by_kind.setdefault(kinds.get(cid, "?"), []).append(rate)
    return {k: float(np.mean(v)) for k, v in sorted(by_kind.items())}


def holm_correction(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for a dict {label: p}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running_max = {}, 0.0
    for rank, (label, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, adj)  # enforce monotonicity
        adjusted[label] = running_max
    return adjusted

def analyze(benchmarks: list, initial_train: dict,
            pairwise_exclude: set = frozenset()) -> dict:
    """Compute the full analysis dict for a list of loaded benchmarks."""
    kinds = load_case_kinds()
    models = {}
    for b in benchmarks:
        ov = b["overall"]
        test_rates = ov["test"]["per_run_pass_rates"]
        train_rates = ov["train"]["per_run_pass_rates"]
        gaps = ov["per_run_generalization_gaps"]
        init = initial_train.get(b["model"], {})
        n_train_cases = ov["train"]["total_cases_per_run"]
        init_rates = ([init[r] / n_train_cases for r in sorted(init)]
                      if init else None)

        valid = [r for r in b["runs"] if not r.get("error")]
        valid_test_rates = [r["test"]["pass_rate"] for r in valid]
        models[b["model"]] = {
            "dir": b["dir"],
            "num_runs": ov["num_runs"],
            "errored_runs": b["errored_runs"],
            "test_mean": float(np.mean(test_rates)),
            "test_ci": bootstrap_ci(test_rates),
            "test_per_run": test_rates,
            "num_valid_runs": len(valid),
            "test_mean_valid": (float(np.mean(valid_test_rates))
                                if valid_test_rates else None),
            "test_ci_valid": (bootstrap_ci(valid_test_rates)
                              if valid_test_rates else None),
            "train_mean": float(np.mean(train_rates)),
            "train_ci": bootstrap_ci(train_rates),
            "train_per_run": train_rates,
            "gap_mean": float(np.mean(gaps)),
            "gap_ci": bootstrap_ci(gaps),
            "gap_per_run": gaps,
            "perfect_test_runs": ov["test"]["perfect_runs"],
            "perfect_train_runs": ov["train"]["perfect_runs"],
            "fix_attempts": ov["fix_attempts_per_run"],
            "fix_attempts_mean": float(np.mean(ov["fix_attempts_per_run"])),
            "initial_train_per_run": init_rates,
            "initial_train_by_run": ({str(r): init[r] / n_train_cases
                                      for r in sorted(init)} if init else {}),
            "initial_train_mean": (float(np.mean(init_rates))
                                   if init_rates else None),
            "initial_train_ci": (bootstrap_ci(init_rates)
                                 if init_rates else None),
            "per_case_test_rates": per_case_rates(b),
            "per_kind_test_rates": per_kind_rates(b, kinds),
        }

    by_name = {b["model"]: b for b in benchmarks
               if b["model"] not in pairwise_exclude}
    pairwise, mcnemar_raw, wilcoxon_raw = {}, {}, {}
    for ma, mb in itertools.combinations(sorted(by_name), 2):
        a, bvec, runs, cases = paired_case_run_outcomes(by_name[ma], by_name[mb])
        n10 = int(np.sum(a & ~bvec))   # ma passes, mb fails
        n01 = int(np.sum(~a & bvec))
        p_mc = mcnemar_exact(n10, n01)

        ra = [models[ma]["per_case_test_rates"][c] for c in cases]
        rb = [models[mb]["per_case_test_rates"][c] for c in cases]
        diffs = np.array(ra) - np.array(rb)
        if np.all(diffs == 0):
            w_stat, p_w = float("nan"), 1.0
        else:
            w_stat, p_w = wilcoxon(ra, rb)
        label = f"{ma} vs {mb}"
        mcnemar_raw[label] = p_mc
        wilcoxon_raw[label] = float(p_w)
        pairwise[label] = {
            "n_pairs": len(a),
            "n_runs_paired": len(runs),
            "n_cases_paired": len(cases),
            "discordant_first_only": n10,
            "discordant_second_only": n01,
            "mcnemar_exact_p": p_mc,
            "wilcoxon_stat": None if math.isnan(w_stat) else float(w_stat),
            "wilcoxon_p": float(p_w),
        }

    for label, adj in holm_correction(mcnemar_raw).items():
        pairwise[label]["mcnemar_p_holm"] = adj
    for label, adj in holm_correction(wilcoxon_raw).items():
        pairwise[label]["wilcoxon_p_holm"] = adj

    return {"alpha": ALPHA, "bootstrap_B": BOOTSTRAP_B,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "models": models, "pairwise": pairwise}


def fmt_pct(x):
    return f"{x*100:.1f}"


def fmt_ci(ci):
    return f"[{ci[0]*100:.1f}, {ci[1]*100:.1f}]"


def fmt_p(p):
    return f"{p:.1e}" if p < 1e-4 else f"{p:.4f}"


def write_latex_tables(analysis: dict, out_dir: str):
    """Creates LaTeX tables."""
    models = analysis["models"]
    lines = [
        "% Auto-generated by runner/stats.py - don't edit by hand",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Benchmark results per model ($n$ runs each; pass rates in \\%; "
        "95\\,\\% percentile-bootstrap confidence intervals, $B=10{,}000$).}",
        "\\label{tab:benchmark-results}",
        "\\small",
        "\\begin{tabular}{lrrrrrrr}",
        "\\hline",
        "Model & $n$ & $n_{\\mathrm{valid}}$ & Initial train & Final train & Held-out test & Gap (pp) & Perfect \\\\",
        "\\hline",
    ]
    for name in sorted(models):
        m = models[name]
        init = (f"{fmt_pct(m['initial_train_mean'])}"
                if m["initial_train_mean"] is not None else "--")
        lines.append(
            f"\\texttt{{{name}}} & {m['num_runs']} & {m['num_valid_runs']} & {init} & "
            f"{fmt_pct(m['train_mean'])} {fmt_ci(m['train_ci'])} & "
            f"{fmt_pct(m['test_mean'])} {fmt_ci(m['test_ci'])} & "
            f"{m['gap_mean']*100:+.1f} & "
            f"{m['perfect_test_runs']}/{m['num_runs']} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]

    lines += [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Pairwise model comparison on the held-out suite: exact McNemar "
        "test on (case, run)-paired outcomes and Wilcoxon signed-rank test on "
        "per-case pass rates; Holm-corrected $p$-values, family-wise "
        "$\\alpha=0.05$.}",
        "\\label{tab:pairwise-tests}",
        "\\small",
        "\\begin{tabular}{lrrrr}",
        "\\hline",
        "Pair & Discordant & McNemar $p$ (Holm) & Wilcoxon $p$ (Holm) & Sig. \\\\",
        "\\hline",
    ]
    for label in sorted(analysis["pairwise"]):
        p = analysis["pairwise"][label]
        sig = "yes" if p["mcnemar_p_holm"] < ALPHA else "no"
        pair_tex = label.replace(" vs ", " vs.\\ \\texttt{")
        lines.append(
            f"\\texttt{{{pair_tex}}}}} & "
            f"{p['discordant_first_only']}/{p['discordant_second_only']} & "
            f"{fmt_p(p['mcnemar_exact_p'])} ({fmt_p(p['mcnemar_p_holm'])}) & "
            f"{fmt_p(p['wilcoxon_p'])} ({fmt_p(p['wilcoxon_p_holm'])}) & {sig} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]

    all_kinds = sorted({k for n in models for k in models[n]["per_kind_test_rates"]})
    if all_kinds:
        colspec = "l" + "r" * len(models)
        lines += [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Held-out pass rate (\\%) by statutory mechanism "
            "(\\texttt{kind} taxonomy), averaged over cases and runs.}",
            "\\label{tab:per-kind}",
            "\\small",
            f"\\begin{{tabular}}{{{colspec}}}",
            "\\hline",
            "Kind & " + " & ".join(f"\\texttt{{{n}}}" for n in sorted(models)) + " \\\\",
            "\\hline",
        ]
        for kind in all_kinds:
            row = " & ".join(
                fmt_pct(models[n]["per_kind_test_rates"].get(kind, float("nan")))
                for n in sorted(models))
            lines.append(f"\\texttt{{{kind.replace('_', '\\_')}}} & {row} \\\\")
        lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]

    path = os.path.join(out_dir, "tables.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"LaTeX tables written to {path}")


def plot_comparison(analysis: dict, out_dir: str):
    """Cross-model figure: pass rates with CI bars + per-case heatmap."""
    models = analysis["models"]
    names = sorted(models)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # mean rates with bootstrap CI error bars
    ax = axes[0]
    x = np.arange(len(names))
    width = 0.35
    for off, key, ci_key, label, color in (
        (-width / 2, "train_mean", "train_ci", "Train (post-fix)", "#3498db"),
        (width / 2, "test_mean", "test_ci", "Held-out test", "#9b59b6"),
    ):
        means = [models[n][key] * 100 for n in names]
        los = [(models[n][key] - models[n][ci_key][0]) * 100 for n in names]
        his = [(models[n][ci_key][1] - models[n][key]) * 100 for n in names]
        ax.bar(x + off, means, width, yerr=[los, his], capsize=5,
               label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Pass rate (%)")
    ax.set_ylim(0, 105)
    ax.axhline(100, color="#2ecc71", linestyle="--", alpha=0.5)
    ax.set_title("Mean pass rate per model (95% bootstrap CI)")
    ax.legend()

    # per-case reliability heatmap on held-out suite
    ax = axes[1]
    case_ids = sorted(models[names[0]]["per_case_test_rates"])
    data = np.array([[models[n]["per_case_test_rates"][c] for c in case_ids]
                     for n in names])
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xticks(range(len(case_ids)))
    ax.set_xticklabels(case_ids, rotation=90, fontsize=6)
    ax.set_title("Per-case reliability, held-out suite (fraction of runs passed)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    path = os.path.join(out_dir, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Comparison figure written to {path}")


def plot_convergence(analysis: dict, benchmarks: list, out_dir: str):
    """Fix-loop convergence: train pass rate per fix attempt, per model."""
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(5 * len(benchmarks), 4.5),
                             sharey=True)
    if len(benchmarks) == 1:
        axes = [axes]
    for ax, bench in zip(axes, sorted(benchmarks, key=lambda b: b["model"])):
        model = bench["model"]
        init_by_run = analysis["models"][model]["initial_train_by_run"]
        for run in bench["runs"]:
            run_dir = os.path.join(bench["dir"], f"run_{run['run']:03d}")
            traj = []
            if str(run["run"]) in init_by_run:
                traj.append(init_by_run[str(run["run"])] * 100)
            for k in range(1, run["fix_attempts"] + 1):
                with open(os.path.join(run_dir, f"fix_{k}", "train_report.json"),
                          encoding="utf-8") as f:
                    traj.append(json.load(f)["train"]["pass_rate"] * 100)
            if not traj:
                traj = [run["train"]["pass_rate"] * 100]
            ax.plot(range(len(traj)), traj, marker="o", alpha=0.55)
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Fix attempt")
        ax.set_xticks(range(0, 4))
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Train pass rate (%)")
    fig.suptitle("Fix-loop convergence (one line per run; attempt 0 = initial formalization)")
    plt.tight_layout()
    path = os.path.join(out_dir, "fix_convergence.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Convergence figure written to {path}")


def print_summary(analysis: dict):
    models = analysis["models"]
    print(f"\n{'='*78}\n  CROSS-MODEL ANALYSIS\n{'='*78}")
    hdr = (f"{'model':<22}{'init train':>11}{'train':>8}{'test':>7}"
           f"{'test 95% CI':>16}{'gap':>7}{'perfect':>9}{'fixes':>7}")
    print(hdr)
    print("-" * len(hdr))
    for name in sorted(models):
        m = models[name]
        init = fmt_pct(m["initial_train_mean"]) if m["initial_train_mean"] is not None else "--"
        print(f"{name:<22}{init:>11}{fmt_pct(m['train_mean']):>8}"
              f"{fmt_pct(m['test_mean']):>7}{fmt_ci(m['test_ci']):>16}"
              f"{m['gap_mean']*100:>+7.1f}"
              f"{str(m['perfect_test_runs'])+'/'+str(m['num_runs']):>9}"
              f"{m['fix_attempts_mean']:>7.1f}")
        if m["errored_runs"]:
            print(f"  !! runs aborted by fatal errors: {m['errored_runs']} "
                  f"(count as 0.0); held-out mean over the {m['num_valid_runs']} "
                  f"valid runs: {fmt_pct(m['test_mean_valid'])}% "
                  f"{fmt_ci(m['test_ci_valid'])}")
    print("\nPairwise (held-out suite):")
    for label in sorted(analysis["pairwise"]):
        p = analysis["pairwise"][label]
        print(f"  {label}: discordant {p['discordant_first_only']}/"
              f"{p['discordant_second_only']}, "
              f"McNemar exact p={fmt_p(p['mcnemar_exact_p'])} "
              f"(Holm {fmt_p(p['mcnemar_p_holm'])}), "
              f"Wilcoxon p={fmt_p(p['wilcoxon_p'])} "
              f"(Holm {fmt_p(p['wilcoxon_p_holm'])})")


def main():
    ap = argparse.ArgumentParser(description="Cross-model benchmark analysis")
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="benchmark result directories (one per model)")
    ap.add_argument("--log", nargs="*", default=[],
                    help="console log(s) to recover initial train pass rates; "
                         "a later log fully overrides a model present in an "
                         "earlier one (e.g. a re-run)")
    ap.add_argument("--out", default="results/analysis",
                    help="output directory")
    ap.add_argument("--pairwise-exclude", nargs="*", default=[],
                    help="model names to keep out of the paired tests")
    args = ap.parse_args()

    benchmarks = [load_benchmark(d) for d in args.dirs]
    initial_train = {}
    for log_path in args.log:
        for model, runs in parse_initial_train_rates(log_path).items():
            initial_train[model] = runs  # whole-model override, no merge

    analysis = analyze(benchmarks, initial_train,
                       pairwise_exclude=set(args.pairwise_exclude))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis JSON written to {os.path.join(args.out, 'analysis.json')}")

    print_summary(analysis)
    write_latex_tables(analysis, args.out)
    plot_comparison(analysis, args.out)
    plot_convergence(analysis, benchmarks, args.out)


if __name__ == "__main__":
    main()
