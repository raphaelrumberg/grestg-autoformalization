# GrEStG Autoformalization

LLM-based autoformalization of Austrian real estate transfer tax law (GrEStG § 1 Abs 3).

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your API keys:

```
ANTHROPIC_API_KEY=sk-...
GEMINI_API_KEY=...
OPENAI_API_KEY=sk-...
```

## Test suite design (train / held-out test)

The suite contains **71 cases** stratified into:

- `test_suite/train.json`: **49 cases**. The only suite the LLM ever sees:
  failing train cases are fed back into the reflection/fix loop.
- `test_suite/test.json`: **22 cases, held out**. Evaluated exactly once per
  benchmark run, after the fix loop terminates; never placed in any LLM
  prompt. The **generalization gap** (train − test pass rate) is reported per
  run and in the overall report.

Every case carries a `kind` (statutory mechanism: `z2_direct`, `z2_chain`,
`z2_summed_paths`, `z2_mixed`, `multi_acquirer`, `group_direct`,
`group_mixed`, `precedence`, `z1_gesellschafterwechsel`) and a `legal_rationale` citing the relevant
Ziffer/sentence of § 1 Abs 3. Eleven train cases form metamorphic-relation sets
(relabeling, scaling, zero-edge, window-shift). Both splits cover every kind, both outcome
labels where the kind has both, and threshold boundaries (74.99 / 75.0 /
75.01).

The suites are curated by hand as JSON and checked into the repository. Each
case pairs an ownership structure with the tax outcome the statute prescribes,
a `kind` label, and a `legal_rationale`. The test runner reads them directly.

## Commands

### `formalize`: Generate formalization via LLM

Sends the statutory text and prompt to the selected LLM and writes the generated Python code to `generated/generated_code.py`.

```bash
python main.py formalize
python main.py formalize --model gemini
python main.py formalize --model openai
```

| Option    | Values                        | Default  |
|-----------|-------------------------------|----------|
| `--model` | `claude`, `gemini`, `openai`  | `claude` |

### `test`: Run a test suite

Runs the selected suite against the current `generated/generated_code.py` and outputs results to the console and `results/`.

```bash
python main.py test                 # train suite (default for ad-hoc development)
python main.py test --suite test    # held-out suite, reserved for final metrics
```

| Option    | Values                      | Default |
|-----------|-----------------------------|---------|
| `--suite` | `train`, `test`             | `train` |

Output:
- Console table with per-case pass/fail
- `results/<suite>_results.png`: bar chart + pie chart
- `results/graph_tc_*.png`: ownership graph per test case

### `benchmark`: Run multiple formalize+fix+test iterations

For each run: formalize → evaluate on **train** → up to 3 reflection/fix
attempts driven by failing **train** cases only → evaluate the **held-out
test suite once** (never shown to the LLM). Saves per-run reports and an
overall summary.

```bash
python main.py benchmark --runs 10
python main.py benchmark --runs 5 --model gemini
```

| Option    | Values                        | Default  |
|-----------|-------------------------------|----------|
| `--runs`  | integer (required)            | —        |
| `--model` | `claude`, `gemini`, `openai`  | `claude` |

Output (saved to `results/benchmark_<model>_<timestamp>/`):
- `run_001/`, `run_002/`, ...: per-run directories containing:
  - `generated_code.py`: archived generated code
  - `fix_1/`, `fix_2/`, ...: per-fix-attempt code, reflection, train results
  - `run_report.json`: train + held-out test results, generalization gap
  - `train_results.png`, `test_results.png`: visualizations
- `overall_report.json`: per-suite avg/min/max pass rates, perfect runs,
  per-case reliability, per-run generalization gaps, fix attempts
- `overall_report.png`: 4-panel visualization (train-vs-test per run,
  gap per run, per-case reliability for both suites)

### `visualize`: Visualize a single test case graph

Renders the ownership graph for a specific test case (searched across the train and test suites).

```bash
python main.py visualize --case tc_001
```

| Option   | Values                      | Default |
|----------|-----------------------------|---------|
| `--case` | test case ID (required)     | —       |

Output: `results/graph_<case_id>.png`

## Project Structure

```
├── main.py                  # CLI entry point
├── config.py                # Configuration (model names, paths)
├── prompt.txt               # LLM prompt template
├── statutory_text/          # Legal source text
├── test_suite/
│   ├── train.json           # 49 cases: LLM-visible (fix loop)
│   └── test.json            # 22 cases: held out (final metric)
├── llm/                     # LLM client implementations
├── runner/
│   ├── formalizer.py        # Code generation + reflection/fix via LLM
│   ├── test_runner.py       # Test execution & visualization
│   ├── benchmark.py         # Multi-run benchmark runner (train/test split)
│   └── visualizer.py        # Graph rendering
├── generated/               # LLM-generated code output
└── results/                 # Test results & benchmark reports
```
