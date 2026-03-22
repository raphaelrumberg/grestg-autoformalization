# GrEStG Autoformalization

LLM-based autoformalization of Austrian real estate transfer tax law (GrEStG § 1 Abs 3).

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your API keys:

```
ANTHROPIC_API_KEY=sk-...
GOOGLE_API_KEY=...
OPENAI_API_KEY=sk-...
```

## Commands

### `formalize` — Generate formalization via LLM

Sends the statutory text and prompt to the selected LLM and writes the generated Python code to `generated/generated_code.py`.

```bash
python main.py formalize
python main.py formalize --model gemini
python main.py formalize --model openai
```

| Option    | Values                        | Default  |
|-----------|-------------------------------|----------|
| `--model` | `claude`, `gemini`, `openai`  | `claude` |

### `test` — Run test suite

Runs all 15 test cases against the current `generated/generated_code.py` and outputs results to the console and `results/`.

```bash
python main.py test
```

Output:
- Console table with per-case pass/fail
- `results/test_results.png` — bar chart + pie chart
- `results/graph_tc_*.png` — ownership graph per test case

### `benchmark` — Run multiple formalize+test iterations

Runs formalization and testing N times, saves per-run reports, and generates an overall summary report.

```bash
python main.py benchmark --runs 10
python main.py benchmark --runs 5 --model gemini
```

| Option    | Values                        | Default  |
|-----------|-------------------------------|----------|
| `--runs`  | integer (required)            | —        |
| `--model` | `claude`, `gemini`, `openai`  | `claude` |

Output (saved to `results/benchmark_<timestamp>/`):
- `run_001/`, `run_002/`, ... — per-run directories containing:
  - `generated_code.py` — archived generated code
  - `test_report.json` — detailed test results
  - `test_results.png` — test visualization
- `overall_report.json` — summary with avg/min/max pass rates, per-case reliability
- `overall_report.png` — 4-panel visualization (pass rate per run, distribution, per-case reliability, perfect runs)

### `visualize` — Visualize a single test case graph

Renders the ownership graph for a specific test case.

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
├── test_suite/cases.json    # 15 test cases
├── llm/                     # LLM client implementations
├── runner/
│   ├── formalizer.py        # Code generation via LLM
│   ├── test_runner.py       # Test execution & visualization
│   ├── benchmark.py         # Multi-run benchmark runner
│   └── visualizer.py        # Graph rendering
├── generated/               # LLM-generated code output
└── results/                 # Test results & benchmark reports
```
