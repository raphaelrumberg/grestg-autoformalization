import json
import re

import config
from llm.base import LLMClient

SYSTEM_PROMPT = (
    "You are a legal formalization expert. Generate only valid Python code. "
    "Your entire response must be a single ```python code block. "
    "Define exactly one function: def is_grest_triggered(graph: dict, "
    "target_entity: str, acquirer_groups: list = None) -> bool"
)

REFLECTION_SYSTEM_PROMPT = (
    "You are a legal formalization expert reviewing your previous implementation. "
    "Analyze the failing test cases and explain concisely what the previous "
    "implementation got wrong or forgot. Do NOT output any code — only your analysis."
)

FIX_SYSTEM_PROMPT = (
    "You are a legal formalization expert. Generate only valid Python code. "
    "Your entire response must be a single ```python code block."
)


def _extract_code(raw_response: str) -> str:
    """Extract Python code from an LLM response."""
    match = re.search(r"```python\s*\n(.*?)```", raw_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    code = raw_response.strip()
    code = re.sub(r"^```(?:python)?\s*\n?", "", code)
    code = re.sub(r"\n?```\s*$", "", code)
    return code


def _write_code(code: str):
    """Write code to the configured output path."""
    with open(config.GENERATED_CODE_PATH, "w", encoding="utf-8") as f:
        f.write(code + "\n")


def formalize(client: LLMClient) -> str:
    """
    Use an LLM to formalize the statutory text into executable Python code.

    Reads the statutory text and prompt template from disk, sends them to the
    LLM, extracts the Python code block from the response, and writes it to
    the configured output path.

    Args:
        client: An LLMClient instance to use for code generation.

    Returns:
        The extracted Python code as a string.
    """
    with open(config.STATUTORY_TEXT_PATH, "r", encoding="utf-8") as f:
        statutory_text = f.read()

    with open(config.PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    user_message = prompt_text + "\n\n" + statutory_text
    raw_response = client.generate(SYSTEM_PROMPT, user_message)

    code = _extract_code(raw_response)
    _write_code(code)

    line_count = code.count("\n") + 1
    print(f"Formalization complete: {line_count} lines written to {config.GENERATED_CODE_PATH}")

    return code


def fix_code(client: LLMClient, code: str, failed_cases: list) -> tuple:
    """
    Ask the LLM to reflect on what went wrong, then fix the generated code.

    First performs a reflection step where the LLM analyzes the failures
    without producing code, then uses that analysis to generate a fix.

    Args:
        client: An LLMClient instance to use for code generation.
        code: The current (broken) generated code.
        failed_cases: List of dicts with keys: id, description, graph,
            target_entity, acquirer_groups, expected, actual.

    Returns:
        A tuple of (fixed_code, reflection) where reflection is the LLM's
        analysis of what went wrong.
    """
    failures_text = ""
    for fc in failed_cases:
        failures_text += (
            f"\n--- {fc['id']}: {fc['description']} ---\n"
            f"Input graph: {json.dumps(fc['graph'])}\n"
            f"target_entity: {fc['target_entity']}\n"
            f"acquirer_groups: {json.dumps(fc['acquirer_groups'])}\n"
            f"Expected: {fc['expected']}\n"
            f"Actual:   {fc['actual']}\n"
        )

    reflection_message = (
        f"Review the following code and its failing test cases. "
        f"Explain what the implementation got wrong or forgot.\n\n"
        f"## Current code\n```python\n{code}\n```\n\n"
        f"## Failing test cases\n{failures_text}"
    )

    reflection = client.generate(REFLECTION_SYSTEM_PROMPT, reflection_message)

    print(f"\n  Reflection:\n{reflection}\n")

    fix_message = (
        f"The following code has failing test cases. "
        f"Fix the code so that all test cases pass.\n\n"
        f"## Analysis of what went wrong\n{reflection}\n\n"
        f"## Current code\n```python\n{code}\n```\n\n"
        f"## Failing test cases\n{failures_text}"
    )

    raw_response = client.generate(FIX_SYSTEM_PROMPT, fix_message)

    code = _extract_code(raw_response)
    _write_code(code)

    line_count = code.count("\n") + 1
    print(f"Fix complete: {line_count} lines written to {config.GENERATED_CODE_PATH}")

    return code, reflection
