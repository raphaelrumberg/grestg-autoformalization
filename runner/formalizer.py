import re

import config
from llm.base import LLMClient

SYSTEM_PROMPT = (
    "You are a legal formalization expert. Generate only valid Python code. "
    "Your entire response must be a single ```python code block. "
    "Start with 'from typing import Optional, Tuple' and then define exactly one "
    "function: def check_anteilsvereinigung(graph: dict, "
    "target_entity: str, acquirer_groups: list = None) -> Tuple[bool, Optional[str]]"
)


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

    # Extract Python code block; fall back to full response if no fences
    match = re.search(r"```python\s*\n(.*?)```", raw_response, re.DOTALL)
    if match:
        code = match.group(1).strip()
    else:
        # Strip any leftover fence markers
        code = raw_response.strip()
        code = re.sub(r"^```(?:python)?\s*\n?", "", code)
        code = re.sub(r"\n?```\s*$", "", code)

    with open(config.GENERATED_CODE_PATH, "w", encoding="utf-8") as f:
        f.write(code + "\n")

    line_count = code.count("\n") + 1
    print(f"Formalization complete: {line_count} lines written to {config.GENERATED_CODE_PATH}")

    return code
