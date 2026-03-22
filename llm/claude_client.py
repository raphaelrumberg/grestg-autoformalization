import anthropic

import config
from llm.base import LLMClient


class ClaudeClient(LLMClient):
    """LLMClient implementation using Claude API."""

    def __init__(self, model: str = config.MODEL_NAME):
        self._model = model
        self._client = anthropic.Anthropic()

    def generate(self, system_prompt: str, user_message: str) -> str:
        """Send a prompt to Claude and return the raw text response."""
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        if not response.content or not response.content[0].text:
            raise ValueError(
                f"Claude API returned no text content. "
                f"Stop reason: {response.stop_reason}"
            )
        return response.content[0].text

    def model_name(self) -> str:
        """Return the identifier string of the underlying model."""
        return self._model
