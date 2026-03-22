import json
import os
import time

import requests

import config
from llm.base import LLMClient

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
MAX_RETRIES = 5
INITIAL_BACKOFF = 2


class OpenAIClient(LLMClient):
    """LLMClient implementation using the OpenAI REST API."""

    def __init__(self, model: str = config.OPENAI_MODEL_NAME):
        self._model_name = model
        self._api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set."
            )

    def generate(self, system_prompt: str, user_message: str) -> str:
        """Send a prompt to OpenAI and return the raw text response."""
        payload = {
            "model": self._model_name,
            "max_tokens": 4096,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }

        last_error = None
        for attempt in range(MAX_RETRIES):
            resp = requests.post(
                OPENAI_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                json=payload,
                timeout=120,
            )
            if resp.status_code == 429:
                body = resp.json() if resp.headers.get("Content-Type", "").startswith("application/json") else {}
                error_msg = body.get("error", {}).get("message", resp.text)
                wait = INITIAL_BACKOFF * (2 ** attempt)
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    wait = max(wait, int(retry_after))
                print(f"  Rate limited (429): {error_msg}")
                print(f"  Retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait)
                last_error = resp
                continue
            resp.raise_for_status()
            break
        else:
            raise requests.exceptions.HTTPError(
                f"429 Too Many Requests after {MAX_RETRIES} retries. "
                f"Check your OpenAI API quota/billing at https://platform.openai.com/settings/organization/billing",
                response=last_error,
            )

        data = resp.json()

        choices = data.get("choices", [])
        if not choices:
            raise ValueError(
                f"OpenAI API returned no choices. Response: {json.dumps(data, indent=2)}"
            )
        content = choices[0].get("message", {}).get("content")
        if not content:
            raise ValueError(
                f"OpenAI API returned no content. Response: {json.dumps(data, indent=2)}"
            )
        return content

    def model_name(self) -> str:
        """Return the identifier string of the underlying model."""
        return self._model_name
