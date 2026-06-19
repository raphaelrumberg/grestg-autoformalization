import json
import os
import time

import requests

import config
from llm.base import LLMClient

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
MAX_RETRIES = 5
INITIAL_BACKOFF = 2


class GeminiClient(LLMClient):
    """LLMClient implementation using the Gemini REST API."""

    def __init__(self, model: str = config.GEMINI_MODEL_NAME):
        self._model_name = model
        self._api_key = os.environ.get("GEMINI_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. "
                "Get an API key from https://aistudio.google.com/apikey"
            )

    def generate(self, system_prompt: str, user_message: str) -> str:
        """Send a prompt to Gemini and return the raw text response."""
        url = GEMINI_API_URL.format(model=self._model_name)
        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_message}]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 8192,
                "thinkingConfig": {"thinkingBudget": 0},
            }
        }
        last_error = None
        for attempt in range(MAX_RETRIES):
            resp = requests.post(
                url,
                params={"key": self._api_key},
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=120,
            )
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = INITIAL_BACKOFF * (2 ** attempt)
                print(f"  Gemini API {resp.status_code}; retrying in {wait}s "
                      f"(attempt {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait)
                last_error = resp
                continue
            resp.raise_for_status()
            break
        else:
            raise requests.exceptions.HTTPError(
                f"Gemini API still failing after {MAX_RETRIES} retries "
                f"(last status {last_error.status_code}).",
                response=last_error,
            )
        data = resp.json()

        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError(
                f"Gemini API returned no candidates. Response: {json.dumps(data, indent=2)}"
            )
        finish_reason = candidates[0].get("finishReason", "")
        if finish_reason == "MAX_TOKENS":
            raise ValueError(
                "Gemini response truncated (finishReason=MAX_TOKENS); "
                "refusing to return partial code."
            )
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts or not parts[0].get("text"):
            raise ValueError(
                f"Gemini API returned no text content. Response: {json.dumps(data, indent=2)}"
            )
        return parts[0]["text"]

    def model_name(self) -> str:
        """Return the identifier string of the underlying model."""
        return self._model_name
