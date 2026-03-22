import json
import os

import requests

import config
from llm.base import LLMClient

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


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
            }
        }
        resp = requests.post(
            url,
            params={"key": self._api_key},
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError(
                f"Gemini API returned no candidates. Response: {json.dumps(data, indent=2)}"
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
