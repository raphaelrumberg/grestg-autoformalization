from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Interface for LLMs used in autoformalization."""

    @abstractmethod
    def generate(self, system_prompt: str, user_message: str) -> str:
        """
        Send a prompt to the LLM and return the raw text response.

        Args:
            system_prompt: Instructions setting the model's role and output format.
            user_message: The actual content to process (statutory text + prompt).

        Returns:
            The model's raw text response.
        """
        pass

    @abstractmethod
    def model_name(self) -> str:
        """Return the identifier string of the underlying model."""
        pass
