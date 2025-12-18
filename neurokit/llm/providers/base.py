from abc import ABC, abstractmethod
from typing import Optional, Any


from neurokit.llm.enums import LLMRole
from neurokit.llm.entity import LLMMessage, LLMResponse, StreamingChunk


class LLMProvider(ABC):
    """Abstract base class for Language Model Providers."""

    def __init__(
        self,
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.config = kwargs

    @abstractmethod
    def generate(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        """Generate a response from the language model based on the provided messages."""
        pass

    @abstractmethod
    def stream(self, messages: list[LLMMessage], **kwargs) -> StreamingChunk:
        """Stream a response from the language model based on the provided messages."""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text for the specific model."""
        pass

    def generate_text(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response from the language model based on a text prompt."""
        message = LLMMessage(role=LLMRole.USER.value, content=prompt)
        return self.generate(messages=[message], **kwargs)

    def get_model_info(self) -> dict[str, Any]:
        """Retrieve information about the language model."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "config": self.config,
        }
