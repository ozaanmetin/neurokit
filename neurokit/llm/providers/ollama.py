from typing import Optional, Iterator


from neurokit.llm.providers.base import LLMProvider
from neurokit.llm.entity import (
    LLMMessage,
    LLMResponse,
    StreamingChunk,
    TokenUsage
)


class OllamaLLMProvider(LLMProvider):
    """
    Ollama Language Model Provider.

    Note: Ollama runs locally and doesn't require an API key.
    Make sure Ollama is running (e.g., via Docker or locally on port 11434).
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        super().__init__(model, max_tokens, temperature, **kwargs)

        try:
            import ollama
        except ImportError:
            raise ImportError(
                "The 'ollama' package is required to use OllamaLLMProvider. "
                "Please install it via 'pip install ollama'."
            )

        self.ollama = ollama
        self.base_url = base_url
        # Configure the client to use the specified base URL
        self.client = ollama.Client(host=base_url)

    def generate(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        """Generate a response using Ollama API."""
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        temperature = kwargs.pop("temperature", self.temperature)

        # Build options dict for Ollama
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        # Convert messages to Ollama format
        ollama_messages = [
            {"role": message.role, "content": message.content}
            for message in messages
        ]

        response = self.client.chat(
            model=self.model,
            messages=ollama_messages,
            options=options if options else None,
            stream=False,
            **kwargs,
        )

        import pprint
        pprint.pprint(response)
        # Ollama response structure:
        # {
        #   'message': {'role': 'assistant', 'content': '...'},
        #   'model': 'llama2',
        #   'created_at': '...',
        #   'done': True,
        #   'total_duration': ...,
        #   'prompt_eval_count': ...,
        #   'eval_count': ...,
        # }

        token_usage = None
        if "prompt_eval_count" in response and "eval_count" in response:
            token_usage = TokenUsage(
                prompt=response.get("prompt_eval_count", 0),
                completion=response.get("eval_count", 0),
                total=response.get("prompt_eval_count", 0) + response.get("eval_count", 0),
            )

        return LLMResponse(
            content=response["message"]["content"],
            model=self.model,
            usage=token_usage,
            metadata={
                "created_at": response.get("created_at"),
                "total_duration": response.get("total_duration"),
            },
        )

    def stream(self, messages: list[LLMMessage], **kwargs) -> Iterator[StreamingChunk]:
        """Stream a response using Ollama API."""
        temperature = kwargs.pop("temperature", self.temperature)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)

        # Build options dict for Ollama
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        # Convert messages to Ollama format
        ollama_messages = [
            {"role": message.role, "content": message.content}
            for message in messages
        ]

        stream = self.client.chat(
            model=self.model,
            messages=ollama_messages,
            options=options if options else None,
            stream=True,
            **kwargs,
        )

        for chunk in stream:
            if chunk.get("message") and chunk["message"].get("content"):
                yield StreamingChunk(
                    content=chunk["message"]["content"],
                    metadata={
                        "model": chunk.get("model"),
                        "created_at": chunk.get("created_at"),
                    }
                )

    def count_tokens(self, text: str) -> int:
        """
        Count tokens for the given text.

        Note: Ollama doesn't provide a direct tokenization API,
        so we use a rough estimate of ~4 characters per token.
        For more accurate counts, you would need to use the model's
        specific tokenizer.
        """
        # Rough estimate: ~4 characters per token
        return len(text) // 4
