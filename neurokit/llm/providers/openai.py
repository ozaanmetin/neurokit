import tiktoken
from typing import Optional, Iterator

from neurokit.llm.providers.base import LLMProvider
from neurokit.llm.entity import (
    LLMMessage, 
    LLMResponse, 
    StreamingChunk, 
    TokenUsage
)


class OpenAIProvider(LLMProvider):
    """
    OpenAI Language Model Provider.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(model, max_tokens, temperature, **kwargs)
        
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required to use OpenAIProvider. "
                "Please install it via 'pip install openai'."
            )
        
        self.client = OpenAI(api_key=api_key)

    def generate(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        # Implementation for generating a response using OpenAI API
        token_usage = None
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        temperature = kwargs.pop("temperature", self.temperature)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[message.to_dict() for message in messages],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        if response.usage:
            token_usage = TokenUsage(
                prompt=response.usage.prompt_tokens,
                completion=response.usage.completion_tokens,
                total=response.usage.total_tokens,
            )

        choice = response.choices[0]

        return LLMResponse(
            content=choice.message.content or "",
            model=self.model,
            usage=token_usage,
            metadata={"response_id": response.id},
        )

    def stream(self, messages: list[LLMMessage], **kwargs) -> Iterator[StreamingChunk]:
        # Implementation for streaming a response using OpenAI API
        temperature = kwargs.pop("temperature", self.temperature)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[message.to_dict() for message in messages],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **kwargs,
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield StreamingChunk(
                    content=content,
                    metadata={"response_id": chunk.id}
                )

    def count_tokens(self, text: str) -> int:
        try:
            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(text))
        except Exception:
            # Fallback: rough estimate
            return len(text) // 4


