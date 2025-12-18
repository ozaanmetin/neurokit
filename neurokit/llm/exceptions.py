from neurokit.core.exceptions import NeuroKitError


class LLMError(NeuroKitError):
    """Base class for exceptions in the LLM module."""


class LLMRateLimitError(LLMError):
    default_message = "Rate limit exceeded for LLM API."


class TokenLimitExceeded(LLMError):
    """Exception raised when the token limit is exceeded."""
    def __init__(self, tokens: int, limit: int, model: str):
        super().__init__(
            message="Token limit exceeded",
            details={"tokens": tokens, "limit": limit, "model": model}
        )
