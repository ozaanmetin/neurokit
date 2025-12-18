from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from neurokit.llm.enums import LLMRole


@dataclass
class TokenUsage:
    """Class to represent token usage statistics."""
    
    prompt: int = 0
    completion: int = 0
    total: int = 0

    def to_dict(self) -> dict[str, int]:
        """Convert TokenUsage to a dictionary."""
        return asdict(self) 
    

@dataclass
class LLMMessage:
    """Class to represent a message in a language model conversation."""
    
    role: LLMRole
    content: str
    name: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert LLMMessage to a dictionary."""
        return asdict(self)
    

@dataclass
class LLMResponse:
    """Class to represent a response from a language model."""
    
    content: str
    model: str
    usage: Optional[TokenUsage] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert LLMResponse to a dictionary."""
        response_dict = asdict(self)
        if self.usage:
            response_dict["usage"] = self.usage.to_dict()
        return response_dict
    

@dataclass
class StreamingChunk:
    """Class to represent a chunk of streaming response from a language model."""
    
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert StreamingChunk to a dictionary."""
        return asdict(self)