from enum import Enum


class LLMRole(str, Enum):
    """
    Enum representing different roles in a language model conversation.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"