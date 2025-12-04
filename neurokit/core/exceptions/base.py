"""
neurokit.core.exceptions

It contains the custom exceptions for the NeuroKit library.
Provide the common exceptions which can be raised throughout the library.
"""

from typing import Any, Optional


class NeuroKitError(Exception):
    """
    Base exception class for all NeuroKit errors.
    """

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        return f"{self.message} ({self.details_str})" if self.details else self.message
    
    @property
    def details_str(self) -> str:
        if not self.details:
            return ""
        return " | " + ", ".join(f"{key}={value}" for key, value in self.details.items())
