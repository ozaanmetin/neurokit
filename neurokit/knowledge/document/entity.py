"""
neurokit.documents.entity

It contains the entity definitions and behaviors for document processing in the NeuroKit library.
"""

# Built-in imports
from typing import Optional, Any
from dataclasses import dataclass, field, asdict

from neurokit.core.utils.id import IDHelper


@dataclass
class Document:
    """
    Represents a document entity in the NeuroKit library.

    :param id: unique identifier for the document (optional) if not provided it will be generated.
    :param content: the main content of the document.
    :param content_id: an optional identifier for the content.
    :param metadata: metadata associated with the document (optional).
    """

    content: str
    content_id: Optional[str] = None
    
    id: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate a unique identifier if not provided."""

        if self.id is None:
            self.id = IDHelper.generate_document_id()

    def to_dict(self) -> dict[str, Any]:
        """
        Convert instance to a dictionary.

        :return: dictionary representation of the document.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        """
        Create a Document instance from a dictionary.

        :param data: dictionary containing document attributes.
        :return: Document instance.
        """
        return cls(**data)
    
    