from pydoc import text
import re
import unicodedata

from abc import ABC, abstractmethod
from neurokit.knowledge.document.entity import Document
from neurokit.knowledge.chunking.entity import Chunk


class Chunker(ABC):
    """
    Interface for chunking documents into smaller parts.
    Protocol uses duck typing to define the expected methods for chunking.

    """

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Chunks a document into smaller documents.
        
        :param document: The document to be chunked.
        :return: A list of chunked documents.
        """

    def _ensure_chunk_size_and_overlap(self):
        """
        Ensure chunk_size and overlap have valid values.
        """
        self.chunk_size = max(1, self.chunk_size)
        self.overlap = max(0, min(self.overlap, self.chunk_size - 1))
    