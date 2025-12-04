from dataclasses import dataclass
from typing import Optional

from neurokit.core.utils.id import IDHelper


@dataclass
class Chunk:
    content: str
    document_id: str
    chunk_index: int
    start_pos: int
    end_pos: int

    id: Optional[str] = None

    def __post_init__(self):
        """Generate a unique identifier if not provided."""
        if self.id is None:
            self.id = IDHelper.generate_chunk_id(document_id=self.document_id, chunk_index=self.chunk_index)