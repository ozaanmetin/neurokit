from typing import Callable

from neurokit.knowledge.chunking.base import Chunker
from neurokit.knowledge.chunking.entity import Chunk
from neurokit.knowledge.document.entity import Document


class FixedSizeChunker(Chunker):
    """
    Chunks documents into fixed-size pieces with optional overlap.
    
    Splits text by a configurable length function (characters or tokens),
    useful for embedding models or LLMs with size limits.
    
    Features:
    - Character or token-based sizing via length_function
    - Configurable overlap for context continuity
    - Optional whitespace stripping
    
    Args:
        chunk_size: Maximum size of each chunk (default: 512)
        overlap: Number of units to overlap between chunks (default: 50)
        length_function: Function to measure text length (default: len for characters)
        strip_whitespace: Whether to strip whitespace from chunks (default: True)
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        length_function: Callable[[str], int] | None = None,
        strip_whitespace: bool = True,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.length_function = length_function or len
        self.strip_whitespace = strip_whitespace
        self._ensure_chunk_size_and_overlap()

    def chunk(self, document: Document) -> list[Chunk]:
        """Chunk a document into fixed-size pieces."""
        text = document.content
        if not text or not text.strip():
            return []

        chunks: list[Chunk] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Find the end position that fits within chunk_size
            end = self._find_chunk_end(text, start)
            
            chunk_text = text[start:end]
            if self.strip_whitespace:
                chunk_text = chunk_text.strip()
            
            if chunk_text:  # Only add non-empty chunks
                chunks.append(Chunk(
                    content=chunk_text,
                    document_id=str(document.id),
                    chunk_index=len(chunks),
                    start_pos=start,
                    end_pos=end,
                ))

            # Move start position, accounting for overlap
            if end >= text_length:
                break
            
            # Calculate overlap in original text positions
            overlap_start = self._find_overlap_start(text, start, end)
            start = overlap_start

        return chunks

    def _find_chunk_end(self, text: str, start: int) -> int:
        """Find the end position for a chunk starting at 'start'."""
        text_length = len(text)
        
        # If using character-based length, simple slice
        if self.length_function is len:
            return min(start + self.chunk_size, text_length)
        
        # For token-based, binary search for the right end position
        low, high = start, min(start + self.chunk_size * 4, text_length)  # Estimate max chars
        
        # Quick check if remaining text fits
        if self.length_function(text[start:high]) <= self.chunk_size:
            return high
        
        while low < high:
            mid = (low + high + 1) // 2
            if self.length_function(text[start:mid]) <= self.chunk_size:
                low = mid
            else:
                high = mid - 1
        
        return low if low > start else min(start + 1, text_length)

    def _find_overlap_start(self, text: str, chunk_start: int, chunk_end: int) -> int:
        """Find the start position for the next chunk considering overlap."""
        if self.overlap <= 0:
            return chunk_end
        
        # If using character-based length, simple calculation
        if self.length_function is len:
            return max(chunk_start + 1, chunk_end - self.overlap)
        
        # For token-based, find position where overlap tokens fit
        target_advance = self.chunk_size - self.overlap
        
        # Binary search for position that gives us target_advance worth of content
        low, high = chunk_start, chunk_end
        
        while low < high:
            mid = (low + high + 1) // 2
            if self.length_function(text[chunk_start:mid]) <= target_advance:
                low = mid
            else:
                high = mid - 1
        
        return max(chunk_start + 1, low)

