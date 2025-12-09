from typing import Callable

from neurokit.knowledge.chunking.base import Chunker
from neurokit.knowledge.chunking.entity import Chunk
from neurokit.knowledge.document.entity import Document


# Default separators optimized for RAG/AI applications
DEFAULT_SEPARATORS = [
    "\n\n",      # Paragraphs - strongest semantic boundary
    "\n",        # Lines
    ". ",        # Sentences
    "? ",        # Questions
    "! ",        # Exclamations
    "; ",        # Clauses
    ", ",        # Phrases
    " ",         # Words
    "",          # Characters (fallback)
]

# Markdown-aware separators for documentation/markdown content
MARKDOWN_SEPARATORS = [
    "\n## ",     # H2 headers
    "\n### ",    # H3 headers
    "\n#### ",   # H4 headers
    "\n\n",      # Paragraphs
    "\n```\n",   # Code blocks
    "\n",        # Lines
    ". ",        # Sentences
    " ",         # Words
    "",          # Characters
]

# Code-aware separators
CODE_SEPARATORS = [
    "\nclass ",  # Class definitions
    "\ndef ",    # Function definitions
    "\n\n",      # Double newlines
    "\n",        # Single newlines
    "; ",        # Statement separators
    ", ",        # List separators
    " ",         # Words
    "",          # Characters
]


class RecursiveChunker(Chunker):
    """
    Recursive text chunker optimized for RAG and AI applications.
    
    Splits text hierarchically using semantic boundaries (paragraphs → sentences → words),
    preserving context for better retrieval and embedding quality.
    
    Features:
    - Semantic boundary preservation: Splits at natural text boundaries
    - Configurable overlap: Maintains context continuity between chunks  
    - Length function support: Use character count or token count
    - Multiple separator presets: Default, Markdown, Code
    
    Args:
        chunk_size: Maximum size of each chunk (default: 512)
        overlap: Number of units to overlap between chunks (default: 50)
        separators: List of separators in order of priority
        length_function: Function to measure text length (default: len)
        keep_separator: Whether to keep separators in the output (default: True)
        strip_whitespace: Whether to strip whitespace from chunks (default: True)
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        separators: list[str] | None = None,
        length_function: Callable[[str], int] | None = None,
        keep_separator: bool = True,
        strip_whitespace: bool = True,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or DEFAULT_SEPARATORS
        self.length_function = length_function or len
        self.keep_separator = keep_separator
        self.strip_whitespace = strip_whitespace
        self._ensure_chunk_size_and_overlap()

    @classmethod
    def for_markdown(cls, chunk_size: int = 512, overlap: int = 50, **kwargs) -> "RecursiveChunker":
        """Create a chunker optimized for markdown content."""
        return cls(chunk_size=chunk_size, overlap=overlap, separators=MARKDOWN_SEPARATORS, **kwargs)

    @classmethod
    def for_code(cls, chunk_size: int = 512, overlap: int = 50, **kwargs) -> "RecursiveChunker":
        """Create a chunker optimized for source code."""
        return cls(chunk_size=chunk_size, overlap=overlap, separators=CODE_SEPARATORS, **kwargs)

    def chunk(self, document: Document) -> list[Chunk]:
        """Chunk a document into smaller pieces."""
        text = document.content
        if not text or not text.strip():
            return []
        
        raw_chunks = self._split_text(text)
        
        # Build chunks with position tracking
        chunks = []
        current_pos = 0
        
        for idx, content in enumerate(raw_chunks):
            # Find actual position in original text
            start_pos = text.find(content, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            end_pos = start_pos + len(content)
            
            chunks.append(Chunk(
                content=content,
                document_id=document.id,
                chunk_index=idx,
                start_pos=start_pos,
                end_pos=end_pos,
            ))
            
            # Move position forward, accounting for overlap
            current_pos = start_pos + 1
        
        return chunks

    def _split_text(self, text: str) -> list[str]:
        """Split text recursively using separator hierarchy."""
        return self._split_recursive(text, self.separators)

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the separator list."""
        if not text:
            return []
        
        # Check if text fits in a single chunk
        text_len = self.length_function(text)
        if text_len <= self.chunk_size:
            result = text.strip() if self.strip_whitespace else text
            return [result] if result else []
        
        # Find the best separator to use
        separator = self._find_separator(text, separators)
        
        # Split text
        splits = self._split_by_separator(text, separator)
        
        # Get remaining separators for recursion
        remaining_separators = self._get_remaining_separators(separator, separators)
        
        # Merge splits into chunks
        return self._merge_splits(splits, remaining_separators)

    def _find_separator(self, text: str, separators: list[str]) -> str:
        """Find the first separator that exists in the text."""
        for sep in separators:
            if sep == "":
                return sep
            if sep in text:
                return sep
        return ""

    def _split_by_separator(self, text: str, separator: str) -> list[str]:
        """Split text by separator, optionally keeping the separator."""
        if separator == "":
            return list(text)
        
        parts = text.split(separator)
        
        if not self.keep_separator:
            return [p for p in parts if p]
        
        # Keep separator attached to the preceding part
        result = []
        for i, part in enumerate(parts):
            if i < len(parts) - 1:
                result.append(part + separator)
            elif part:  # Last part, only add if not empty
                result.append(part)
        
        return result

    def _get_remaining_separators(self, current_sep: str, separators: list[str]) -> list[str]:
        """Get separators that come after the current one."""
        if current_sep not in separators:
            return separators
        idx = separators.index(current_sep)
        return separators[idx + 1:] if idx + 1 < len(separators) else [""]

    def _merge_splits(self, splits: list[str], remaining_separators: list[str]) -> list[str]:
        """Merge splits into chunks respecting size limits."""
        chunks = []
        current_chunks: list[str] = []
        current_length = 0
        
        for split in splits:
            split_length = self.length_function(split)
            
            # If single split exceeds chunk size, recursively split it
            if split_length > self.chunk_size:
                # Save current accumulated content first
                if current_chunks:
                    chunk_content = self._join_chunks(current_chunks)
                    if chunk_content:
                        chunks.append(chunk_content)
                    current_chunks = []
                    current_length = 0
                
                # Recursively split the large piece
                sub_chunks = self._split_recursive(split, remaining_separators)
                chunks.extend(sub_chunks)
                continue
            
            # Check if adding this split would exceed chunk size
            if current_length + split_length > self.chunk_size:
                # Save current chunk
                chunk_content = self._join_chunks(current_chunks)
                if chunk_content:
                    chunks.append(chunk_content)
                
                # Start new chunk with overlap from previous
                overlap_content = self._get_overlap(current_chunks)
                current_chunks = [overlap_content, split] if overlap_content else [split]
                current_length = self.length_function("".join(current_chunks))
            else:
                current_chunks.append(split)
                current_length += split_length
        
        # Don't forget the last chunk
        if current_chunks:
            chunk_content = self._join_chunks(current_chunks)
            if chunk_content:
                chunks.append(chunk_content)
        
        return chunks

    def _join_chunks(self, chunks: list[str]) -> str:
        """Join chunk parts into a single string."""
        content = "".join(chunks)
        if self.strip_whitespace:
            content = content.strip()
        return content

    def _get_overlap(self, chunks: list[str]) -> str:
        """Get overlap content from the end of chunk list."""
        if not chunks or self.overlap <= 0:
            return ""
        
        # Collect from end until we reach overlap size
        overlap_parts = []
        current_length = 0
        
        for chunk in reversed(chunks):
            chunk_len = self.length_function(chunk)
            
            if current_length + chunk_len <= self.overlap:
                overlap_parts.insert(0, chunk)
                current_length += chunk_len
            elif not overlap_parts:
                # First chunk is larger than overlap - take from the end
                if chunk_len > self.overlap:
                    # Approximate: take last N characters/tokens
                    # For character-based, this is exact; for tokens, it's approximate
                    overlap_parts.append(chunk[-self.overlap:])
                else:
                    overlap_parts.append(chunk)
                break
            else:
                break
        
        return "".join(overlap_parts)
