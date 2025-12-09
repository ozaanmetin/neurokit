import re
from typing import Callable

from neurokit.knowledge.chunking.base import Chunker
from neurokit.knowledge.chunking.entity import Chunk
from neurokit.knowledge.document.entity import Document


class SemanticChunker(Chunker):
    """
    Chunks documents based on semantic similarity between sentences.
    
    Uses embedding similarity to find natural breakpoints, grouping
    semantically related content together. This produces higher quality
    chunks for RAG applications compared to fixed-size splitting.
    
    How it works:
     - Splits text into sentences
     - Creates sentence groups (using buffer_size for context)
     - Computes embeddings for each group
     - Calculates similarity between adjacent groups
     - Splits at points where similarity drops below threshold
    
    Args:
        embedding_function: Function that takes list of texts and returns embeddings.
        buffer_size: Number of sentences to combine for comparison (default: 1)
        breakpoint_percentile_threshold: Percentile of distance to use as breakpoint. (higher = fewer breaks) (default: 95)
        min_chunk_size: Minimum characters per chunk. Small chunks are merged. (default: 100)
        max_chunk_size: Maximum characters per chunk. Large chunks are split. (default: None)
    """

    def __init__(
        self,
        embedding_function: Callable[[list[str]], list[list[float]]],
        buffer_size: int = 1,
        breakpoint_percentile_threshold: int = 95,
        min_chunk_size: int = 100,
        max_chunk_size: int | None = None,
    ):
        self.embedding_function = embedding_function
        self.buffer_size = buffer_size
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        """Chunk a document based on semantic similarity."""
        text = document.content
        if not text or not text.strip():
            return []
        
        # Split into sentences
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [Chunk(
                content=text.strip(),
                document_id=document.id,
                chunk_index=0,
                start_pos=0,
                end_pos=len(text),
            )]
        
        # Find semantic breakpoints
        breakpoints = self._find_breakpoints(sentences)
        
        # Create chunks from breakpoints
        raw_chunks = self._create_chunks_from_breakpoints(sentences, breakpoints)
        
        # Merge small chunks and split large ones if needed
        raw_chunks = self._enforce_chunk_sizes(raw_chunks)
        
        # Build Chunk objects with position tracking
        return self._build_chunk_objects(document, text, raw_chunks)

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex."""
        # Pattern handles common sentence endings while preserving abbreviations
        pattern = r'(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ])|(?<=[.!?])\s*\n+'
        
        sentences = re.split(pattern, text)
        
        # Filter empty and clean up
        return [s.strip() for s in sentences if s.strip()]

    def _find_breakpoints(self, sentences: list[str]) -> list[int]:
        """Find indices where semantic similarity drops significantly."""
        if len(sentences) <= 2:
            return []
        
        # Create sentence groups with buffer
        groups = self._create_sentence_groups(sentences)

        if len(groups) < 2:
            return []
        
        # Get embeddings for all groups
        embeddings = self.embedding_function(groups)
        
        # Calculate distances between adjacent groups
        distances = []
        for i in range(len(embeddings) - 1):
            dist = self._cosine_distance(embeddings[i], embeddings[i + 1])
            distances.append(dist)
        
        if not distances:
            return []
        
        # Find threshold based on percentile
        threshold = self._percentile(distances, self.breakpoint_percentile_threshold)
        # Find breakpoints where distance exceeds threshold
        breakpoints = []
        for i, dist in enumerate(distances):
            if dist >= threshold:
                # Breakpoint is after sentence at index (i + buffer_size)
                breakpoint_idx = i + self.buffer_size
                if breakpoint_idx < len(sentences):
                    breakpoints.append(breakpoint_idx)
        
        return breakpoints

    def _create_sentence_groups(self, sentences: list[str]) -> list[str]:
        """Create sentence groups for embedding comparison."""
        groups = []
        for i in range(len(sentences) - self.buffer_size):
            # Combine buffer_size sentences for context
            group = " ".join(sentences[i:i + self.buffer_size + 1])
            groups.append(group)
        return groups

    def _cosine_distance(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine distance between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 1.0  # Maximum distance
        
        similarity = dot_product / (norm1 * norm2)
        return 1.0 - similarity

    def _percentile(self, values: list[float], percentile: int) -> float:
        """Calculate percentile of a list of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * (percentile / 100)
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_values) else f
        
        if f == c:
            return sorted_values[f]
        
        return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])

    def _create_chunks_from_breakpoints(
        self, 
        sentences: list[str], 
        breakpoints: list[int]
    ) -> list[str]:
        """Create chunk texts from sentences and breakpoints."""
        if not breakpoints:
            return [" ".join(sentences)]
        
        chunks = []
        start = 0
        
        for bp in breakpoints:
            chunk_sentences = sentences[start:bp]
            if chunk_sentences:
                chunks.append(" ".join(chunk_sentences))
            start = bp
        
        # Add remaining sentences
        if start < len(sentences):
            chunks.append(" ".join(sentences[start:]))
        
        return chunks

    def _enforce_chunk_sizes(self, chunks: list[str]) -> list[str]:
        """Merge small chunks and split large ones."""
        if not chunks:
            return chunks
        
        # Merge small chunks
        merged = []
        current = ""
        
        for chunk in chunks:
            if len(current) + len(chunk) < self.min_chunk_size:
                current = (current + " " + chunk).strip() if current else chunk
            else:
                if current:
                    merged.append(current)
                current = chunk
        
        if current:
            merged.append(current)
        
        # Split large chunks if max_chunk_size is set
        if self.max_chunk_size:
            final = []
            for chunk in merged:
                if len(chunk) > self.max_chunk_size:
                    # Simple split by sentences for large chunks
                    sentences = self._split_sentences(chunk)
                    current = ""
                    for sent in sentences:
                        if len(current) + len(sent) > self.max_chunk_size and current:
                            final.append(current.strip())
                            current = sent
                        else:
                            current = (current + " " + sent).strip() if current else sent
                    if current:
                        final.append(current)
                else:
                    final.append(chunk)
            return final
        
        return merged

    def _build_chunk_objects(
        self, 
        document: Document, 
        original_text: str, 
        chunks: list[str]
    ) -> list[Chunk]:
        """Build Chunk objects with position tracking."""
        result = []
        search_start = 0
        
        for idx, content in enumerate(chunks):
            # Find position in original text
            # Use first few words to locate since content might be modified
            search_text = content[:50] if len(content) > 50 else content
            start_pos = original_text.find(search_text, search_start)
            
            if start_pos == -1:
                start_pos = search_start
            
            end_pos = start_pos + len(content)
            search_start = start_pos + 1
            
            result.append(Chunk(
                content=content,
                document_id=document.id,
                chunk_index=idx,
                start_pos=start_pos,
                end_pos=end_pos,
            ))
        
        return result
