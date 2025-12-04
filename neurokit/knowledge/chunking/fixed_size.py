from neurokit.knowledge.chunking.base import Chunker

from neurokit.knowledge.chunking.entity import Chunk
from neurokit.knowledge.document.entity import Document


class FixedSizeChunker(Chunker):
    """
    Chunks documents into fixed-size pieces with optional overlap.
    
    This chunker splits text by character count, which is useful for
    maintaining consistent chunk sizes for embedding models or LLMs
    with token limits.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 0, clean: bool = True):
        """
        Initialize the FixedSizeChunker.

        :param chunk_size: The size of each chunk in characters.
        :param overlap: The number of overlapping characters between chunks.
        """

        # initialize parameters
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.clean = clean
        # ensure valid chunk size and overlap values
        self._ensure_chunk_size_and_overlap()
        
    def chunk(self, document: Document) -> list[Chunk]:
        """
        Chunks a document into smaller documents of fixed size.

        :param document: The document to be chunked.
        :return: A list of chunked documents.
        """

        text = document.content

        # clean text if asked
        if self.clean:
            text = self.clean_text(text)

        text_length = len(text)
        start = 0
        chunks: list[Chunk] = []

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk_text = text[start:end]
            chunk_index = len(chunks)

            chunk = Chunk(
                content=chunk_text,
                document_id=str(document.id),
                chunk_index=chunk_index,
                start_pos=start,
                end_pos=end,
            )

            chunks.append(chunk)
            start += end - self.overlap if end < text_length else end

        return chunks



