from neurokit.knowledge.chunking.base import Chunker
from neurokit.knowledge.chunking.entity import Chunk
from neurokit.knowledge.document.entity import Document


class RecursiveChunker(Chunker):
    """
    Chunks documents using recursive text splitting with custom separators.

    This chunker splits text hierarchically using a list of separators,
    attempting to keep semantically related content together. It tries
    larger separators first (e.g., paragraphs) and recursively falls back
    to smaller ones (e.g., sentences, words) when chunks are too large.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 0,
        clean: bool = True,
        separators: list[str] = None
    ):
        """
        Initialize the RecursiveChunker.

        :param chunk_size: The maximum size of each chunk in characters.
        :param overlap: The number of overlapping characters between chunks.
        :param clean: Whether to clean the text before chunking.
        :param separators: List of separators to use for splitting, in order of priority.
        """
        
        # initialize parameters
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.clean = clean
        self.seperators = separators if separators is not None else ['\n\n', '\n', '.', ' ', '']
        # ensure valid chunk size and overlap values
        self._ensure_chunk_size_and_overlap()

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Chunks a document into smaller documents using recursive splitting.

        :param document: The document to be chunked.
        :return: A list of chunked documents.
        """
        text = self._prepare_text(document.content)
        raw_chunks = self._create_raw_chunks(text)
        return self._convert_to_chunk_objects(raw_chunks, document.id)

    def _prepare_text(self, text: str) -> str:
        """
        Clean text if cleaning is enabled.

        :param text: The text to prepare.
        :return: The prepared (possibly cleaned) text.
        """
        # clean text if asked
        if self.clean:
            return self.clean_text(text)
        return text

    def _create_raw_chunks(self, text: str) -> list[tuple[str, int, int]]:
        """
        Create raw chunks with their positions.

        :param text: The text to chunk.
        :return: List of tuples containing (content, start_pos, end_pos).
        """
        # get the first seperator that works
        seperator = self._select_initial_seperator(text, self.seperators)
        splits = self._split_text(text, seperator)
        
        chunks = []
        current_texts = []
        current_length = 0
        current_start_pos = 0
        
        position = 0
        
        for split in splits:
            split_length = len(split)
            
            if split_length > self.chunk_size:
                # If we have accumulated text, save it as a chunk first
                if current_texts:
                    chunks.append(self._create_chunk_tuple(current_texts, current_start_pos))
                    current_texts = []
                    current_length = 0
                
                # Handle oversized split
                oversized_chunks = self._handle_oversized_split(split, position)
                chunks.extend(oversized_chunks)
                
            elif current_length + split_length <= self.chunk_size:
                # Add to current chunk
                if not current_texts:
                    current_start_pos = position
                current_texts.append(split)
                current_length += split_length
                
            else:
                # Save current chunk and start new one
                chunks.append(self._create_chunk_tuple(current_texts, current_start_pos))
                
                # Handle overlap
                overlap_data = self._create_overlap(current_texts, position)
                current_texts = overlap_data['texts']
                current_length = overlap_data['length']
                current_start_pos = overlap_data['start_pos']
                
                # Add new split
                current_texts.append(split)
                current_length += split_length
            
            position += split_length
        
        # Don't forget the last chunk
        if current_texts:
            chunks.append(self._create_chunk_tuple(current_texts, current_start_pos))
        
        return chunks

    def _create_chunk_tuple(self, texts: list[str], start_pos: int) -> tuple[str, int, int]:
        """
        Create a chunk tuple from accumulated texts.

        :param texts: List of text segments to join.
        :param start_pos: Starting position of the chunk in the original text.
        :return: Tuple containing (content, start_pos, end_pos).
        """
        content = ''.join(texts)
        end_pos = start_pos + len(content)
        return (content, start_pos, end_pos)

    def _create_overlap(self, current_texts: list[str], position: int) -> dict:
        """
        Create overlap data for the next chunk.

        :param current_texts: Current accumulated text segments.
        :param position: Current position in the text.
        :return: Dictionary with keys 'texts', 'length', and 'start_pos'.
        """
        if self.overlap <= 0:
            return {
                'texts': [],
                'length': 0,
                'start_pos': position
            }
        
        full_text = ''.join(current_texts)
        overlap_text = full_text[-self.overlap:]
        overlap_start = position - len(overlap_text)
        
        return {
            'texts': [overlap_text],
            'length': len(overlap_text),
            'start_pos': overlap_start
        }

    def _handle_oversized_split(self, split: str, position: int) -> list[tuple[str, int, int]]:
        """
        Handle a split that's too large by recursively chunking it.

        :param split: The oversized text segment.
        :param position: Current position in the text.
        :return: List of chunk tuples created from the oversized split.
        """
        # Recursively chunk the oversized split with next separator
        remaining_separators = self._get_remaining_separators(self.seperators[0])
        
        if remaining_separators:
            return self._recursive_split(split, remaining_separators, position)
        else:
            # No more separators, force chunk by character
            return self._force_split(split, position)

    def _get_remaining_separators(self, current_separator: str) -> list[str]:
        """
        Get the list of separators after the current one.

        :param current_separator: The current separator being used.
        :return: List of remaining separators.
        """
        try:
            current_index = self.seperators.index(current_separator)
            return self.seperators[current_index + 1:]
        except ValueError:
            return []

    def _recursive_split(self, text: str, separators: list[str], start_pos: int) -> list[tuple[str, int, int]]:
        """
        Recursively split text using remaining separators.

        :param text: The text to split.
        :param separators: List of separators to try.
        :param start_pos: Starting position in the original text.
        :return: List of chunk tuples.
        """
        separator = self._select_initial_seperator(text, separators)
        splits = self._split_text(text, separator)
        
        chunks = []
        position = start_pos
        
        for split in splits:
            if len(split) > self.chunk_size:
                # Continue recursing or force chunk
                remaining_seps = self._get_remaining_separators(separator)
                if remaining_seps:
                    sub_chunks = self._recursive_split(split, remaining_seps, position)
                    chunks.extend(sub_chunks)
                else:
                    forced_chunks = self._force_split(split, position)
                    chunks.extend(forced_chunks)
            else:
                end_pos = position + len(split)
                chunks.append((split, position, end_pos))
            
            position += len(split)
        
        return chunks

    def _force_split(self, text: str, start_pos: int) -> list[tuple[str, int, int]]:
        """
        Force split text at chunk_size boundaries.

        :param text: The text to split.
        :param start_pos: Starting position in the original text.
        :return: List of chunk tuples.
        """
        chunks = []
        position = start_pos
        
        for i in range(0, len(text), self.chunk_size):
            chunk_text = text[i:i + self.chunk_size]
            end_pos = position + len(chunk_text)
            chunks.append((chunk_text, position, end_pos))
            position = end_pos
        
        return chunks

    def _convert_to_chunk_objects(self, raw_chunks: list[tuple[str, int, int]], document_id: str) -> list[Chunk]:
        """
        Convert raw chunks to Chunk objects.

        :param raw_chunks: List of tuples containing (content, start_pos, end_pos).
        :param document_id: The ID of the document being chunked.
        :return: List of Chunk objects.
        """
        return [
            Chunk(
                content=content,
                document_id=document_id,
                chunk_index=idx,
                start_pos=start_pos,
                end_pos=end_pos
            )
            for idx, (content, start_pos, end_pos) in enumerate(raw_chunks)
        ]

    def _split_text(self, text: str, seperator: str):
        """
        Split text by separator and reattach separators.

        :param text: The text to split.
        :param seperator: The separator to use for splitting.
        :return: List of text segments with separators reattached.
        """
        if seperator == '':
            return list(text)
        
        splits = text.split(seperator)
        return self._reattach_separator(splits, seperator)

    def _reattach_separator(self, splits: list[str], separator: str) -> list[str]:
        """
        Reattach separator to splits (except last one if empty).

        :param splits: List of text segments after splitting.
        :param separator: The separator to reattach.
        :return: List of text segments with separators reattached.
        """
        splits_count = len(splits)
        # Reattach separator to splits (except last one if it's empty)
        results = []
        for i, split in enumerate(splits):
            if i < splits_count - 1 and split:
                results.append(split + separator)
            elif split:
                results.append(split)
        
        return results
    
    def _join_splits(self, splits: list[str], seperator: str) -> str:
        """
        Join splits using a separator.

        :param splits: List of text segments to join.
        :param seperator: The separator to use for joining.
        :return: Joined text string.
        """
        return seperator.join(splits)

    def _select_initial_seperator(self, text: str, separators: list[str]) -> str:
        """
        Select the first separator that exists in the text.

        :param text: The text to search for separators.
        :param separators: List of separators to try.
        :return: The first matching separator, or empty string if none found.
        """
        # character level seperator
        sep = ''
        # find the first separator that can be used
        for sep in separators:
            if sep == '' or sep in text:
                return sep
        return sep