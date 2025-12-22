from typing import Optional

# Retrieval imports
from neurokit.retrieval.base import Retriever
from neurokit.retrieval.entity import RetrievalResult, RetrievalContext

# Knowledge imports
from neurokit.knowledge.vector_store import VectorStore
from neurokit.knowledge.embedding import EmbeddingProvider


class VectorRetriever(Retriever):
    """
    A retriever that uses a vector store to retrieve relevant documents based on embeddings.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        # TODO: make collection_name default value compatible with VectorStore DEFAULT_COLLECTION and consider implementing through CollectionConfig
        collection_name: str = "default",
        score_threshold: Optional[float] = None,
    ):
        
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.collection_name = collection_name
        self.score_threshold = score_threshold

    def retrieve(
            self, 
            context: RetrievalContext, 
            include_vectors: bool = False, 
            include_payloads: bool = True
        ) -> list[RetrievalResult]:
        
        # embed the query
        query_embedding = self.embedding_provider.embed_text(context.query)

        # query the vector store
        results =  self.vector_store.query(
            vector=query_embedding,
            top_k=context.top_k,
            filter=context.filters,
            collection=self.collection_name,
            include_vectors=include_vectors,
            include_payloads=include_payloads,
        )

        # convert to RetrievalResult
        retrieval_results = []

        for r in results:
            # apply score threshold filtering
            if self.score_threshold is not None and r.score < self.score_threshold:
                continue
            
            # convert VectorSearchResult to RetrievalResult
            retrieval_results.append(
                RetrievalResult(
                id=r.id,
                content=r.content,  # Use content field directly from VectorSearchResult
                score=r.score,
                metadata=r.metadata,
            ))

        return retrieval_results
    
