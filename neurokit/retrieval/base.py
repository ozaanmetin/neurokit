from abc import ABC, abstractmethod

from neurokit.retrieval.entity import RetrievalResult, RetrievalContext


class Retriever(ABC):
    """
    Abstract base class for all retrievers.
    """

    @abstractmethod
    def retrieve(self, context: RetrievalContext) -> list[RetrievalResult]:
        """
        Retrieve relevant documents based on the given context.

        Args:
            context (RetrievalContext): The context for the retrieval operation.

        Returns:
            list[RetrievalResult]: A list of retrieval results.
        """
        pass

    def get_retirever_info(self) -> dict:
        """
        Get information about the retriever.

        Returns:
            dict: A dictionary containing retriever information.
        """
        return {
            "type": self.__class__.__name__,
        }