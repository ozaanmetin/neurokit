"""
neurokit.core.utils.id
"""


import uuid
from uuid_extensions import uuid7


class IDHelper:
    @staticmethod
    def generate_uuid() -> uuid.UUID:
        """
        Generate a unique identifier using UUID7.
        Which is time-ordered and more efficient for databases.

        :return: generated UUID7
        :rtype: uuid.UUID
        """
        return uuid7()
    
    @staticmethod
    def generate_document_id(source = None) -> str:
        """
        Generate a unique document ID.
        
        :param source: optional source in order to create deterministic ids
        :return: generated document id
        :rtype: str
        """

        if not source:
            return str(IDHelper.generate_uuid())

        namespace = uuid.NAMESPACE_DNS
        return str(uuid.uuid5(namespace, source))
        
    @staticmethod
    def generate_chunk_id(document_id: str, chunk_index: int) -> str:
        """
        Generate a unique chunk ID based on document ID and chunk index.

        :param document_id: the ID of the document the chunk belongs to.
        :param chunk_index: the index of the chunk within the document.
        :return: generated chunk id
        :rtype: str
        """
        namespace = uuid.NAMESPACE_DNS
        name = f"{document_id}-{chunk_index}"
        return str(uuid.uuid5(namespace, name))
        
