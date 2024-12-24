from abc import ABC, abstractmethod
from typing import List, Optional, Callable, Any

from clickhouse_connect.driver import Client
from myscale_llmhub.dsl import EmbeddingModel

from myscalekb_agent.schemas.knowledge.knowledge_scope import KnowledgeScope


class Retriever(ABC):
    """Retriever is flexible in use. You can implement a custom Retriever to complete any data query.
    You can use the embedding_model / myscale_client and other components managed by `SubAgent` to conveniently instantiate a Retriever.
    """

    def __init__(
        self, embedding_model: EmbeddingModel, myscale_client: Client, knowledge_scopes: List[KnowledgeScope] = None
    ):
        self.embedding_model = embedding_model
        self.myscale_client = myscale_client
        self.knowledge_scopes = knowledge_scopes

    @abstractmethod
    async def retrieve(
        self,
        queries: List[str],
        format_output: Optional[Callable[[List], Any]] = None,
        *args,
        **kwargs,
    ):
        """
        Asynchronously retrieve information or data for the given queries.

        This abstract method defines the interface for a retrieval operation,
        which should be implemented by subclasses to fetch data based on input queries.

        :param queries: A list of string queries to be used for retrieval.
        :type queries: List[str]

        :param format_output: An optional callable that transforms the retrieved
                               data into a desired output format. If not provided,
                               the raw retrieved data is returned.
        :type format_output: Optional[Callable[[List], Any]], optional

        :param args: Additional positional arguments that may be needed
                     for specific retrieval implementations.

        :param kwargs: Additional keyword arguments that may be needed
                       for specific retrieval implementations.

        :return: Retrieved data, potentially formatted by the format_output function.
        :rtype: Any

        :raises NotImplementedError: This is an abstract method that must be
                                     implemented by subclasses.
        """
        raise NotImplementedError
