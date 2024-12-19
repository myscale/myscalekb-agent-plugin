from typing import List, Optional, Callable, Any

from myscalekb_agent.base.retriever import Retriever


class SimpleRagRetriever(Retriever):

    async def retrieve(
        self,
        queries: List[str],
        format_output: Optional[Callable[[List], Any]] = None,
        *args,
        **kwargs,
    ):
        """retrieve workflow:
        1. embed query
        2. use distance(VectorSearch) to query MyScale
        """
