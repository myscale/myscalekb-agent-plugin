import asyncio
from typing import List, Optional, Callable, Any

from myscalekb_agent.base.retriever import Retriever
from myscalekb_agent.queries.base import MyScaleQuery


class TitleRetriever(Retriever):

    async def retrieve(
        self,
        queries: List[str],
        format_output: Optional[Callable[[List], Any]] = None,
        *args,
        **kwargs,
    ):
        """Use full_text_search to search for the corresponding document/(paper) based on the title"""
        title = queries[0]
        text_escaping = MyScaleQuery.text_escape(title)
        where_str = MyScaleQuery.gen_where_str(self.knowledge_scopes)

        # Many retrieval methods and Data Schema are predefined in MyScaleQuery and its related classes.
        # Here, we use relatively straightforward SQL to express the ability to use text search to retrieve related papers based on Title.
        q_str = f"SELECT title, abstract, authors, bm25_score AS score FROM full_text_search('{MyScaleQuery.database}.documents', 'multi_idx', '{text_escaping}', with_score=1) {where_str} LIMIT 1"
        res = MyScaleQuery.query(self.myscale_client, q_str)

        if format_output:
            return format_output(res.named_results())

        return list(res.named_results())


class TopicRetriever(Retriever):

    async def retrieve(
        self,
        queries: List[str],
        format_output: Optional[Callable[[List], Any]] = None,
        *args,
        **kwargs,
    ):
        """Use vector_search to search for the corresponding document/(paper) based on the topics, with concurrent execution"""
        topics = queries

        # Prepare the vector search query
        where_str = MyScaleQuery.gen_where_str(self.knowledge_scopes)

        # Define an async helper function for individual topic search
        async def search_topic(topic: str):
            # Escape the text to prevent SQL injection
            text_escaping = MyScaleQuery.text_escape(topic)
            text_vec = self.embedding_model.embed_query(text_escaping)

            # Construct the vector search query
            q_str = f"""SELECT title, abstract, authors, distance(vector, {text_vec}) AS d FROM {MyScaleQuery.database}.documents {where_str} ORDER BY d ASC LIMIT 10"""

            res = MyScaleQuery.query(self.myscale_client, q_str)
            return list(res.named_results())

        # Use asyncio.gather to run searches concurrently
        results = await asyncio.gather(*[search_topic(topic) for topic in topics])

        # Flatten the results
        all_results = [item for sublist in results for item in sublist]

        # Apply optional output formatting
        if format_output:
            return format_output(all_results)

        return all_results
