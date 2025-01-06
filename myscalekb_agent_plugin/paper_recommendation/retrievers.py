import asyncio
from operator import attrgetter
from typing import List, Optional, Callable, Any, Dict

from myscalekb_agent_base.myscale_query import MyScaleQuery
from myscalekb_agent_base.retriever import Retriever


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
        res = await MyScaleQuery.aquery(self.myscale_client, q_str)

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
            text_vec = await self.embedding_model.aembed_query(text_escaping)

            # Construct the vector search query
            q_str = f"""SELECT doc_id, title, abstract, authors, distance(vector, {text_vec}) AS d FROM {MyScaleQuery.database}.documents {where_str} ORDER BY d ASC LIMIT 10"""

            res = await MyScaleQuery.aquery(self.myscale_client, q_str)
            return list(res.named_results())

        # Use asyncio.gather to run searches concurrently
        results = await asyncio.gather(*[search_topic(topic) for topic in topics])

        # Flatten the results
        all_results = [item for sublist in results for item in sublist]

        # Create a dictionary to store the best result for each doc_id
        deduplicated: Dict[str, dict] = {}

        for result in all_results:
            # If we haven't seen this doc_id before, or if this result has a better (lower) distance
            if result["doc_id"] not in deduplicated or result["d"] < deduplicated[result["doc_id"]]["d"]:
                deduplicated[result["doc_id"]] = result

        # Convert back to list and sort by distance
        final_results = sorted(deduplicated.values(), key=lambda x: x["d"])

        # Apply optional output formatting
        if format_output:
            return format_output(final_results)

        return final_results
