from myscalekb_agent_base.graph_builder import node, edge
from myscalekb_agent_base.sub_agent import SubAgent


class SimpleRag(SubAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @node
    @edge(target_node="retrieve")
    async def understand(self, data):
        """run llm to get queries from user input"""

    @node
    @edge(target_node="summary")
    async def retrieve(self, data):
        """use retriever to search contexts by queries"""

    @node
    @edge(target_node="__end__")
    async def summarize(self, data):
        """summarize contexts and answer user input"""
