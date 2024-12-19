from typing import TypedDict, List

from langchain.agents import create_openai_tools_agent
from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain_core.runnables import RunnableConfig

from myscale_agent.agents.control_tags import ControlTags
from myscale_agent.agents.slide_deck_ai.helpers.json_helper import load_and_fix_json
from myscale_agent.agents.state import AgentState
from myscale_agent.base.graph_builder import node, conditional_edge, entry, GraphBuilder, edge
from myscale_agent.base.sub_agent import SubAgent
from myscale_agent_plugin.paper_recommendation.prompt import PaperRecommendationPrompt
from myscale_agent_plugin.paper_recommendation.retrievers import TitleRetriever, TopicRetriever
from myscale_agent_plugin.paper_recommendation.tools import SimilarityRecommendation, TopicBasedRecommendation


class PaperRecommendationAgent(SubAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title_retriever = TitleRetriever(
            embedding_model=self.embedding_model,
            myscale_client=self.myscale_client,
            knowledge_scopes=self.knowledge_scopes,
        )
        self.topic_retriever = TopicRetriever(
            embedding_model=self.embedding_model,
            myscale_client=self.myscale_client,
            knowledge_scopes=self.knowledge_scopes,
        )
        self.similarity_recommendation = SimilarityRecommendation(self.title_retriever)
        self.topic_based_recommendation = TopicBasedRecommendation(self.topic_retriever)
        self.prompt = PaperRecommendationPrompt(memory=self.memory)

    @classmethod
    def name(cls) -> str:
        return "paper_recommendation_agent"

    @classmethod
    def description(cls) -> str:
        return "Recommends papers to users."

    @classmethod
    def state_definition(cls) -> TypedDict:
        class PaperRecommendationState(AgentState):
            # Record the topics generated in process
            topics: List[str]
            # Similarity recommendations used as the basis for the paper
            base_paper: str

        return PaperRecommendationState

    @node
    @entry
    @conditional_edge(
        path=lambda data: (
            data["agent_outcome"][0].tool
            if isinstance(data["agent_outcome"], list) and isinstance(data["agent_outcome"][0], ToolAgentAction)
            else (data["agent_outcome"].tool if isinstance(data["agent_outcome"], ToolAgentAction) else "end")
        ),
        path_map={
            "end": GraphBuilder.END,
            "similarity_recommendation": "generate_topics_from_similar_paper",
            "topic_based_recommendation": "extract_topics",
        },
    )
    async def entry(self, data):
        """Entry point for the paper recommendation workflow.

        Initializes an OpenAI tools agent with predefined tools for similarity and topic-based recommendations.
        Invokes the agent with the given input data and returns the agent's outcome.
        """
        runnable = create_openai_tools_agent(
            llm=self.llm,
            tools=[self.similarity_recommendation.tool, self.topic_based_recommendation.tool],
            prompt=self.prompt.entry_prompt(),
        )

        agent_outcome = await runnable.ainvoke(data)
        return {"agent_outcome": agent_outcome}

    @node
    @edge(target_node="run_topic_recommendation")
    async def generate_topics_from_similar_paper(self, data):
        """Generates topics based on a similar paper's title and abstract.

        Workflow:
        1. Retrieve the base paper's title from the tool arguments
        2. Search and retrieve context for the paper using its title
        3. Generate topics using the retrieved context and the LLM

        Returns:
            dict: A dictionary with generated topics and the base paper's title
        """
        tool_args = self._get_tool_args(data["agent_outcome"])

        base_paper = tool_args["title"]
        contexts = await self.title_retriever.retrieve(queries=[base_paper], format_output=self._format_chunk)
        runnable = self.prompt.generate_topics_prompt(contexts) | self.llm
        response = await runnable.ainvoke(data, config=RunnableConfig(tags=[ControlTags.EXCLUDE_STREAM]))
        json_result = load_and_fix_json(response.content)
        return {"topics": json_result["topics"], "base_paper": base_paper}

    @node
    @edge(target_node="run_topic_recommendation")
    async def extract_topics(self, data):
        """Extracts topics directly from the tool call arguments.

        This method is used when topics are already provided in the tool arguments,
        bypassing the need to generate topics from a similar paper.

        Returns:
            dict: A dictionary with the extracted topics
        """
        tool_args = self._get_tool_args(data["agent_outcome"])
        return {"topics": tool_args["topics"]}

    @node
    @edge(target_node=GraphBuilder.END)
    async def run_topic_recommendation(self, data):
        """Recommends papers based on the extracted or generated topics.

        Workflow:
        1. Retrieve context for the given topics
        2. Generate paper recommendations using the retrieved context, topics, and optional base paper

        Returns:
            At the final node, there is no need to write state anymore.
        """
        topics = data["topics"]
        base_paper = data.get("base_paper")

        contexts = await self.topic_retriever.retrieve(queries=topics, format_output=self._format_chunk)
        runnable = self.prompt.recommend_prompt(contexts, topics, base_paper) | self.llm
        await runnable.ainvoke(data)
        return {}

    @staticmethod
    def _format_chunk(rows) -> str:
        contexts = []
        for row in rows:
            contexts.append(f"Title:{row.get('title')}")
            contexts.append(f"Authors:{row.get('authors')}")
            contexts.append(f"Abstract:{row.get('abstract')}")
            contexts.append("\n")
        return "\n".join(contexts)
