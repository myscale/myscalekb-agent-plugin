import logging
from abc import ABC, abstractmethod
from typing import TypedDict, List, Type, Any

from clickhouse_connect.driver import Client
from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain_core.agents import AgentFinish
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.tools import StructuredTool
from langgraph.graph.graph import CompiledGraph
from pydantic import BaseModel

from myscalekb_agent.base.graph_builder import GraphBuilder
from myscalekb_agent.base.schemas.knowledge_base import KnowledgeScope
from myscalekb_agent.base.state import AgentState
from myscalekb_agent.base.utils import AutoDiscover

logger = logging.getLogger(__name__)


class SubAgent(ABC, GraphBuilder):
    """An abstract base class for creating specialized sub-agents that can build and process graphs.
    Inherits from ABC (Abstract Base Class) and GraphBuilder to provide graph manipulation capabilities.

    Attributes:
        llm (BaseChatModel): The underlying chat model used for natural language processing tasks.
            Extracted from the LLM management class for direct interaction.

        embedding_model (Embeddings): Model for generating vector embeddings from text.
            Used for semantic search and similarity comparisons.

        myscale_client (Client): Client interface for MyScale database operations.
            Handles vector storage, retrieval, and database operations.

        memory (BaseChatMessageHistory): System for managing conversation history.
            Maintains context and enables coherent multi-turn interactions.

        knowledge_scopes (List[KnowledgeScope]): List of knowledge domains available to the agent.
            Defines the boundaries and areas of expertise the agent can operate within.
    """

    def __init__(self, ctx, llm, memory, *args, **kwargs):
        """Initialize a new SubAgent instance. Some types are erased for import safety reasons.

        Args:
            ctx (Context): Context object that provides environment and configuration settings.
            llm (LLM): Customized LLM Management Class.
            memory (MyScaleChatMemory): Memory system for storing and retrieving conversation history
            *args: Variable length argument list for future extensibility
            **kwargs: Arbitrary keyword arguments for future extensibility
        """
        logger.info("Initializing SubAgent - %s", self.__class__.__name__)

        self.llm: BaseChatModel = llm.model
        self.embedding_model: Embeddings = ctx.embedding_model
        self.myscale_client: Client = ctx.myscale_client
        self.memory: BaseChatMessageHistory = memory
        self.knowledge_scopes: List[KnowledgeScope] = ctx.variables.get("knowledge_scopes")

    @classmethod
    def register(cls, master_builder, ctx, llm, memory, *args, **kwargs):
        """Factory method for creating a SubAgent instance to ensure graph builder works"""
        agent = cls(ctx=ctx, llm=llm, memory=memory, *args, **kwargs)
        agent._finalize_initialization(master_builder)
        return agent

    def _finalize_initialization(self, master_builder):
        forward_func = StructuredTool(
            name=self.name(),
            description=self.description(),
            args_schema=self.forward_schema(),
            func=self.__placeholder_func,
        )

        graph = self.graph()
        master_builder.register_sub_agent(forward_func, (self.name(), graph))

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Subclasses must implement this method to register with the MasterAgent."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def description(cls) -> str:
        """Subclasses must implement this method to register with the MasterAgent."""
        raise NotImplementedError

    @classmethod
    def state_definition(cls) -> Type[TypedDict]:
        """Subclasses must implement this method to define Graph State Data.
        Of course, developer can use the default State - `AgentState`.
        The purpose of showing the definition is to let developers understand the importance of the data structure in AgentState.
        Understanding the meaning of the AgentState field is the starting point for Edit Graph.
        """
        return AgentState

    @classmethod
    def forward_schema(cls) -> Type[BaseModel]:
        """Subclasses can inherit this method to implement the arguments that Forward can contain.
        This structure can be obtained from state["forward_args"]

        Default is empty.
        """
        return cls.EmptySchema

    def graph(self) -> CompiledGraph:
        """Subclass can use this method to customize the implementation of Graph instead of using annotation to define it."""
        return self._build_graph(self.state_definition())

    class EmptySchema(BaseModel):
        """An empty schema for tools without arguments."""

        pass

    def __placeholder_func(self, **kwargs):
        raise NotImplementedError("This tool is a placeholder and cannot be called.")

    @staticmethod
    def _get_tool_args(agent_outcome) -> dict:
        """Utils method: Get tool calling arguments from the previous agent_outcome"""
        if isinstance(agent_outcome, ToolAgentAction):
            return agent_outcome.tool_input
        if isinstance(agent_outcome, list) and isinstance(agent_outcome[0], ToolAgentAction):
            return agent_outcome[0].tool_input

        return {}

    @staticmethod
    def _make_agent_finish(output: Any) -> AgentFinish:
        return AgentFinish(return_values={"output": output}, log="")


class SubAgentRegistry:
    _agents = {}

    @classmethod
    def register(cls, agent_class):
        cls._agents[agent_class.__name__] = agent_class

    @classmethod
    def get_enabled_agents(cls, enabled_agent_names: List[str]) -> List[Type[SubAgent]]:
        """Get enabled agents by class name"""
        enabled_agents = []

        for name, agent_class in cls._agents.items():
            if name in enabled_agent_names:
                enabled_agents.append(agent_class)

        return enabled_agents

    @classmethod
    def auto_discover_agents(cls, base_packages: List[str], max_depth: int = 3):
        """
        Recursively scan packages and subpackages for SubAgent classes

        Args:
            base_packages (List[str]): List of base package names to start discovery
            max_depth (int): Maximum recursion depth to prevent infinite loops, default 3
        """
        AutoDiscover.discover_subclasses(SubAgent, cls.register, base_packages, max_depth)
