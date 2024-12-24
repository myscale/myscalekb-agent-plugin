from abc import abstractmethod, ABC
from typing import Type

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from myscalekb_agent.base.retriever import Retriever


class BaseTool(ABC):
    """Base class for all tools"""

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Subclasses must implement this method to define the tool name."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def description(cls) -> str:
        """Subclasses must implement this method to define the tool description."""
        raise NotImplementedError

    @abstractmethod
    def params(self) -> Type[BaseModel]:
        """Subclasses must implement this method to define the tool parameters."""
        raise NotImplementedError

    @abstractmethod
    def execute(self, *args, **kwargs):
        """Subclasses must implement this method to define the tool execution method."""
        raise NotImplementedError

    @property
    def tool(self) -> StructuredTool:
        return StructuredTool(
            name=self.name(),
            description=self.description(),
            args_schema=self.params(),
            func=self.execute,
        )
