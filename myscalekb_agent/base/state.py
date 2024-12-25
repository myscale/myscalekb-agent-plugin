import operator
from typing import TypedDict, Union, Annotated, List

from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage

from myscalekb_agent.base.schemas.agent_metadata import AgentMetadata
from myscalekb_agent.base.schemas.message import UserMessage


class AgentState(TypedDict):
    """Common Agent State."""

    # The input UserMessage
    input: UserMessage
    # The translated input message, str content
    query: str
    # The list of previous messages in the conversation
    chat_history: List[BaseMessage]
    # The current running agent metadata
    agent_metadata: AgentMetadata
    # The outcome of a given call to the agent
    # Needs `None` as a valid type, since this is what this will start as
    agent_outcome: Union[ToolAgentAction, AgentAction, AgentFinish, None]
    # List of actions and corresponding observations
    # Here we annotate this with `operator.add` to indicate that operations to
    # this state should be ADDED to the existing values (not overwrite it)
    intermediate_steps: Annotated[list[tuple[ToolAgentAction, AgentAction, str]], operator.add]
    # A complete Agent Query shares one trace id
    trace_id: Union[str, None]
