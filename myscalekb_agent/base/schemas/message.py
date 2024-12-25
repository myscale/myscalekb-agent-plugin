import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Annotated

from pydantic import Field, BaseModel

from myscalekb_agent.base.schemas.agent_metadata import AgentMetadata


class MessageRoles(str, Enum):
    USER = "user"
    AGENT = "agent"
    MODEL = "model"
    RETRIEVER = "retriever"


def generate_uuid() -> str:
    return str(uuid.uuid4())


def generate_now() -> datetime:
    return datetime.now()


class Message(BaseModel):
    id: str = Field(
        description="Message ID. For chunks of model stream, the same call will share one message id.",
        default_factory=generate_uuid,
    )
    role: MessageRoles = Field(description="The role of the message creator.")
    create_time: datetime = Field(description="Message create timestamp.", default_factory=generate_now)


class SubmitFormData(BaseModel):
    """This data structure is used to submit jsonforms returned by the model."""

    source_agent: AgentMetadata = Field(
        description="Indicates which agent the form data was derived from, usually from the `agent_metadata` of the previous response."
    )
    data: Any = Field(description="The actual form data.")


class UserMessage(Message):
    content: Annotated[str, Field(title="String", description="String message.")] | SubmitFormData = Field(
        description="Message content. It is usually a normal string or the data structured with jsonforms returned by the model."
    )
    referred_agent: str | None = Field(
        default=None,
        description="The referable agent ID from `list_referable_agents` endpoint.",
        examples=["ppt_agent"],
    )
