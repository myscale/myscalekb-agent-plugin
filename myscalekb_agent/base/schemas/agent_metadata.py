from pydantic import BaseModel, Field


class AgentMetadata(BaseModel):
    name: str = Field(description="Agent name. The creator.")
    step: str = Field(description="The current step of the workflow.")
