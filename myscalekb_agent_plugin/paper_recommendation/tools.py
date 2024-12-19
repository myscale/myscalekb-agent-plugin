from typing import Type

from pydantic import BaseModel, Field

from myscale_agent.base.tool import BaseTool


class SimilarityRecommendation(BaseTool):

    @classmethod
    def name(cls) -> str:
        return "similarity_recommendation"

    @classmethod
    def description(cls) -> str:
        return "Recommends papers to users based on their similarity to a specified paper."

    def params(self) -> Type[BaseModel]:
        class ToolParams(BaseModel):
            title: str = Field(description="The title of the paper for which similar papers are to be recommended.")

        return ToolParams

    def execute(self, *args, **kwargs):
        pass


class TopicBasedRecommendation(BaseTool):

    @classmethod
    def name(cls) -> str:
        return "topic_based_recommendation"

    @classmethod
    def description(cls) -> str:
        return "Recommends papers to users based on specified topics or categories."

    def params(self) -> Type[BaseModel]:
        class ToolParams(BaseModel):
            topics: list[str] = Field(
                description="A list of topics or categories used to filter and recommend relevant papers."
            )

        return ToolParams

    def execute(self, *args, **kwargs):
        pass
