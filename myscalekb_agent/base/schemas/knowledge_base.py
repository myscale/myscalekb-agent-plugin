from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class KnowledgeBaseType(str, Enum):
    GENERAL = "general"
    PAPER = "paper"
    PATENT = "patent"
    BOOK = "book"


class KnowledgeScope(BaseModel):
    """Represents a specific collection or a selection of documents within a knowledge base."""

    kb_type: KnowledgeBaseType = Field(
        description="The type of the collection this knowledge belongs to.",
        examples=[KnowledgeBaseType.PAPER],
    )
    kb_id: str = Field(description="Unique identifier for the knowledge base.", examples=["s2"])
    doc_ids: List[int] = Field(
        description="List of document IDs. An empty list indicates that all documents within the knowledge base are included.",
        default=[],
        examples=[[2988078]],
    )


class KnowledgeChunkScope(BaseModel):
    """Represents a specific collection or a selection of documents within a knowledge base."""

    kb_type: KnowledgeBaseType = Field(
        description="The type of the collection this knowledge belongs to.",
        examples=[KnowledgeBaseType.PAPER],
    )
    kb_id: str = Field(description="Unique identifier for the knowledge base.", examples=["s2"])
    doc_id: int = Field(description="Globally unique identifier.", examples=[2988078])
    chunk_ids: List[int] = Field(description="List of chunk IDs.", default=[])
