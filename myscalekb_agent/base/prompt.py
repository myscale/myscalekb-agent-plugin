from abc import ABC
from typing import List

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from myscale_llmhub.dsl.memory import MyScaleChatMemory


class Prompt(ABC):
    """Base class for prompts."""

    def __init__(self, memory: MyScaleChatMemory):
        self.memory = memory

    def prompt_template(
        self,
        messages: List[BaseMessage],
        with_history: bool = False,
        with_user_query: bool = False,
        with_agent_scratchpad: bool = False,
    ) -> ChatPromptTemplate:
        """
        Construct ChatPromptTemplate from list of messages.
        :param messages: The list of messages controlled by this prompt.
        :param with_history: If append history messages to prompt.
        :param with_user_query: If append user query message to prompt.
        :param with_agent_scratchpad: If append agent scratchpad to prompt to track llm reasoning.
        :return:
        """
        if with_history:
            messages.append(MessagesPlaceholder(variable_name="chat_history"))
        if with_user_query:
            messages.extend([HumanMessagePromptTemplate.from_template("{query}")])
        if with_agent_scratchpad:
            messages.extend([MessagesPlaceholder(variable_name="agent_scratchpad")])

        return ChatPromptTemplate.from_messages(messages)
