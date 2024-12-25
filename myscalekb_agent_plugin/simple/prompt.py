from langchain_core.messages import SystemMessage

from myscalekb_agent_base.prompt import Prompt


class SimpleRagPrompt(Prompt):

    def understand_prompt_template(self):
        system_prompt = """You are an intelligent assistant integrated with a Retrieval-Augmented Generation (RAG) system and various specialized tools. 
Your primary task is to answer user queries by retrieving relevant information exclusively from a predefined set of documents available to you. 
Utilize the tools at your disposal to enhance the accuracy and relevance of your responses.
"""
        messages = [SystemMessage(content=system_prompt)]
        return self.prompt_template(messages, with_history=True, with_user_query=True)

    def summarize_prompt_template(self, contexts: str):
        system_prompt = """You are an intelligent assistant integrated with a Retrieval-Augmented Generation (RAG) system. 
Your primary task is to answer user queries by retrieved relevant contexts available to you.

Below are the contexts available to you:
{contexts}

Exclude any information that is not directly relevant to the question, and avoid redundancy. 
Clearly state "information is missing on [topic]" if the provided contexts do not contain enough information to fully answer the question. 
And recommend more suitable questions to users based on the questions and context.
"""

        messages = [SystemMessage(content=system_prompt.format(contexts=contexts))]
        return self.prompt_template(messages, with_history=True, with_user_query=True)
