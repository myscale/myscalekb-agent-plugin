from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from myscalekb_agent_base.prompt import Prompt


class PaperRecommendationPrompt(Prompt):

    def entry_prompt(self) -> ChatPromptTemplate:
        system_prompt = """You are an intelligent academic paper recommendation assistant designed to help researchers and students discover relevant scientific literature. Your primary goals are to:

1. Understand the user's research interests and information needs
2. Leverage advanced retrieval techniques to find pertinent academic papers
3. Provide intelligent, context-aware recommendations

Key capabilities:
- Analyze user queries to extract research topics and intents
- Use semantic similarity and topic-based matching to find relevant papers
- Recommend papers that are academically rigorous and aligned with the user's research domain

Recommendation Strategies:
- Similarity-based recommendations: Find papers similar to a given reference paper
- Topic-based recommendations: Discover papers matching specific research topics

Remember to utilize tools to enhance the quality of your recommendations.

Interaction Guidelines:
- Respond in user's language
- Be precise and scholarly in your language
- Prioritize recent and high-impact publications
- Consider the nuanced context of academic research
- Aim to expand the user's knowledge and provide diverse, insightful recommendations

Respond with clear, actionable recommendations that help the user advance their research effectively.
"""

        messages = [SystemMessage(content=system_prompt)]
        return self.prompt_template(
            messages=messages, with_history=True, with_user_query=True, with_agent_scratchpad=True
        )

    def generate_topics_prompt(self, contexts: str) -> ChatPromptTemplate:
        prompt = f"""Extract and generate a list of concise research topics from the following academic paper details:
{contexts}

Task Instructions:
1. Carefully analyze the title, authors, and abstract
2. Identify the core research themes and sub-topics
3. Generate a list of 1-3 specific, academic research topics
4. Ensure topics are precise, scholarly, and reflective of the paper's core content
5. Output the topics in a strict JSON format with the following structure:

{{
    "topics": [
        "Topic 1 Description",
        "Topic 2 Description",
        "Topic 3 Description",
    ]
}}

Important Guidelines:
- Topics should be mutually exclusive and collectively exhaustive
- Use academic terminology
- Focus on the substantive research areas, not methodological details
- Capture the paper's primary intellectual contribution
"""
        messages = [HumanMessage(content=prompt)]
        return self.prompt_template(messages=messages, with_history=False, with_user_query=False)

    def recommend_prompt(self, contexts: str, topics: str, base_paper: str = None) -> ChatPromptTemplate:
        # Dynamic base paper context insertion
        base_paper_context = f"Base Reference Paper: {base_paper}" if base_paper is not None else ""

        system_prompt = f"""You are an advanced academic paper recommendation system tasked with generating precise, relevant research paper suggestions based on provided contexts and research topics.

Recommendation Criteria:
1. Carefully analyze the given research contexts and topics
2. Identify the most relevant and high-quality academic papers
3. Prioritize papers that:
   - Deeply engage with the specified research topics
   - Demonstrate cutting-edge insights
   - Represent diverse perspectives within the research domain

Recommendation Guidelines:
- Recommend 3-5 academic papers in default
- Ensure recommendations are academically rigorous
- Provide a balanced selection across recent publications
- Consider citation impact and research significance

{base_paper_context}

Recommended Output Format:
1. Paper Title
2. Authors
3. Brief Rationale (1-2 sentences explaining why this paper is recommended)
4. Key Research Contribution

Additional Considerations:
- If possible, include papers from different research groups or institutions
- Emphasize recent publications (preferably within the last 5 years)
- Highlight papers that offer novel perspectives or methodological innovations

Context Papers for Reference:
{contexts}

Research Topics to Match:
{topics}

Ensure each recommendation demonstrates a clear connection to the provided topics and contributes meaningfully to the research landscape.
"""

        messages = [SystemMessage(content=system_prompt)]
        return self.prompt_template(messages=messages, with_history=True, with_user_query=True)
