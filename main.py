import os
from typing import TypedDict
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults

# ==========================
# Load Environment Variables
# ==========================
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

# ==========================
# Define State Schema
# ==========================
class ResearchState(TypedDict):
    query: str
    raw_results: str
    sources: list[str]
    research_notes: str
    drafted_answer: str
    final_answer: str
    review: str

# ==========================
# Set up LLM
# ==========================
llm = ChatOpenAI(temperature=0.3, model="gpt-4", openai_api_key=openai_key)

# ==========================
# Tavily Tool
# ==========================
tavily_tool = TavilySearchResults(api_key=tavily_key)

# ==========================
# Prompt Templates
# ==========================
summary_prompt_template = PromptTemplate.from_template(
    """
    You are a research summarizer AI. Your task is to take the following raw web data and extract key insights, facts, and important points related to: "{query}"

    Please structure the notes clearly using bullet points or short paragraphs.

    Raw Data:
    {raw_results}
    """
)

analysis_prompt_template = PromptTemplate.from_template(
    """
    You are an analysis agent. Based on the following research notes, synthesize a comprehensive and insightful answer.

    1. Start with a brief introduction to the topic: "{query}". Explain what it is and why it’s important.
    2. Then, identify and explain the key trends or insights related to it.
    3. For each point, provide at least one real-world example or case study (e.g., companies, use-cases, or recent innovations).
    4. Conclude with the implications for professionals, industries, or society, and mention key skills or tools relevant to the topic.

    Research Notes: {research_notes}
    """
)

critic_prompt_template = PromptTemplate.from_template(
    """
    You are a review agent. Critically evaluate the drafted answer for the topic: "{query}"

    Drafted Answer: {drafted_answer}

    Check if it:
    - Starts with a clear, informative introduction
    - Covers important points thoroughly
    - Includes real-world examples
    - Ends with a meaningful conclusion

    Suggest any improvements or confirm its quality.
    """
)

# ==========================
# Convert to Runnable Chains
# ==========================
summary_chain = summary_prompt_template | llm
analysis_chain = analysis_prompt_template | llm
critic_chain = critic_prompt_template | llm

# ==========================
# Define Agent Nodes
# ==========================
def researcher_node(state: ResearchState) -> ResearchState:
    search_results = tavily_tool.run(state["query"])
    raw = "\n".join([result["content"] for result in search_results])
    sources = [result["url"] for result in search_results if "url" in result]

    return {**state, "raw_results": raw, "sources": sources}

def summarizer_node(state: ResearchState) -> ResearchState:
    result = summary_chain.invoke({
        "raw_results": state["raw_results"],
        "query": state["query"]
    })
    return {**state, "research_notes": result.content}

def drafter_node(state: ResearchState) -> ResearchState:
    result = analysis_chain.invoke({
        "research_notes": state["research_notes"],
        "query": state["query"]
    })
    return {**state, "drafted_answer": result.content}

def critic_node(state: ResearchState) -> ResearchState:
    result = critic_chain.invoke({
        "drafted_answer": state["drafted_answer"],
        "query": state["query"]
    })

    sources_text = "\n\nSources:\n" + "\n".join(state["sources"])
    final = state["drafted_answer"] + sources_text

    return {
        **state,
        "final_answer": final,
        "review": result.content
    }

# ==========================
# Build LangGraph
# ==========================
builder = StateGraph(ResearchState)
builder.add_node("researcher", RunnableLambda(researcher_node))
builder.add_node("summarizer", RunnableLambda(summarizer_node))
builder.add_node("drafter", RunnableLambda(drafter_node))
builder.add_node("critic", RunnableLambda(critic_node))

builder.set_entry_point("researcher")
builder.add_edge("researcher", "summarizer")
builder.add_edge("summarizer", "drafter")
builder.add_edge("drafter", "critic")
builder.add_edge("critic", END)

graph = builder.compile()

# ==========================
# Run the Graph
# ==========================
if __name__ == "__main__":
    user_query = input("Enter your research topic or question: ")
    output = graph.invoke({"query": user_query})

    print("\n===== FINAL ANSWER =====")
    print(output["final_answer"])
    print("\n===== REVIEW =====")
    print(output["review"])
