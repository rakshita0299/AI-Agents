# Research Assistant Using LangChain and Tavily

## Overview

This project implements an automated research assistant using LangChain, OpenAI's GPT model, and Tavily's search API to perform structured research, summarize findings, analyze results, and critique drafted answers. The flow of operations is modeled using a directed acyclic graph (DAG) where each node represents a specific task or step in the research process. The agent takes in a user-defined research query, processes it through a series of stages, and outputs a final answer along with a critical review.

## Key Components

1. **LangChain**: A library used to build language model-powered applications, enabling integration of AI models into workflows.
2. **Tavily**: A search tool that allows access to relevant online content and integrates it into the research process.
3. **OpenAI's GPT-4**: The language model that performs tasks such as summarization, analysis, and critique.
4. **LangGraph**: A framework to represent tasks as nodes, linked by edges, allowing us to define a clear process flow.

## Project Structure

The program follows a sequence of operations, each of which can be thought of as a "node" in a process chain:

1. **Researcher Node**: Queries Tavily to fetch relevant online data.
2. **Summarizer Node**: Uses a GPT model to summarize the raw results from the research.
3. **Drafter Node**: Synthesizes the summarized notes into a detailed draft answer.
4. **Critic Node**: Critiques the drafted answer for quality and completeness, and appends references.

### Workflow Breakdown

1. **Loading Environment Variables**:  
   The environment variables, such as OpenAI and Tavily API keys, are loaded from a `.env` file using the `dotenv` library to keep them secure and out of the source code.

2. **State Schema (TypedDict)**:  
   The `ResearchState` dictionary defines the data structure used to track various pieces of information throughout the research process. It contains:
   - `query`: The user's input query for research.
   - `raw_results`: Raw data from Tavily search results.
   - `sources`: URLs of the sources.
   - `research_notes`: Summarized research notes.
   - `drafted_answer`: The initial draft answer based on the notes.
   - `final_answer`: The final, polished answer after critique.
   - `review`: A critical review of the drafted answer.

3. **LLM Setup**:  
   A language model (LLM) instance is initialized using OpenAI’s GPT-4 API, with a low temperature (0.3) to ensure the model’s responses are more factual and less creative.

4. **Tavily Tool**:  
   The Tavily tool is initialized using the Tavily API key. This tool fetches relevant web content based on the research query.

5. **Prompt Templates**:  
   Three different prompt templates are created using LangChain:
   - **Summary Prompt**: To summarize raw research results into concise notes.
   - **Analysis Prompt**: To synthesize the summarized notes into a comprehensive answer with insights and examples.
   - **Critic Prompt**: To evaluate the drafted answer and provide feedback for improvements.

6. **Runnable Chains**:  
   Each of the prompt templates is connected to the GPT model, forming a chain of operations that transforms the input data into the desired output.

7. **Agent Nodes**:  
   The core of the program consists of four functions:
   - `researcher_node`: Queries Tavily for research data based on the user's query.
   - `summarizer_node`: Summarizes the raw search results into research notes.
   - `drafter_node`: Creates a draft answer by analyzing the research notes.
   - `critic_node`: Reviews the drafted answer, adds sources, and prepares the final answer with feedback.

8. **LangGraph Construction**:  
   The nodes are arranged in a directed graph using `StateGraph` from LangGraph, which defines the sequence of steps: researcher → summarizer → drafter → critic → END. The graph is compiled and ready to execute.

9. **Running the Graph**:  
   In the `main` block, the user is prompted to input a research topic or question. The graph is then invoked, and the program outputs the final answer along with a review of the drafted response.

## Requirements

- Python 3.7 or higher
- `langchain` library
- `langchain_core` library
- `langgraph` library
- `tavily_search` library
- `python-dotenv` library
- OpenAI API key (GPT-4)
- Tavily API key

You can install the required dependencies with the following:

```bash
pip install langchain langchain-core langgraph tavily_search python-dotenv
```

## Usage

1. Set up your `.env` file with your OpenAI and Tavily API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

2. Run the script:

```bash
python research_assistant.py
```

3. Enter a research query when prompted, such as:
   - "What are the latest trends in artificial intelligence?"
   - "How do machine learning models impact the healthcare industry?"

4. The program will return a polished final answer, followed by a critique of the draft.

## Example

Input query:  
`"What is quantum computing and how does it work?"`

Output:
```
===== FINAL ANSWER =====
Quantum computing is a type of computation that takes advantage of quantum mechanical phenomena, such as superposition and entanglement, to process information in ways classical computers cannot. By using quantum bits (qubits) instead of binary bits, quantum computers can solve certain complex problems exponentially faster.

Key trends in quantum computing include advancements in quantum error correction, quantum cryptography, and the development of practical quantum algorithms. For example, companies like IBM and Google are investing heavily in quantum research, and startups are exploring quantum applications in fields like drug discovery and materials science.

===== REVIEW =====
The answer provides a clear and concise explanation of quantum computing. It includes important trends and real-world examples, such as IBM and Google's quantum research efforts. The answer is well-structured and covers the topic comprehensively.
```

## Conclusion

This project demonstrates how AI and automation can assist in performing comprehensive research tasks. By integrating OpenAI's GPT, Tavily, and LangChain, we create a seamless experience where a user can input a query, and the system will automatically research, summarize, draft, and critique an answer in real-time.

