"""
Exercise 10 - Task 1: Basic Token Streaming

Goal: Stream LLM responses token-by-token for immediate user feedback

YOUR TASK:
1. Replace .invoke() with .stream(stream_mode="messages") in the main loop
2. Display tokens progressively with flush=True
3. Add typing indicator before first token
4. Handle empty chunks gracefully
5. Track metrics (TTFT, total time, token count)

Based on: Exercise 7 Multi-Document RAG Agent
"""

import os
import time
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
import tiktoken

load_dotenv(override=True)

# ==============================================================================
# 1. SETUP & INITIALIZATION (Same as Exercise 7)
# ==============================================================================

DOCUMENTS = {
    "python": {
        "path": "docs/python_async_guide.pdf",
        "collection_name": "python_async_guide",
        "description": "Guide to Python Async/Await",
        "chunk_size": 1200,
        "chunk_overlap": 250,
    },
    "javascript": {
        "path": "docs/javascript_promises_guide.pdf",
        "collection_name": "javascript_promises_guide",
        "description": "Guide to JavaScript Promises",
        "chunk_size": 1000,
        "chunk_overlap": 200,
    },
    "concepts": {
        "path": "docs/programming_concepts.pdf",
        "collection_name": "programming_concepts",
        "description": "General Programming Concepts",
        "chunk_size": 800,
        "chunk_overlap": 150,
    },
}


def initialize_vector_stores():
    """Initialize a vector store for EACH document in the DOCUMENTS dict."""
    print("\n" + "=" * 80)
    print("🚀 INITIALIZING VECTOR STORES")
    print("=" * 80)
    stores = {}
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    for doc_key, doc_config in DOCUMENTS.items():
        pdf_path = os.path.join(os.path.dirname(__file__), "..", doc_config["path"])

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"No file is found at {pdf_path}")

        try:
            print(f"\n📄 Loading '{doc_key}' document...")
            pdf_loader = PyPDFLoader(pdf_path)
            pdf_pages = pdf_loader.load()
        except Exception as e:
            print(f"❌ Error loading {doc_key} pdf: {e}")
            raise

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=doc_config["chunk_size"],
            chunk_overlap=doc_config["chunk_overlap"],
        )

        splitted_docs = text_splitter.split_documents(pdf_pages)

        print(
            f"   ✂️  Split into {len(splitted_docs)} chunks (size={doc_config['chunk_size']}, overlap={doc_config['chunk_overlap']})"
        )

        persist_directory = os.path.join(
            os.path.dirname(__file__), f"chroma_db_multi/{doc_key}"
        )

        if os.path.exists(persist_directory):
            print(f"   ✅ Loaded existing vector store")
            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                collection_name=doc_config["collection_name"],
            )
        else:
            print(f"   🆕 Creating new vector store")
            vector_store = Chroma.from_documents(
                persist_directory=persist_directory,
                documents=splitted_docs,
                embedding=embeddings,
                collection_name=doc_config["collection_name"],
            )

        stores[doc_key] = vector_store
    return stores


VECTOR_STORES = {}

# ==============================================================================
# 2. TOOLS (Same as Exercise 7)
# ==============================================================================


@tool
def route_query(query: str) -> str:
    """Analyze the user's query and decide which collection(s) to search."""
    llm_model = ChatOpenAI(model="gpt-4o")

    system_prompt = """
        You are an expert document routing assistant.
        Your task is to analyze the user's query and decide which document collection(s) should be searched to answer it.

        The available document collections are:
        1. "python" - Contains a guide on Python Async/Await programming.
        2. "javascript" - Contains a guide on JavaScript Promises and Async programming.
        3. "concepts" - Contains a guide on General Programming Concepts (CS fundamentals).

        Rules:
        - If the query mentions specific languages (Python/JS), include the relevant collection.
        - If the query is about general concepts (e.g., "what is a variable"), include "concepts".
        - If the query compares two things (e.g., "Python async vs JS promises"), include BOTH "python" and "javascript".
        - If the query is ambiguous, you can include multiple relevant collections.

        Output Format:
        You MUST return a valid JSON object in the following format:
        {
            "collections": ["python", "javascript"],
            "reasoning": "The user is asking for a comparison between Python and JS async features."
        }
    """

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=query)]
    response = llm_model.invoke(messages)

    import json

    try:
        response_text = response.content
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        else:
            json_str = response_text
        routing_data = json.loads(json_str)
        print(f"\n🧭 ROUTING: {', '.join(routing_data.get('collections', []))}")
        print(f"   💭 Reasoning: {routing_data.get('reasoning', 'N/A')}")
    except:
        print(f"\n🧭 ROUTING DECISION: {response.content[:100]}...")

    return response.content


@tool
def multi_retriever_tool(query: str, collections: str) -> str:
    """Retrieve relevant passages from one or more document-specific vector stores."""
    global VECTOR_STORES
    collection_keys = [c.strip() for c in collections.split(",") if c.strip()]
    vector_store_results = []
    for collection in collection_keys:
        if collection not in VECTOR_STORES:
            vector_store_results.append(
                f"=== SOURCE : {collection}===\n [ERROR]: Unknown collection key: {collection}"
            )
            continue
        vector_store = VECTOR_STORES[collection]
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )
        doc_results = []
        docs = retriever.invoke(query)
        for i, doc in enumerate(docs):
            page = doc.metadata.get("page", "N/A")
            doc_results.append(f"Document {i + 1} (page={page}):\n{doc.page_content}")

        vector_store_results.append(
            f"===SOURCE: {collection}===\n" + "\n\n".join(doc_results)
        )

    return "\n\n" + ("-" * 80) + "\n\n".join(vector_store_results)


# ==============================================================================
# 3. GRAPH & AGENT (Same as Exercise 7)
# ==============================================================================


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


tools = [route_query, multi_retriever_tool]
tool_dict = {tool.name: tool for tool in tools}
llm_model = ChatOpenAI(model="gpt-4o").bind_tools(tools)


def call_llm(state: AgentState):
    """The main agent node."""
    SYSTEM_PROMPT = """
    You are a helpful technical tutor answering questions using retrieved context from documentation.

    Rules:
    - Use ONLY the provided context to answer. If the context is insufficient, say what's missing and ask a short follow-up question.
    - Be concise and correct. Prefer bullet points for explanations.
    - If the user asks for a comparison, explicitly compare side-by-side.
    - When you mention a fact from the context, reference its SOURCE label (e.g., "SOURCE: python") when possible.
    """

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    response = llm_model.invoke(messages)
    return {"messages": [response]}


def retriever_node(state: AgentState) -> AgentState:
    """Execute tool calls from the last message."""
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls

    tool_total_result = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        query_preview = tool_args.get("query", "No query provided")
        if len(query_preview) > 50:
            query_preview = query_preview[:50] + "..."

        tool_icon = "🔧" if tool_name == "route_query" else "🔍"
        print(f"\n{tool_icon} CALLING: {tool_name}")
        print(f"   Query: '{query_preview}'")

        if tool_name not in tool_dict:
            print(f"   ❌ Error: Unknown tool '{tool_name}'")
            tool_result = f"Incorrect tool name: {tool_name}, please check and use in the list of available tools"
        else:
            tool_result = tool_dict[tool_name].invoke(tool_args)
            if tool_name == "multi_retriever_tool":
                chunk_count = tool_result.count("Document ")
                print(f"   ✅ Retrieved {chunk_count} chunks")

        tool_total_result.append(
            ToolMessage(
                tool_call_id=tool_call["id"], name=tool_name, content=tool_result
            )
        )
    return {"messages": tool_total_result}


def should_continue(state: AgentState) -> bool:
    return (
        hasattr(state["messages"][-1], "tool_calls")
        and len(state["messages"][-1].tool_calls) > 0
    )


graph = StateGraph(AgentState)
graph.add_node("call_llm", call_llm)
graph.add_node("retriver_node", retriever_node)
graph.add_edge(START, "call_llm")
graph.add_conditional_edges(
    "call_llm", should_continue, {True: "retriver_node", False: END}
)
graph.add_edge("retriver_node", "call_llm")
multi_doc_agent = graph.compile()

# ==============================================================================
# 4. EXECUTION - YOUR TASK: ADD STREAMING HERE!
# ==============================================================================


def main():
    global VECTOR_STORES
    VECTOR_STORES = initialize_vector_stores()

    print("\n" + "=" * 80)
    print("✨ MULTI-DOC RAG AGENT WITH TOKEN STREAMING!")
    print("=" * 80)
    print("💡 Available collections: python, javascript, concepts")
    print("💡 Type 'quit' or 'exit' to stop\n")

    while True:
        user_input = input("\n❓ User: ")
        if user_input.lower() in ["quit", "exit"]:
            print("\n👋 Goodbye!\n")
            break

        print("\n" + "─" * 80)

        # TODO 1: Add typing indicator
        # Hint: print("🤖 AI: 💭 Thinking...", end="", flush=True)
        print("🤖 AI: 💭 Thinking...", end="", flush=True)

        # TODO 2: Initialize metrics tracking
        # Hint: start_time = time.time()
        #       first_token_time = None
        #       token_count = 0
        start_time = time.time()
        first_token_time = None
        token_count = 0

        # TODO 4: Replace .invoke() with .stream(stream_mode="messages")
        # CURRENT CODE (Exercise 7):
        llm_response = multi_doc_agent.stream(
            {"messages": [HumanMessage(content=user_input)]}, stream_mode="messages"
        )
        encoder = tiktoken.encoding_for_model("gpt-4o")
        for message_chunk, metadata in llm_response:
            if message_chunk.content and metadata.get("langgraph_node") == "call_llm":
                if first_token_time is None:
                    print("\r🤖 AI: ", end="", flush=True)
                    first_token_time = time.time()

                print(message_chunk.content, end="", flush=True)
                actual_token = len(encoder.encode(message_chunk.content))
                token_count += actual_token

        # YOUR TASK: Replace the above 3 lines with streaming code
        # Hint: for message_chunk, metadata in multi_doc_agent.stream(..., stream_mode="messages"):
        #           if message_chunk.content:
        #               print(message_chunk.content, end="", flush=True)
        #               token_count += 1

        # TODO 5: Calculate and print metrics
        # Hint: total_time = time.time() - start_time
        #       ttft = first_token_time - start_time if first_token_time else 0
        #       print(f"\n⏱️  Metrics: TTFT={ttft:.2f}s | Total={total_time:.2f}s | Tokens={token_count}")
        total_time = time.time() - start_time
        ttft = first_token_time - start_time if first_token_time else 0
        print(
            f"\n⏱️  Metrics: TTFT={ttft:.2f}s | Total={total_time:.2f}s | Tokens={token_count}"
        )

        print("─" * 80)


if __name__ == "__main__":
    main()
