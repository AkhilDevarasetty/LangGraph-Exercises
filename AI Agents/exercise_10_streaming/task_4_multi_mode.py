"""
Exercise 10 - Task 4: Multi-Mode Streaming with Colors

Goal: Combine updates, messages, and custom modes with color-coded output

YOUR TASK:
1. Install and import colorama for colored terminal output
2. Add stream_mode="updates" to see state changes
3. Handle all three modes: updates, messages, custom
4. Color-code each mode:
   - Blue for state updates
   - Green for LLM tokens
   - Yellow for custom progress
5. Test all three modes work simultaneously

Based on: Task 3 (Custom Progress Updates)
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

# TODO 1: Import get_stream_writer from langgraph.config
# Hint: from langgraph.config import get_stream_writer
import tiktoken
import asyncio
from langgraph.config import get_stream_writer

# TODO 1: Install and import colorama for colored output
# Run: pip install colorama
# Then import: from colorama import Fore, Style, init
# And initialize: init(autoreset=True)
from colorama import Fore, Style, init

init(autoreset=True)

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


# ==============================================================================
# TASK 2: YOUR ASYNC RETRIEVAL CODE GOES HERE
# ==============================================================================

# TODO 1: Create async helper function to query a single collection
# Hint: async def query_single_collection_async(collection_key: str, query: str) -> str:
#           """Query a single collection asynchronously."""
#           # Get vector store
#           # Create retriever
#           # Query (use await if needed)
#           # Format results
#           # Return formatted string


async def query_single_collection_async(collection_key: str, query: str) -> str:
    """Query a single collection asynchronously."""
    global VECTOR_STORES

    # Check if collection exists
    if collection_key not in VECTOR_STORES:
        return f"===SOURCE: {collection_key}===\n[ERROR]: Unknown collection key: {collection_key}"

    # Get vector store
    vector_store = VECTOR_STORES[collection_key]

    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    # Query retriever in a thread (to make it truly async!)
    docs = await asyncio.to_thread(retriever.invoke, query)

    # Format results
    doc_results = []
    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", "N/A")
        doc_results.append(f"Document {i + 1} (page={page}):\n{doc.page_content}")

    return f"===SOURCE: {collection_key}===\n" + "\n\n".join(doc_results)


# TODO 2: Convert multi_retriever_tool to use async
@tool
async def multi_retriever_tool_async(query: str, collections: str) -> str:
    """
    Retrieve relevant passages from multiple collections CONCURRENTLY.

    YOUR TASK:
    1. Parse collections string into list
    2. Create list of async tasks (one per collection)
    3. Use asyncio.gather() to run all tasks concurrently
    4. Combine results
    5. Return combined string

    HINT:
    tasks = [query_single_collection_async(col, query) for col in collection_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    """
    collection_keys = [c.strip() for c in collections.split(",") if c.strip()]

    # Create list of async tasks
    async_tasks = [
        query_single_collection_async(col_key, query) for col_key in collection_keys
    ]

    # Run all tasks concurrently with asyncio.gather()
    results = await asyncio.gather(*async_tasks)

    # Combine results
    combined_results = "\n".join(results)

    # Return combined string
    return combined_results


# Keep the old sync version for comparison
@tool
def multi_retriever_tool(query: str, collections: str) -> str:
    """OLD SYNC VERSION - Keep for comparison"""
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


# TODO 3: After implementing async functions, switch to async tool
# Change this line to: tools = [route_query, multi_retriever_tool_async]
tools = [route_query, multi_retriever_tool_async]
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
    # TODO 2: Get stream writer to send progress updates
    # Hint: writer = get_stream_writer()
    writer = get_stream_writer()

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

        # Send progress update BEFORE calling tool
        writer(
            {"status": f"{tool_icon} Calling {tool_name}...", "timestamp": time.time()}
        )

        if tool_name not in tool_dict:
            print(f"   ❌ Error: Unknown tool '{tool_name}'")
            tool_result = f"Incorrect tool name: {tool_name}, please check and use in the list of available tools"
        else:
            # Check if tool is async
            if tool_name == "multi_retriever_tool_async":
                # Run async tool in sync context
                tool_result = asyncio.run(tool_dict[tool_name].ainvoke(tool_args))
            else:
                # Regular sync tool
                tool_result = tool_dict[tool_name].invoke(tool_args)

            if tool_name in ["multi_retriever_tool", "multi_retriever_tool_async"]:
                chunk_count = tool_result.count("Document ")
                # Send progress update AFTER successful retrieval
                writer(
                    {
                        "status": f"✅ Retrieved {chunk_count} chunks from {tool_args.get('collections', 'unknown')}",
                        "timestamp": time.time(),
                    }
                )

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
    print("✨ MULTI-DOC RAG AGENT WITH MULTI-MODE STREAMING!")
    print("=" * 80)
    print("💡 Available collections: python, javascript, concepts")
    print("💡 Type 'quit' or 'exit' to stop\n")

    while True:
        user_input = input("\n❓ User: ")
        if user_input.lower() in ["quit", "exit"]:
            print("\n👋 Goodbye!\n")
            break

        print("\n" + "─" * 80)

        # Typing indicator
        print("🤖 AI: 💭 Thinking...", end="", flush=True)

        # Initialize metrics tracking
        start_time = time.time()
        first_token_time = None
        token_count = 0

        # TODO 2: Add "updates" mode to see state changes
        # Change from: stream_mode=["messages", "custom"]
        # To: stream_mode=["updates", "messages", "custom"]
        llm_response = multi_doc_agent.stream(
            {"messages": [HumanMessage(content=user_input)]},
            stream_mode=["updates", "messages", "custom"],  # TODO: Add "updates" here
        )
        encoder = tiktoken.encoding_for_model("gpt-4o")

        # TODO 3: Handle all three modes with color coding
        # Add a new elif for mode == "updates"
        # Use Fore.BLUE for updates, Fore.GREEN for messages, Fore.YELLOW for custom

        for mode, chunk in llm_response:
            # TODO 4: Add handler for "updates" mode
            # Hint: if mode == "updates":
            #           node_name = list(chunk.keys())[0]
            #           print(f"\n{Fore.BLUE}[STATE] Node: {node_name}{Style.RESET_ALL}", flush=True)

            if mode == "messages":
                message_chunk, metadata = chunk
                if (
                    message_chunk.content
                    and metadata.get("langgraph_node") == "call_llm"
                ):
                    if first_token_time is None:
                        print("\r🤖 AI: ", end="", flush=True)
                        first_token_time = time.time()

                    # TODO 5: Add color to LLM tokens
                    # Hint: print(f"{Fore.GREEN}{message_chunk.content}{Style.RESET_ALL}", end="", flush=True)
                    print(
                        f"{Fore.GREEN}{message_chunk.content}{Style.RESET_ALL}",
                        end="",
                        flush=True,
                    )
                    actual_token = len(encoder.encode(message_chunk.content))
                    token_count += actual_token

            elif mode == "custom":
                status = chunk.get("status", "")
                timestamp = chunk.get("timestamp", 0)
                elapsed = timestamp - start_time
                # TODO 6: Add color to custom progress
                # Hint: print(f"\n{Fore.YELLOW}[{elapsed:.2f}s] {status}{Style.RESET_ALL}", flush=True)
                print(
                    f"\n{Fore.YELLOW}[{elapsed:.2f}s] {status}{Style.RESET_ALL}",
                    flush=True,
                )

            elif mode == "updates":
                node_name = list(chunk.keys())[0]
                print(
                    f"\n {Fore.BLUE} [STATE] Node: {node_name}{Style.RESET_ALL}",
                    flush=True,
                )

        # Calculate and print metrics
        total_time = time.time() - start_time
        ttft = first_token_time - start_time if first_token_time else 0
        print(
            f"\n⏱️  Metrics: TTFT={ttft:.2f}s | Total={total_time:.2f}s | Tokens={token_count}"
        )

        print("─" * 80)


if __name__ == "__main__":
    main()
