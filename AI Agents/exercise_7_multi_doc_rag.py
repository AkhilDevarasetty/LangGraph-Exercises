"""
Exercise 7: Multi-Document RAG with Query Routing

Challenge:
1. Initialize multiple vector stores (one per document)
2. Create a "Router" tool that decides which document to search
3. Create a "Retriever" tool that searches the chosen vector store(s)
4. Wire it all together in a LangGraph

documents available:
- docs/python_async_guide.pdf (Python Async Guide)
- docs/javascript_promises_guide.pdf (JavaScript Promises Guide)
- docs/programming_concepts.pdf (General Programming Concepts)
"""

import os
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

load_dotenv(override=True)

# ==============================================================================
# 1. SETUP & INITIALIZATION
# ==============================================================================

# TODO: Define a configuration dictionary for documents
# Tip: It should store path, collection_name, and maybe chunking params for each
DOCUMENTS = {
    # "python": { ... },
    "python": {
        "path": "docs/python_async_guide.pdf",
        "collection_name": "python_async_guide",
        "description": "Guide to Python Async/Await",
        "chunk_size": 1200,
        "chunk_overlap": 250,
    },
    # "javascript": { ... },
    "javascript": {
        "path": "docs/javascript_promises_guide.pdf",
        "collection_name": "javascript_promises_guide",
        "description": "Guide to JavaScript Promises",
        "chunk_size": 1000,
        "chunk_overlap": 200,
    },
    # "concepts": { ... }
    "concepts": {
        "path": "docs/programming_concepts.pdf",
        "collection_name": "programming_concepts",
        "description": "General Programming Concepts",
        "chunk_size": 800,
        "chunk_overlap": 150,
    },
}


def initialize_vector_stores():
    """
    TODO: Initialize a vector store for EACH document in the DOCUMENTS dict.

    1. Loop through DOCUMENTS
    2. Load PDF
    3. Split text (maybe use different chunk sizes?)
    4. Create/Load ChromaDB collection

    Returns:
        dict: A dictionary mapping document_key -> Chroma vector store object
    """
    print("\n" + "=" * 80)
    print("🚀 INITIALIZING VECTOR STORES")
    print("=" * 80)
    stores = {}
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    for doc_key, doc_config in DOCUMENTS.items():
        pdf_path = os.path.join(os.path.dirname(__file__), doc_config["path"])

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


# Global variable to hold our stores (simulating a "database" connection)
# In a real app, you might pass this in state or config
VECTOR_STORES = {}

# ==============================================================================
# 2. TOOLS
# ==============================================================================


@tool
def route_query(query: str) -> str:
    """
    TODO: Analyze the user's query and decide which collection(s) to search.

    Args:
        query: The user's question

    Returns:
        JSON string or structured output indicating:
        - Which collections to search (e.g., ["python", "javascript"])
        - Reasoning why
    """
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

    # Extract JSON from response for cleaner logging
    import json

    try:
        # Try to parse the JSON from the response
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
    """
    Retrieve relevant passages from one or more document-specific vector stores.

    This tool is used AFTER routing. The router decides which document collections
    are likely relevant (e.g., "python", "javascript", "concepts"), and this tool
    searches the corresponding Chroma vector stores and returns the top-k chunks.

    Expected usage flow:
        1) route_query(query) -> decides collections
        2) multi_retriever_tool(query, collections) -> fetches context
        3) LLM uses returned context to answer the question

    Args:
        query: The user's natural-language question to search for.
        collections: Comma-separated list of collection keys to search
            (e.g., "python,javascript"). Whitespace is allowed and will be stripped.

    Returns:
        A single formatted string containing retrieved chunks grouped by source
        collection. Each chunk includes its page number (if available) and content.
        If an unknown collection key is provided, an error block is included for that key.
    """
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
# 3. GRAPH & AGENT
# ==============================================================================


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # Do we need anything else in state? Maybe 'routing_decision'?


tools = [route_query, multi_retriever_tool]
tool_dict = {tool.name: tool for tool in tools}
llm_model = ChatOpenAI(model="gpt-4o").bind_tools(tools)


def call_llm(state: AgentState):
    """
    The main agent node.
    It should have access to the routing tool and retrieval tool.
    """
    SYSTEM_PROMPT = """
    You are a helpful technical tutor answering questions using retrieved context from documentation.

    Rules:
    - Use ONLY the provided context to answer. If the context is insufficient, say what’s missing and ask a short follow-up question.
    - Be concise and correct. Prefer bullet points for explanations.
    - If the user asks for a comparison, explicitly compare side-by-side.
    - When you mention a fact from the context, reference its SOURCE label (e.g., "SOURCE: python") when possible.
    """

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    response = llm_model.invoke(messages)
    return {"messages": [response]}


def retriever_node(state: AgentState) -> AgentState:
    """Checks weather the last message contains the tool_calls and executes"""

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
                # Count retrieved chunks
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
# 4. EXECUTION
# ==============================================================================


def main():
    global VECTOR_STORES
    VECTOR_STORES = initialize_vector_stores()

    # TODO: Compile graph
    # app = ...

    print("\n" + "=" * 80)
    print("✨ MULTI-DOC RAG AGENT READY!")
    print("=" * 80)
    print("💡 Available collections: python, javascript, concepts")
    print("💡 Type 'quit' or 'exit' to stop\n")

    while True:
        user_input = input("\n❓ User: ")
        if user_input.lower() in ["quit", "exit"]:
            print("\n👋 Goodbye!\n")
            break
        else:
            print("\n" + "─" * 80)
            llm_response = multi_doc_agent.invoke(
                {"messages": [HumanMessage(content=user_input)]}
            )
            final_msg = llm_response["messages"][-1]
            print("\n" + "=" * 80)
            print("🤖 AI RESPONSE")
            print("=" * 80)
            print(f"\n{final_msg.content}\n")
            print("─" * 80)


if __name__ == "__main__":
    main()
