import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START

load_dotenv(override=True)

pdf_path = os.path.join(os.path.dirname(__file__), "Stock_Market_Performance_2024.pdf")

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

try:
    pdf_loader = PyPDFLoader(pdf_path)
    pages = pdf_loader.load()
    print(f"PDF loaded successfully and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


splitted_pages = text_splitter.split_documents(pages)
print(f"Splitted the PDF into {len(splitted_pages)} chunks")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")
collection_name = "stock_market"

# Check if vector store already exists
if os.path.exists(persist_directory):
    print(f"Loading existing ChromaDB vector store from: {persist_directory}")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
else:
    print("Creating new ChromaDB vector store...")
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    vectorstore = Chroma.from_documents(
        documents=splitted_pages,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    print(f"✅ Created ChromaDB vector store at: {persist_directory}")

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})


@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the Stock Market Performance 2024 document.
    """
    # Use similarity_search_with_score to see rankings
    # docs_with_scores = vectorstore.similarity_search_with_score(query, k=2)

    docs = retriever.invoke(query)

    # if not docs_with_scores:
    #     return "No relevant information found in the document."

    if not docs:
        return "No relevant information found in the document."

    results = []

    # for i, (doc, score) in enumerate(docs_with_scores):
    #     results.append(
    #         f"Document {i + 1} (Similarity: {score:.4f}):\n{doc.page_content}"
    #     )

    for i, doc in enumerate(docs):
        results.append(f"Document {i + 1}:\n{doc.page_content}")

    return "\n\n".join(results)


# Test the retriever
# print("\n=== Testing Retriever ===")
# test_result = retriever_tool.invoke(
#     {"query": "What is the S&P 500 index performance in 2024?"}
# )
# print(test_result)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


tools = [retriever_tool]

model = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)


## LLM Agent Node
def call_llm(state: AgentState) -> AgentState:
    """Call the LLM with the current state and return the updated state."""
    system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    result = model.invoke(messages)
    return {"messages": [result]}


tool_dict = {tool.name: tool for tool in tools}


## Retriver Node
def retriver_node(state: AgentState) -> AgentState:
    """Checks the last message for tool calls and executes them."""

    tool_calls = state["messages"][-1].tool_calls
    results = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print(
            f"Calling tool: {tool_name} with query: {tool_args.get('query', 'No query provided')}"
        )

        if tool_name not in tool_dict:
            print(f"Tool: {tool_name} does not exist.")
            tool_result = "Incorrect tool name, please retry and select tool from list of available tools."
        else:
            tool_result = tool_dict[tool_name].invoke(tool_args)
            print(f"Result length: {len(str(tool_result))}")

        results.append(
            ToolMessage(
                tool_call_id=tool_call["id"], name=tool_name, content=str(tool_result)
            )
        )

    print("Tools execution complete. Back to the model!")
    return {"messages": results}


def should_continue(state: AgentState) -> bool:
    """Check if the last message contains tool calls."""
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0


graph = StateGraph(AgentState)
graph.add_node("call_llm", call_llm)
graph.add_node("retriver_node", retriver_node)

graph.add_edge(START, "call_llm")
graph.add_conditional_edges(
    "call_llm",
    should_continue,
    {
        True: "retriver_node",
        False: END,
    },
)
graph.add_edge("retriver_node", "call_llm")

rag_app = graph.compile()


def run_rag_agent():
    while True:
        user_query = input(
            "Ask a question about the stock market performance in 2024: "
        )
        if user_query in ["exit", "quit"]:
            print("Exiting the RAG agent. Goodbye!")
            break

        user_message = HumanMessage(content=user_query)
        state = {"messages": [user_message]}

        result = rag_app.invoke(state)

        print("\n=== RAG Agent Response ===")
        print(result["messages"][-1].content)
        print("=" * 50)


if __name__ == "__main__":
    run_rag_agent()
