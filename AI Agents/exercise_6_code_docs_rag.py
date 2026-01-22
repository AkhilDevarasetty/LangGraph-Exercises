import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.tools import tool
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START

load_dotenv(override=True)


# pdf_path = "python_async_guide.pdf"

## for absolute path,
# __file__ is the path of the current file - /Users/niharikainala/Documents/LangGraph-Exercises/AI Agents/exercise_6_code_docs_rag.py
# os.path.dirname(__file__) is the directory of the current file - /Users/niharikainala/Documents/LangGraph-Exercises/AI Agents
# os.path.join(os.path.dirname(__file__), "python_async_guide.pdf") is the absolute path of the pdf file - /Users/niharikainala/Documents/LangGraph-Exercises/AI Agents/python_async_guide.pdf
pdf_path = os.path.join(os.path.dirname(__file__), "python_async_guide.pdf")

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found at {pdf_path}")

try:
    pdf_loader = PyPDFLoader(pdf_path)
    pages = pdf_loader.load()
    print(f"PDF loaded successfully with {len(pages)} pages")
except Exception as e:
    print(f"Error loading pdf: {e}")
    raise

page_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
splitted_pages = page_splitter.split_documents(pages)
print(f"PDF splitted into {len(splitted_pages)} chunks")

## ChromaDB Vector Store Setup - requires documents, persist_directory, embeddings, collection_name
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db_code_docs")

collection_name = "python_code_docs"

if os.path.exists(persist_directory):
    print(f"Loading existing ChromaDb vector store from {persist_directory}")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
else:
    print(f"Creating new ChromaDb vector store at {persist_directory}")
    vectorstore = Chroma.from_documents(
        documents=splitted_pages,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

print(f"Vector store created successfully at {persist_directory}")

##Retriver Setup to retrive the sematic search results
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


@tool
def retriver_tool(query: str) -> str:
    """
    This tool searches and returns the information related to python async guide based on the query provided.
    """
    # docs = retriever.invoke(query)

    docs = vectorstore.similarity_search_with_score(query, k=3)

    if not docs:
        return "No relevant information found in the document."

    print(f"\n Query: {query}")
    print("-" * 50)
    for i, (doc, score) in enumerate(docs):
        print(
            f"📊Document {i + 1} (Similarity: {score:.4f}): {doc.page_content[:100]}..."
        )

    results = []

    for i, (doc, score) in enumerate(docs):
        results.append(f"Document {i + 1}:\n{doc.page_content}")

    return "\n\n".join(results)


## Graph Setup Steps


## Define State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


tools = [retriver_tool]

model = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)


## LLM Node
def call_llm(state: AgentState) -> AgentState:
    """Call the LLM with the current state and return the updated state."""

    system_prompt = """
    You are an intelligent AI assistant who answers questions about Python async guide based on the python async guidePDF document loaded into your knowledge base.
    Use the retriver tool available to answer questions about the python async guide. You can make multiple calls if needed.
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    Please always cite the specific parts of the documents you use in your answers.
    """

    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])

    response = model.invoke(messages)

    return {"messages": [response]}


tool_dict = {tool.name: tool for tool in tools}


## Tool Node
def retriver_node(state: AgentState) -> AgentState:
    """Call the retriver tool by extracting the last message from the state and invoking the tool."""

    last_message = state["messages"][-1]

    tool_calls = last_message.tool_calls

    results = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print(
            f"Calling tool: {tool_name} with the query: {tool_args.get('query', 'No query provided')}"
        )

        if tool_name not in tool_dict:
            print(f"Tool {tool_name} is not available")
            tool_result = "Incorrect tool name, please check and use in the list of available tools"
        else:
            print(f"Invoking tool {tool_name} with args {tool_args}")
            tool_result = tool_dict[tool_name].invoke(tool_args)

        results.append(
            ToolMessage(
                tool_call_id=tool_call["id"], name=tool_name, content=tool_result
            )
        )

    print("Tools Execution is completed and returning back to the LLM")
    return {"messages": results}


## Should Continue
def should_continue(state: AgentState) -> bool:
    """
    This function decides whether to continue the conversation or not.
    """
    return (
        hasattr(state["messages"][-1], "tool_calls")
        and len(state["messages"][-1].tool_calls) > 0
    )


graph = StateGraph(AgentState)

graph.add_node("call_llm", call_llm)
graph.add_node("retriver_node", retriver_node)

graph.add_edge(START, "call_llm")
graph.add_conditional_edges(
    "call_llm", should_continue, {True: "retriver_node", False: END}
)
graph.add_edge("retriver_node", "call_llm")

app = graph.compile()


def run_rag_agent():
    user_query = input("Please ask your question about the python async concepts: ")

    while True:
        if user_query.lower() in ["exit", "quit"]:
            print("Thank you for using the Python Async RAG agent. Goodbye!")
            break
        else:
            response = app.invoke({"messages": HumanMessage(content=user_query)})
            print(
                f"\n AI Response: \n {'=' * 50}\n{response['messages'][-1].content}\n {'=' * 50}"
            )
            user_query = input(
                "\n Please ask your question about the python async concepts: "
            )


if __name__ == "__main__":
    run_rag_agent()


# text = """# Lists in Python
# Lists are mutable sequences. You can create them using square brackets.
# Example:
# my_list = [1, 2, 3]
# my_list.append(4)
# ## List Methods
# - append(): Adds item to end
# - pop(): Removes last item
# """

# text_splitter_small = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
# small_chunks = text_splitter_small.split_text(text)

# medium_text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
# medium_chunks = medium_text_splitter.split_text(text)

# print("====SMALL CHUNKS====")

# for i, chunk in enumerate(small_chunks):
#     print(f"Chunk {i + 1}: {chunk}")
#     print("-" * 50)

# print("====MEDIUM CHUNKS====")

# for i, chunk in enumerate(medium_chunks):
#     print(f"Chunk {i + 1}: {chunk}")
#     print("-" * 50)
