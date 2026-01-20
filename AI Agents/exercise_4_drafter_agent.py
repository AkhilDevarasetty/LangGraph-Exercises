from typing import TypedDict, Sequence, Annotated
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv(override=True)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


document_content = ""


##-- Tool Functions ---


@tool
def update_document(content: str) -> str:
    """Updates the document content with the given content."""
    global document_content
    document_content = content
    return f"Document has been updated Successfully! and the current document content is:\n {document_content}"


@tool
def save_document(filename: str) -> str:
    """Saves the current document content to a text file and finishes the process.

    Args:
        filename (str): Name of the text file to save the document content.

    Returns:
        str: A message indicating the success or failure of the save operation.
    """

    global document_content

    if not filename.endswith(".txt"):
        filename += ".txt"

    try:
        with open(filename, "w") as f:
            f.write(document_content)
            print(f"📁Document has been saved to {filename}")
        return f"Document has been successfully saved to {filename}"
    except Exception as e:
        print(f"❌Error saving document: {str(e)}")
        return f"Error saving document: {str(e)}"


tools = [update_document, save_document]


model = ChatOpenAI(model_name="gpt-4o", temperature=0).bind_tools(tools)


##-- Node Functions ---
def call_llm(state: AgentState) -> AgentState:
    system_message = f"""You are a Drafter, a helpful writing assistant. You are going to update and save a document.
    
    Tools available:
    1. update_document: Updates the document content with the given content.
    2. save_document: Saves the current document content to a text file and finishes the process.
    3. Make sure to always show the current document content after modifications. 
    
    The current document content is:{document_content}
    """
    system_prompt = SystemMessage(content=system_message)

    if not state["messages"]:
        user_input = input("What would you like to create? :")
        user_prompt = HumanMessage(content=user_input)
    else:
        user_input = input("What would you like to update in the document? :")
        user_prompt = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_prompt]

    llm_response = model.invoke(all_messages)

    print(f"🤖 AI Response: {llm_response.content}")
    if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
        print(f"🛠️ Using tool: {[tc['name'] for tc in llm_response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_prompt, llm_response]}


def should_continue(state: AgentState) -> str:
    """Determines whether to continue the conversation or end it."""
    messages = state["messages"]

    if not messages:
        return "continue"

    last_message = messages[-1]

    if (
        isinstance(last_message, ToolMessage)
        and "saved" in last_message.content.lower()
        and "document" in last_message.content.lower()
    ):
        return "end"

    return "continue"


##-- State Graph ---
graph = StateGraph(AgentState)

graph.add_node("call_llm", call_llm)
graph.add_edge(START, "call_llm")

tool_node = ToolNode(tools=tools)
graph.add_node("tool_node", tool_node)
graph.add_edge("call_llm", "tool_node")
graph.add_conditional_edges(
    "tool_node", should_continue, {"continue": "call_llm", "end": END}
)

app = graph.compile()


def print_messages(messages: list[BaseMessage]):
    """Prints the messages in a formatted way."""

    if not messages:
        return
    for message in messages:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ Tool Result:\n{message.content}\n")


def run_document_draft_agent():
    print("\n ===== Document Draft Agent ===== \n")

    state = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n ===== Document Draft Agent Finished ===== \n")


if __name__ == "__main__":
    run_document_draft_agent()
