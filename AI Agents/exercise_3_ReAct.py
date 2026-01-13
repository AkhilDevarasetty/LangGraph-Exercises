from typing import TypedDict, Sequence, Annotated, Union
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    BaseMessage,
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


@tool
def add_two_numbers(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
    """This tool adds two numbers"""
    return round(x + y, 2)


@tool
def multiply_two_numbers(
    x: Union[int, float], y: Union[int, float]
) -> Union[int, float]:
    """This tool multiplies two numbers"""
    return round(x * y, 2)


@tool
def divide_two_numbers(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
    """This tool divides two numbers"""
    return round(x / y, 2)


@tool
def subtract_two_numbers(
    x: Union[int, float], y: Union[int, float]
) -> Union[int, float]:
    """This tool subtracts two numbers"""
    return round(x - y, 2)


tools = [
    add_two_numbers,
    multiply_two_numbers,
    divide_two_numbers,
    subtract_two_numbers,
]

model = ChatOpenAI(model_name="gpt-4o", temperature=0).bind_tools(tools)


def call_llm(state: AgentState) -> AgentState:
    system_message = SystemMessage(
        content="You are a helpful assistant that can perform basic arithmetic operations. You can use the tools below to perform the operations."
    )
    messages = [system_message] + state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


graph = StateGraph(AgentState)
graph.add_node("llm_process", call_llm)
graph.add_edge(START, "llm_process")

tool_node = ToolNode(tools=tools)
graph.add_node("tool_node", tool_node)

graph.add_conditional_edges(
    "llm_process",
    should_continue,
    {
        "continue": "tool_node",  # When tools need to be called, go to tool_node
        "end": END,  # When no tools are needed, end the graph
    },
)
graph.add_edge("tool_node", "llm_process")  # After tools execute, return to llm_process

reAct_agent = graph.compile()


def print_stream(stream):
    for value in stream:
        message = value["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


user_input = input("Enter your message: ")

user_message = {"messages": [HumanMessage(content=user_input)]}
print_stream(reAct_agent.stream(user_message, stream_mode="values"))
