"""
ReAct Agent - Reasoning + Acting Pattern
=========================================

This file implements a ReAct (Reasoning + Acting) agent that can perform arithmetic operations.

WHAT IS ReAct?
--------------
ReAct is a pattern where the LLM:
1. REASONS: Analyzes the user's request and decides which tools to use
2. ACTS: Executes the tools
3. LOOPS: Reviews tool results and either continues or provides final answer

FLOW DIAGRAM:
-------------
START → llm_process → should_continue?
                      ├─ "continue" → tool_node → llm_process (loop)
                      └─ "end" → END

STATE MANAGEMENT WITH REDUCERS:
--------------------------------
The AgentState uses a REDUCER FUNCTION (add_messages) to manage state updates.

WITHOUT REDUCER (normal dict merge):
    state = {"messages": [msg1, msg2]}
    update = {"messages": [msg3]}
    Result: {"messages": [msg3]}  ❌ Lost msg1, msg2!

WITH REDUCER (add_messages):
    state = {"messages": [msg1, msg2]}
    update = {"messages": [msg3]}
    Result: {"messages": [msg1, msg2, msg3]}  ✅ Appended!

HOW REDUCERS WORK:
------------------
1. You declare the reducer in the type annotation:
   messages: Annotated[Sequence[BaseMessage], add_messages]
                                              ^^^^^^^^^^^^
                                              This is the reducer

2. LangGraph reads this when you create StateGraph(AgentState)

3. Every time a node returns an update, LangGraph uses the reducer:
   new_state = add_messages(old_state["messages"], update["messages"])

4. add_messages is a built-in reducer that appends messages instead of replacing

MESSAGE FLOW EXAMPLE:
---------------------
User: "What is 5 + 5?"

Step 1: Initial state
  {"messages": [HumanMessage(content="What is 5 + 5?")]}

Step 2: llm_process returns AIMessage with tool_calls
  {"messages": [HumanMessage(...), AIMessage(tool_calls=[...])]}

Step 3: tool_node executes tool and returns ToolMessage
  {"messages": [HumanMessage(...), AIMessage(...), ToolMessage(content="10")]}


Step 4: llm_process returns final AIMessage
  {"messages": [HumanMessage(...), AIMessage(...), ToolMessage(...), AIMessage(content="The answer is 10")]}

TOOLS:
------
- Tools return plain values (int, float, etc.)
- ToolNode automatically wraps results in ToolMessage objects
- You never manually create ToolMessage objects

QUICK REFERENCE:
----------------
Key Concepts:
  - Annotated: Adds metadata to type annotations (from typing module)
  - Sequence: Read-only list-like type (accepts list, tuple, etc.)
  - BaseMessage: Parent class for all message types (HumanMessage, AIMessage, ToolMessage, SystemMessage)
  - add_messages: Reducer function that APPENDS messages instead of replacing them
  - Reducer: A function that defines how to merge state updates (used by LangGraph)

Important:
  - model.invoke() returns AIMessage (not a dict or string)
  - LangGraph uses the add_messages reducer automatically when merging state
  - tool_calls is a list on AIMessage; empty list means no tools needed
  - ToolNode wraps tool return values in ToolMessage automatically
"""

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
    """
    State schema for the ReAct agent.

    Attributes:
        messages: A sequence of messages in the conversation.
                  Uses Annotated with add_messages reducer to APPEND new messages
                  instead of replacing the entire list.

    Why Sequence?
        - Accepts both list and tuple (flexible)
        - Signals read-only intent (we don't modify in-place)
        - More general than list or tuple alone

    Why add_messages reducer?
        - Without it: Each node update would OVERWRITE the entire message list
        - With it: Each node update APPENDS to the existing message list
        - This preserves conversation history throughout the ReAct loop
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]


# ============================================================================
# ARITHMETIC TOOLS
# ============================================================================
# These tools perform basic arithmetic operations.
# Note: They return plain values (int/float), NOT ToolMessage objects.
# The ToolNode will automatically wrap the return values in ToolMessage.
# ============================================================================


@tool
def add_two_numbers(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
    """Add two numbers and return the sum rounded to 2 decimal places."""
    return round(x + y, 2)


@tool
def multiply_two_numbers(
    x: Union[int, float], y: Union[int, float]
) -> Union[int, float]:
    """Multiply two numbers and return the product rounded to 2 decimal places."""
    return round(x * y, 2)


@tool
def divide_two_numbers(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
    """Divide two numbers and return the quotient rounded to 2 decimal places."""
    return round(x / y, 2)


@tool
def subtract_two_numbers(
    x: Union[int, float], y: Union[int, float]
) -> Union[int, float]:
    """Subtract two numbers and return the difference rounded to 2 decimal places."""
    return round(x - y, 2)


# ============================================================================
# TOOLS SETUP
# ============================================================================
# Tools are functions that the LLM can call to perform specific operations.
# The @tool decorator converts regular Python functions into LangChain tools.
#
# Important: Tools return PLAIN VALUES (int, float, str, etc.)
# The ToolNode will automatically wrap these in ToolMessage objects.
# ============================================================================

tools = [
    add_two_numbers,
    multiply_two_numbers,
    divide_two_numbers,
    subtract_two_numbers,
]

# Bind tools to the model so it knows what functions are available
model = ChatOpenAI(model_name="gpt-4o", temperature=0).bind_tools(tools)


def call_llm(state: AgentState) -> AgentState:
    """
    Node function: Invokes the LLM with the current conversation state.

    Process:
        1. Prepends a SystemMessage with instructions
        2. Combines with existing messages from state
        3. Invokes GPT-4o model (which has tools bound to it)
        4. Returns the LLM's response (AIMessage)

    The LLM response can be:
        - AIMessage with tool_calls: LLM wants to use tools
        - AIMessage with content only: LLM has the final answer

    Args:
        state: Current agent state containing message history

    Returns:
        Dict with single key 'messages' containing the AIMessage response.
        The add_messages reducer will APPEND this to existing messages.
    """
    system_message = SystemMessage(
        content="You are a helpful assistant that can perform basic arithmetic operations. You can use the tools below to perform the operations."
    )
    messages = [system_message] + state["messages"]
    response = model.invoke(messages)  # Returns AIMessage
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """
    Conditional edge function: Decides whether to continue the ReAct loop or end.

    Logic:
        1. Get the last message (should be an AIMessage from call_llm)
        2. Check if it has tool_calls
        3. If tool_calls exist → return "continue" → route to tool_node
        4. If no tool_calls → return "end" → route to END

    Args:
        state: Current agent state

    Returns:
        "continue": LLM wants to use tools, route to tool_node
        "end": LLM has final answer, end the graph
    """
    messages = state["messages"]
    last_message = messages[-1]  # Get the most recent message

    # Check if LLM requested any tool calls
    if not last_message.tool_calls:
        return "end"  # No tools needed, we're done
    else:
        return "continue"  # Tools needed, go to tool_node


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================
# Build the ReAct agent graph with nodes and edges
# ============================================================================

graph = StateGraph(
    AgentState
)  # LangGraph reads AgentState and sees add_messages reducer

# Add the LLM node
graph.add_node("llm_process", call_llm)
graph.add_edge(START, "llm_process")  # Entry point: START → llm_process

# Add the tool execution node
tool_node = ToolNode(
    tools=tools
)  # ToolNode executes tools and wraps results in ToolMessage
graph.add_node("tool_node", tool_node)

# Add conditional routing from llm_process
graph.add_conditional_edges(
    "llm_process",  # Source node
    should_continue,  # Decision function
    {
        "continue": "tool_node",  # If tools needed → go to tool_node
        "end": END,  # If done → end the graph
    },
)

# After tools execute, always return to llm_process for the LLM to review results
graph.add_edge("tool_node", "llm_process")

# Compile the graph into a runnable agent
reAct_agent = graph.compile()


def print_stream(stream):
    """
    Helper function: Pretty-prints streaming output from the agent.

    The stream yields state snapshots after each node execution.
    This function extracts and prints the last message from each snapshot.

    Args:
        stream: Generator yielding state dictionaries from reAct_agent.stream()
    """
    for value in stream:
        message = value["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================
# Run the agent with streaming output to see each step
# ============================================================================

user_input = input("Enter your message: ")

user_message = {"messages": [HumanMessage(content=user_input)]}
print_stream(reAct_agent.stream(user_message, stream_mode="values"))
