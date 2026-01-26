"""
Exercise 8 - Task 1: Basic Customer Support Agent with MemorySaver

Your Challenge:
Build a customer support agent that:
1. Collects user information (name, email, issue description)
2. Remembers context across conversation turns
3. Uses MemorySaver for in-memory checkpointing
4. Maintains conversation history

Learning Goals:
- Understand how checkpointing works
- See state persistence in action
- Learn thread_id usage for conversation isolation
"""

from typing import TypedDict, Annotated, Sequence, Optional
from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    RemoveMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv(override=True)

# ==============================================================================
# 1. DEFINE STATE
# ==============================================================================


class AgentState(TypedDict):
    """
    TODO: Define the state that will be checkpointed.

    Think about what information a customer support agent needs to remember:
    - Conversation messages
    - User's name (once collected)
    - User's email (once collected)
    - Issue description (once collected)
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    # fields for user_name, user_email, issue_description
    user_name: Optional[str] = None
    user_email: Optional[str] = None
    issue_description: Optional[str] = None


# ==============================================================================
# 2. BUILD THE GRAPH
# ==============================================================================


def call_llm(state: AgentState) -> AgentState:
    """
    Main agent node that processes user messages.

    TODO:
    1. Create a system prompt that instructs the agent to:
       - Greet the user warmly
       - Collect their name, email, and issue description
       - Remember information already collected (check state)
       - Provide helpful responses

    2. Include current state context in the prompt so the LLM knows what's been collected

    3. Call the LLM and return the response
    """

    # TODO: Build system prompt with context from state
    # Hint: Check if user_name, user_email, issue_description exist in state

    system_prompt = """
    You are a helpful customer support agent.
    
    Your job:
    1. Greet the user warmly
    2. Collect their information:
       - Full name
       - Email address
       - Description of their issue
    3. Once you have all information, acknowledge it and ask how you can help
    
    IMPORTANT: Remember what information you've already collected!
    """

    # TODO: Add context about what's already been collected
    # Example: "User's name: {state.get('user_name', 'Not collected yet')}"
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])

    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    response = llm.invoke(messages)

    return {"messages": [response]}


# ==============================================================================
# 3. COMPILE GRAPH WITH CHECKPOINTING
# ==============================================================================


def create_agent():
    """
    TODO: Create and compile the graph with MemorySaver checkpointer.

    Steps:
    1. Create StateGraph with AgentState
    2. Add the call_llm node
    3. Add edges (START -> call_llm -> END)
    4. Create MemorySaver instance
    5. Compile with checkpointer
    """

    # Create graph
    graph = StateGraph(AgentState)

    # Add nodes and edges
    graph.add_node("llm_node", call_llm)
    graph.add_node("prune_node", prune_messages)
    graph.add_edge(START, "llm_node")
    graph.add_edge("llm_node", "prune_node")
    graph.add_edge("prune_node", END)

    # Create checkpointer
    # checkpointer = MemorySaver()
    conn_sqlite = sqlite3.connect("checkpoints.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn_sqlite)

    compiled_graph = graph.compile(checkpointer=checkpointer)

    # Compile with checkpointer and return
    return compiled_graph, checkpointer


def prune_messages(state: AgentState) -> AgentState:
    """
    Keep only the last 10 messages to prevent state bloat.
    Uses RemoveMessage to work with add_messages reducer.
    """
    messages = state["messages"]

    if len(messages) <= 10:
        return {}  ## No pruning needed

    messages_to_remove = []
    messages_to_keep = []

    # Check if first message is SystemMessage (not last!)
    first_message = messages[0]

    if isinstance(first_message, SystemMessage):
        ## Preserving the system message along with the last 9 messages
        messages_to_keep.append(first_message)
        messages_to_keep.extend(messages[-9:])  # ✅ Use extend, not append
    else:
        ## Preserving the last 10 messages
        messages_to_keep.extend(messages[-10:])  # ✅ Use extend, not append

    ## Getting the ids of the messages to keep
    kept_ids = {msg.id for msg in messages_to_keep}

    ## Getting the ids of the messages to remove
    for msg in messages:
        if msg.id not in kept_ids:
            messages_to_remove.append(RemoveMessage(id=msg.id))

    return {"messages": messages_to_remove}


# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================


def main():
    """
    Test the agent with a multi-turn conversation.

    TODO:
    1. Create the agent
    2. Set up a thread_id for the conversation
    3. Have a conversation where you:
       - Introduce yourself
       - Provide your email
       - Describe an issue
       - Ask a follow-up question
    4. Observe how the agent remembers context
    """

    agent, checkpointer = create_agent()

    thread_id = "user-test-112"

    # TODO: Create config with thread_id
    # Hint: config = {"configurable": {"thread_id": "user-test-1"}}
    config = {"configurable": {"thread_id": thread_id}}

    print("=" * 80)
    print("🤖 CUSTOMER SUPPORT AGENT (with 🔄 RESUME FUNCTIONALITY DEMO)")
    print("=" * 80)
    print(f"📌 Thread ID: {thread_id}")
    print("💡 This conversation will persist across restarts!")
    print("💡 Type 'crash' to simulate a crash and restart the script")
    print("💡 Type 'quit' to exit\n")

    checkpoints = list(checkpointer.list(config=config))

    if checkpoints:
        print(f"\nFound {len(checkpoints)} checkpoints for this thread.")
        last_state = checkpoints[0].checkpoint
        if last_state.get("channel_values", {}).get("messages"):
            messages = last_state["channel_values"]["messages"]
            if messages:
                print(f"💬 Last message: {messages[-1].content[:50]}...")
        print("🔄 Resuming conversation...\n")
    else:
        print(
            "\nNo checkpoints found for this thread. Starting a new conversation...\n"
        )

    while True:
        user_input = input("\n👤 You: ")

        if user_input.lower() == "crash":
            print("\n💥 SIMULATING CRASH... Restart the script to resume!\n")
            break

        if user_input.lower() in ["quit", "exit"]:
            print("\n👋 Goodbye!\n")
            break

        # TODO: Invoke agent with message and config
        # response = agent.invoke(
        #     {"messages": [HumanMessage(content=user_input)]},
        #     config=config
        # )
        response = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]}, config=config
        )

        # TODO: Print the agent's response
        # Hint: Get the last message from response["messages"]

        print(f"\n🤖 Agent: \n {response['messages'][-1].content}")


if __name__ == "__main__":
    main()
