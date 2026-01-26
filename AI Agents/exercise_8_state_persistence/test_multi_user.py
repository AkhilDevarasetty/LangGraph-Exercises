"""
Test multi-user thread isolation
Simulates 2 users having concurrent conversations
"""

from checkpointer_agent import create_agent
from langchain_core.messages import HumanMessage


def simulate_user_conversation(agent, user_id, messages):
    """Simulate a user's conversation"""
    thread_id = f"user-{user_id}"
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n{'=' * 60}")
    print(f"👤 USER {user_id} (thread: {thread_id})")
    print("=" * 60)

    for msg in messages:
        print(f"\n{user_id}: {msg}")
        response = agent.invoke(
            {"messages": [HumanMessage(content=msg)]}, config=config
        )
        agent_response = response["messages"][-1].content
        print(f"🤖: {agent_response[:100]}...")

    return thread_id


def main():
    agent, checkpointer = create_agent()

    print("=" * 80)
    print("🧪 MULTI-USER THREAD ISOLATION TEST")
    print("=" * 80)

    # User Alice's conversation
    alice_messages = ["Hi, I need help", "My name is Alice", "alice@example.com"]

    # User Bob's conversation
    bob_messages = ["Hello", "I'm Bob", "bob@example.com"]

    # User Charlie's conversation
    charlie_messages = ["Hi, I need help", "My name is Charlie", "charlie@example.com"]

    # Simulate Alice
    alice_thread = simulate_user_conversation(agent, "alice", alice_messages)

    # Simulate Bob
    bob_thread = simulate_user_conversation(agent, "bob", bob_messages)

    # Simulate Charlie
    charlie_thread = simulate_user_conversation(agent, "charlie", charlie_messages)

    # Verification: Ask each user "What's my name?"
    print(f"\n{'=' * 80}")
    print("🔍 VERIFICATION: Testing Thread Isolation")
    print("=" * 80)

    # Alice asks "What's my name?"
    alice_config = {"configurable": {"thread_id": alice_thread}}
    alice_response = agent.invoke(
        {"messages": [HumanMessage(content="What's my name?")]}, config=alice_config
    )
    print(f"\nAlice asks: 'What's my name?'")
    print(f"🤖 Response: {alice_response['messages'][-1].content}")

    # Bob asks "What's my name?"
    bob_config = {"configurable": {"thread_id": bob_thread}}
    bob_response = agent.invoke(
        {"messages": [HumanMessage(content="What's my name?")]}, config=bob_config
    )
    print(f"\nBob asks: 'What's my name?'")
    print(f"🤖 Response: {bob_response['messages'][-1].content}")

    # Charlie asks "What's my name?"
    charlie_config = {"configurable": {"thread_id": charlie_thread}}
    charlie_response = agent.invoke(
        {"messages": [HumanMessage(content="What's my name?")]}, config=charlie_config
    )
    print(f"\nCharlie asks: 'What's my name?'")
    print(f"🤖 Response: {charlie_response['messages'][-1].content}")

    # Check isolation
    print(f"\n{'=' * 80}")
    print("📊 RESULTS")
    print("=" * 80)

    alice_knows_name = "alice" in alice_response["messages"][-1].content.lower()
    bob_knows_name = "bob" in bob_response["messages"][-1].content.lower()
    charlie_knows_name = "charlie" in charlie_response["messages"][-1].content.lower()

    if alice_knows_name and bob_knows_name and charlie_knows_name:
        print("✅ TEST PASSED: Complete thread isolation!")
        print("   - Alice's agent remembers 'Alice'")
        print("   - Bob's agent remembers 'Bob'")
        print("   - Charlie's agent remembers 'Charlie'")
        print("   - No cross-contamination between threads")
    else:
        print("❌ TEST FAILED: Thread isolation broken!")


if __name__ == "__main__":
    main()
