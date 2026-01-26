"""
Test script to verify message pruning works correctly
"""

from checkpointer_agent import create_agent
from langchain_core.messages import HumanMessage
import uuid


def test_pruning():
    """Send 15 messages and verify only 10 are kept"""

    agent, checkpointer = create_agent()

    # Use unique thread for this test
    thread_id = f"pruning-test-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    print("=" * 80)
    print("🧪 MESSAGE PRUNING TEST")
    print("=" * 80)
    print(f"📌 Thread ID: {thread_id}")
    print("📝 Sending 15 messages to test pruning...\n")

    # Send 15 messages
    for i in range(1, 16):
        message = f"Test message {i}"
        print(f"Sending message {i}/15: {message}")

        response = agent.invoke(
            {"messages": [HumanMessage(content=message)]}, config=config
        )

        # Check message count after each invocation
        message_count = len(response["messages"])
        print(f"  → Total messages in state: {message_count}")

        # After 10 messages, count should stay at 10
        if i > 10:
            if message_count <= 10:
                print(f"  ✅ Pruning working! (kept at {message_count})")
            else:
                print(f"  ❌ Pruning failed! (has {message_count} messages)")

        print()

    # Final verification
    print("=" * 80)
    print("📊 FINAL RESULTS")
    print("=" * 80)

    final_state = agent.get_state(config)
    final_message_count = len(final_state.values["messages"])

    print(f"Total messages sent: 15")
    print(f"Messages in final state: {final_message_count}")

    if final_message_count == 10:
        print("\n✅ TEST PASSED: Message pruning works correctly!")
    else:
        print(f"\n❌ TEST FAILED: Expected 10 messages, got {final_message_count}")

    # Show which messages were kept
    print("\n📋 Messages kept in state:")
    for idx, msg in enumerate(final_state.values["messages"], 1):
        content_preview = msg.content[:50] if hasattr(msg, "content") else str(msg)[:50]
        print(f"  {idx}. {msg.__class__.__name__}: {content_preview}...")


if __name__ == "__main__":
    test_pruning()
