"""
Comprehensive Test Suite for Python Async RAG Agent
Tests both positive and negative scenarios
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from exercise_6_code_docs_rag import app, retriver_tool
from langchain_core.messages import HumanMessage


def print_section(title):
    """Helper to print section headers"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_positive_scenarios():
    """Test positive scenarios - valid queries that should work"""
    print_section("POSITIVE SCENARIO TESTS")

    test_cases = [
        {
            "name": "Test 1: Basic async/await question",
            "query": "What is async/await in Python?",
            "expected": "Should retrieve relevant information about async/await",
        },
        {
            "name": "Test 2: Specific concept - asyncio",
            "query": "How does asyncio work?",
            "expected": "Should retrieve information about asyncio module",
        },
        {
            "name": "Test 3: Practical example request",
            "query": "Can you show me an example of using async functions?",
            "expected": "Should provide examples from the documentation",
        },
        {
            "name": "Test 4: Comparison question",
            "query": "What's the difference between async and sync programming?",
            "expected": "Should retrieve comparative information",
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─' * 80}")
        print(f"🧪 {test['name']}")
        print(f"Query: {test['query']}")
        print(f"Expected: {test['expected']}")
        print(f"{'─' * 80}")

        try:
            response = app.invoke({"messages": [HumanMessage(content=test["query"])]})
            final_response = response["messages"][-1].content

            print(f"\n✅ SUCCESS - Response received:")
            print(f"Response length: {len(final_response)} characters")
            print(f"First 300 chars: {final_response[:300]}...")

            # Check if response is meaningful
            if len(final_response) > 50:
                print("✓ Response has substantial content")
            else:
                print("⚠️  Warning: Response seems short")

        except Exception as e:
            print(f"\n❌ FAILED - Error: {str(e)}")
            print(f"Error type: {type(e).__name__}")


def test_negative_scenarios():
    """Test negative scenarios - edge cases and error handling"""
    print_section("NEGATIVE SCENARIO TESTS")

    test_cases = [
        {
            "name": "Test 1: Completely unrelated query",
            "query": "What is the capital of France?",
            "expected": "Should indicate no relevant information found or gracefully handle",
        },
        {
            "name": "Test 2: Empty/vague query",
            "query": "Tell me something",
            "expected": "Should handle vague queries appropriately",
        },
        {
            "name": "Test 3: Query about non-existent concept",
            "query": "How do I use the quantum async feature in Python?",
            "expected": "Should indicate information not found in documentation",
        },
        {
            "name": "Test 4: Very specific technical detail",
            "query": "What is the exact memory footprint of asyncio event loop?",
            "expected": "Should retrieve what's available or indicate limitations",
        },
        {
            "name": "Test 5: Malformed/nonsensical query",
            "query": "async await def function class import???",
            "expected": "Should handle gracefully without crashing",
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─' * 80}")
        print(f"🧪 {test['name']}")
        print(f"Query: {test['query']}")
        print(f"Expected: {test['expected']}")
        print(f"{'─' * 80}")

        try:
            response = app.invoke({"messages": [HumanMessage(content=test["query"])]})
            final_response = response["messages"][-1].content

            print(f"\n✅ HANDLED - Response received:")
            print(f"Response length: {len(final_response)} characters")
            print(f"First 300 chars: {final_response[:300]}...")

            # Check for appropriate handling
            if any(
                phrase in final_response.lower()
                for phrase in [
                    "no relevant",
                    "not found",
                    "don't have",
                    "cannot find",
                    "not available",
                ]
            ):
                print("✓ Appropriately indicated information not available")
            else:
                print("✓ Provided response based on available information")

        except Exception as e:
            print(f"\n❌ FAILED - Error: {str(e)}")
            print(f"Error type: {type(e).__name__}")


def test_retriever_tool_directly():
    """Test the retriever tool directly"""
    print_section("DIRECT RETRIEVER TOOL TESTS")

    test_queries = [
        "async/await",
        "event loop",
        "coroutines",
        "completely unrelated topic like cooking",
    ]

    for query in test_queries:
        print(f"\n{'─' * 80}")
        print(f"🔍 Testing retriever with: '{query}'")
        print(f"{'─' * 80}")

        try:
            result = retriver_tool.invoke({"query": query})
            print(f"\n✅ Retriever returned results:")
            print(f"Result length: {len(result)} characters")
            print(f"First 200 chars: {result[:200]}...")

        except Exception as e:
            print(f"\n❌ Retriever failed: {str(e)}")


def test_graph_flow():
    """Test the graph execution flow"""
    print_section("GRAPH FLOW TEST")

    print("Testing if the graph properly routes through LLM -> Retriever -> LLM...")

    try:
        query = "Explain async/await in Python"
        print(f"\nQuery: {query}")

        # Use stream to see the flow
        print("\n🔄 Streaming graph execution:")
        for step in app.stream({"messages": [HumanMessage(content=query)]}):
            for key, value in step.items():
                print(f"\n📍 Step: {key}")
                if "messages" in value:
                    msg = value["messages"][-1]
                    msg_type = type(msg).__name__
                    print(f"   Message type: {msg_type}")
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        print(f"   Tool calls: {len(msg.tool_calls)}")

        print("\n✅ Graph flow completed successfully")

    except Exception as e:
        print(f"\n❌ Graph flow failed: {str(e)}")


def run_all_tests():
    """Run all test suites"""
    print("\n" + "🚀" * 40)
    print("  PYTHON ASYNC RAG AGENT - COMPREHENSIVE TEST SUITE")
    print("🚀" * 40)

    try:
        # Test 1: Direct retriever tool
        test_retriever_tool_directly()

        # Test 2: Positive scenarios
        test_positive_scenarios()

        # Test 3: Negative scenarios
        test_negative_scenarios()

        # Test 4: Graph flow
        test_graph_flow()

        print_section("TEST SUMMARY")
        print("✅ All test suites completed!")
        print(
            "\nNote: Review the output above to verify each test behaved as expected."
        )
        print("Look for:")
        print("  - Successful responses for positive scenarios")
        print("  - Graceful handling of negative scenarios")
        print("  - Proper graph flow execution")
        print("  - Appropriate similarity scores in retriever output")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
