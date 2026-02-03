"""
Task 2: Supervisor Pattern with Dynamic Routing

Goal: Add a Supervisor agent that intelligently routes tasks to specialized workers.

Architecture:
           ┌─────────────┐
           │ Supervisor  │ ← Decides which agent to call
           └─────────────┘
                 │
       ┌─────────┼─────────┬─────────┐
       ▼         ▼         ▼         ▼
   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
   │Research│ │Analyst │ │ Writer │ │ FINISH │
   └────────┘ └────────┘ └────────┘ └────────┘
        │         │         │
        └─────────┴─────────┘
                 │
           (Return to Supervisor)

What you'll learn:
- Implementing a supervisor agent with LLM-based routing
- Using conditional edges for dynamic workflow
- Managing agent coordination through shared state
- Handling workflow completion logic

Estimated time: 40-50 minutes
"""

from typing import TypedDict, Annotated, Sequence, Optional, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# ==============================================================================
# SHARED STATE WITH ROUTING
# ==============================================================================


class MultiAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    research_data: Optional[str]
    analysis_report: Optional[str]
    final_report: Optional[str]
    next_agent: str  # Routing field


# ==============================================================================
# SUPERVISOR AGENT
# ==============================================================================


def supervisor_agent(state: MultiAgentState) -> MultiAgentState:
    """Supervisor decides which agent to call next."""
    print("\n🎯 SUPERVISOR: Analyzing workflow state...")

    # Check what's been completed
    has_research = "YES" if state.get("research_data") else "NO"
    has_analysis = "YES" if state.get("analysis_report") else "NO"
    has_report = "YES" if state.get("final_report") else "NO"

    # Create supervisor prompt with current state
    supervisor_prompt = """
    You are a Supervisor coordinating a research team with these agents:
    - researcher: Gathers information from sources
    - analyst: Analyzes research data and extracts insights
    - writer: Writes comprehensive reports
    - FINISH: Complete the task

    Current state:
    - Research data: {has_research}
    - Analysis: {has_analysis}
    - Report: {has_report}

    Which agent should act next? Respond with ONLY the agent name (researcher/analyst/writer/FINISH).
    """

    # Call LLM to decide next agent
    response = llm.invoke(
        [
            SystemMessage(
                content=supervisor_prompt.format(
                    has_research=has_research,
                    has_analysis=has_analysis,
                    has_report=has_report,
                )
            )
        ]
    )

    # Extract agent name from response and clean it
    next_agent = response.content.strip().lower()
    print(f"   → Decision: Route to '{next_agent}'")

    return {"next_agent": next_agent}


# ==============================================================================
# WORKER AGENTS
# ==============================================================================


def researcher_agent(state: MultiAgentState) -> MultiAgentState:
    """Simulates web research by using LLM to generate findings."""
    print("\n🔍 RESEARCHER AGENT: Gathering information...")

    user_query = state["messages"][-1].content

    research_prompt = """
    You are a Research Agent. Your job is to gather information about the user's query.
    Simulate finding 3-5 key facts or data points about the topic.

    User query: {query}

    Provide your research findings in a clear, factual format.
    """

    response = llm.invoke(
        [SystemMessage(content=research_prompt.format(query=user_query))]
    )

    return {"research_data": response.content}


def analyst_agent(state: MultiAgentState) -> MultiAgentState:
    """Analyzes research data and extracts insights."""
    print("\n📊 ANALYST AGENT: Analyzing data...")

    research_data = state["research_data"]

    analysis_prompt = """
    You are an Analysis Agent. Your job is to analyze research data and extract key insights.

    Research data:
    {research_data}

    Provide 3-5 key insights or takeaways from this research.
    """

    response = llm.invoke(
        [SystemMessage(content=analysis_prompt.format(research_data=research_data))]
    )

    return {"analysis_report": response.content}


def writer_agent(state: MultiAgentState) -> MultiAgentState:
    """Writes final report based on research and analysis."""
    print("\n✍️  WRITER AGENT: Composing report...")

    research_data = state["research_data"]
    analysis = state["analysis_report"]

    writing_prompt = """
     You are a Writing Agent. Your job is to write a comprehensive report.

    Research findings:
    {research_data}

    Analysis insights:
    {analysis}

    Write a well-structured report (3-4 paragraphs) that synthesizes this information.
    """

    response = llm.invoke(
        [
            SystemMessage(
                content=writing_prompt.format(
                    research_data=research_data, analysis=analysis
                )
            )
        ]
    )

    return {"final_report": response.content}


# ==============================================================================
# ROUTING FUNCTION
# ==============================================================================


def route_to_agent(
    state: MultiAgentState,
) -> Literal["researcher", "analyst", "writer", "__end__"]:
    """Routes to the agent specified by supervisor."""

    next_agent = state["next_agent"]

    # Map agent names to node names
    switch = {
        "researcher": "researcher",
        "analyst": "analyst",
        "writer": "writer",
        "FINISH": END,
    }

    return switch.get(next_agent, END)


# ==============================================================================
# BUILD THE SUPERVISOR GRAPH
# ==============================================================================


def create_supervisor_team():
    """Creates the supervisor-based multi-agent graph."""

    # Create StateGraph
    graph = StateGraph(MultiAgentState)

    # Add supervisor node
    graph.add_node("supervisor", supervisor_agent)

    # Add worker nodes
    graph.add_node("researcher", researcher_agent)
    graph.add_node("analyst", analyst_agent)
    graph.add_node("writer", writer_agent)

    # Add edge from START to supervisor
    graph.add_edge(START, "supervisor")

    # Add conditional edges from supervisor to workers
    graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "__end__": END,
        },
    )

    # Add edges from workers back to supervisor
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("analyst", "supervisor")
    graph.add_edge("writer", "supervisor")

    return graph.compile()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================


def main():
    """Test the supervisor-based multi-agent system."""

    # Create the graph
    supervisor_team = create_supervisor_team()

    # Test query
    user_query = "What are the main challenges in deploying AI agents to production?"

    print("=" * 80)
    print("🎯 SUPERVISOR-BASED MULTI-AGENT SYSTEM")
    print("=" * 80)
    print(f"\n❓ User Query: {user_query}\n")

    # Run the graph
    initial_state = {"messages": [HumanMessage(content=user_query)]}

    final_state = supervisor_team.invoke(initial_state)

    # Display results
    print("\n" + "=" * 80)
    print("📋 FINAL REPORT")
    print("=" * 80)
    print(final_state.get("final_report", "No report generated"))
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
