# Exercise 11: Multi-Agent Systems & Collaboration

**Objective:** Build production-grade multi-agent systems where specialized agents collaborate to solve complex problems through subgraphs, supervisor patterns, and dynamic routing.

**Priority:** 🔴 CRITICAL  
**Estimated Time:** 6-8 hours  
**Difficulty:** ⭐⭐⭐⭐

---

## 📚 Theory & Concepts

### **What are Multi-Agent Systems?**

A multi-agent system is an architecture where multiple specialized AI agents work together to accomplish tasks that are too complex for a single agent. Each agent has a specific role and expertise.

**Think of it like this:**  
Imagine you're building a house. You don't hire one person to do everything. Instead, you have:

- **Architect** - Designs the structure
- **Electrician** - Handles wiring
- **Plumber** - Installs pipes
- **Carpenter** - Builds frames
- **Project Manager** - Coordinates everyone

Each specialist focuses on what they do best. The same principle applies to AI agents. Instead of one "super agent" that tries to do everything (and does nothing well), you build specialized agents that excel at specific tasks.

### **Why Multi-Agent Systems Matter in Production**

In real-world applications, multi-agent systems enable five critical capabilities:

#### **1. Specialization & Expertise**

Each agent can be optimized for its specific domain with tailored prompts, tools, and models.

**Real-world scenario:**  
You're building a customer support system. Instead of one agent handling everything:

- **Triage Agent** (fast, cheap model) - Routes to the right department
- **Technical Support Agent** (code-aware model) - Debugs technical issues
- **Billing Agent** (secure, audit-logged) - Handles payment questions
- **Escalation Agent** (human-in-loop) - Manages complex cases

Each agent uses different tools, different prompts, and even different LLMs optimized for their task.

#### **2. Parallel Execution**

Multiple agents can work simultaneously, dramatically reducing latency.

**Real-world scenario:**  
A research agent needs to analyze a company. Instead of sequential processing:

**Sequential (slow):**

```
Research financials (30s) → Research competitors (30s) → Research news (30s)
Total: 90 seconds
```

**Parallel (fast):**

```
Research financials (30s) ┐
Research competitors (30s) ├─→ Combine results
Research news (30s)        ┘
Total: 30 seconds (3× faster!)
```

#### **3. Modularity & Reusability**

Agents can be developed, tested, and deployed independently. You can reuse the same agent across different workflows.

**Real-world scenario:**  
You build a "Web Search Agent" for your research system. Later, you need web search in your customer support system. Instead of rebuilding it, you just plug in the same agent.

```python
# Reuse the same agent in different workflows
research_workflow.add_node("search", web_search_agent)
support_workflow.add_node("search", web_search_agent)
```

#### **4. Fault Isolation**

If one agent fails, others can continue working. The system degrades gracefully instead of crashing completely.

**Real-world scenario:**  
Your research system has three agents: Web Search, Database Query, and Report Writer. The Web Search API goes down. Without multi-agent architecture, the entire system crashes. With multi-agent architecture:

```
Web Search Agent → ❌ FAILED (API down)
Database Query Agent → ✅ SUCCESS (still works)
Report Writer Agent → ✅ SUCCESS (writes report with available data)

Result: Partial report delivered instead of total failure
```

#### **5. Iterative Refinement**

Agents can critique and improve each other's work through feedback loops.

**Real-world scenario:**  
A content generation system:

1. **Writer Agent** - Drafts an article
2. **Critic Agent** - Reviews it: "Too technical, needs examples"
3. **Writer Agent** - Revises based on feedback
4. **Critic Agent** - Reviews again: "Much better, approved!"

This produces higher-quality output than a single-pass generation.

---

### **How Multi-Agent Systems Work in LangGraph**

LangGraph implements multi-agent systems through three core primitives:

#### **1. Subgraphs (Agent Encapsulation)**

Each agent is its own **subgraph** - a complete LangGraph workflow that can be embedded in a larger graph.

```python
# Define a specialized agent as a subgraph
def create_research_agent():
    builder = StateGraph(ResearchState)
    builder.add_node("search", search_web)
    builder.add_node("summarize", summarize_results)
    builder.add_edge("search", "summarize")
    builder.add_edge("summarize", END)
    return builder.compile()

# Embed it in the main graph
main_graph.add_node("researcher", create_research_agent())
```

**Why subgraphs?**

- **Encapsulation:** Agent's internal logic is hidden
- **Testability:** Test each agent independently
- **Reusability:** Use the same agent in multiple workflows
- **Observability:** See which step in which agent failed

#### **2. Supervisor Pattern (Coordination)**

A **supervisor** agent coordinates multiple worker agents, deciding which agent to call and when.

```
        ┌─────────────┐
        │ Supervisor  │ ← Decides which agent to call
        └─────────────┘
              │
    ┌─────────┼─────────┬─────────┐
    ▼         ▼         ▼         ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Research│ │Analyst │ │ Writer │ │ Critic │
└────────┘ └────────┘ └────────┘ └────────┘
```

The supervisor:

- Receives the user's request
- Analyzes what needs to be done
- Routes to the appropriate agent
- Collects results
- Decides next steps (continue, loop, or finish)

#### **3. Shared State (Communication)**

All agents communicate through a **shared state** object that flows through the graph.

```python
class MultiAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    research_data: Optional[str]  # Written by Researcher
    analysis: Optional[str]        # Written by Analyst
    draft: Optional[str]           # Written by Writer
    feedback: Optional[str]        # Written by Critic
    next_agent: str                # Set by Supervisor
```

Each agent:

- **Reads** from shared state (sees what previous agents did)
- **Writes** to shared state (shares its output with future agents)
- **Updates** control flow (signals which agent should run next)

---

### **Multi-Agent Architecture Patterns**

There are four common patterns for organizing multi-agent systems:

#### **Pattern 1: Sequential Pipeline**

Agents execute in a fixed order, each building on the previous agent's work.

```
User Query → Agent A → Agent B → Agent C → Final Answer
```

**Example:** Research pipeline

```
User: "Analyze Tesla's Q4 earnings"
  ↓
Researcher → Finds earnings report
  ↓
Analyst → Extracts key metrics
  ↓
Writer → Generates summary
  ↓
Response: "Tesla's Q4 revenue was $25.2B..."
```

**Pros:**

- ✅ Simple to understand and debug
- ✅ Predictable execution flow
- ✅ Easy to optimize (know exactly what each agent does)

**Cons:**

- ❌ Slow (agents run sequentially)
- ❌ Inflexible (can't skip steps)
- ❌ Brittle (one failure stops everything)

**When to use:** Simple workflows with clear, linear steps.

---

#### **Pattern 2: Supervisor with Dynamic Routing**

A supervisor agent decides which worker agent to call based on the current state.

```
        ┌─────────────┐
        │ Supervisor  │
        └─────────────┘
              │
         (decides)
              │
    ┌─────────┼─────────┬─────────┐
    ▼         ▼         ▼         ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Agent A │ │Agent B │ │Agent C │ │Agent D │
└────────┘ └────────┘ └────────┘ └────────┘
```

**Example:** Customer support routing

```
User: "My payment failed"
  ↓
Supervisor: "This is a billing issue" → Routes to Billing Agent
  ↓
Billing Agent: Checks payment status, provides solution
  ↓
Supervisor: "Issue resolved" → END
```

**Pros:**

- ✅ Flexible (adapts to different requests)
- ✅ Efficient (only calls necessary agents)
- ✅ Scalable (easy to add new agents)

**Cons:**

- ❌ Complex routing logic
- ❌ Supervisor can make wrong decisions
- ❌ Harder to debug (non-deterministic flow)

**When to use:** Complex workflows where the path depends on the input.

---

#### **Pattern 3: Hierarchical (Nested Supervisors)**

Supervisors manage other supervisors, creating a hierarchy.

```
        ┌──────────────────┐
        │ Main Supervisor  │
        └──────────────────┘
              │
    ┌─────────┴─────────┐
    ▼                   ▼
┌─────────────┐   ┌─────────────┐
│ Research    │   │ Writing     │
│ Supervisor  │   │ Supervisor  │
└─────────────┘   └─────────────┘
    │                   │
┌───┴───┐           ┌───┴───┐
▼       ▼           ▼       ▼
Web   Database    Draft   Edit
Search  Query     Writer  Reviewer
```

**Example:** Enterprise content generation

```
Main Supervisor: "Create a market analysis report"
  ↓
Research Supervisor: Coordinates data gathering
  ├─ Web Search Agent
  ├─ Database Query Agent
  └─ API Integration Agent
  ↓
Writing Supervisor: Coordinates content creation
  ├─ Draft Writer Agent
  ├─ Fact Checker Agent
  └─ Editor Agent
  ↓
Final Report
```

**Pros:**

- ✅ Handles very complex workflows
- ✅ Clear separation of concerns
- ✅ Each supervisor optimizes its domain

**Cons:**

- ❌ High complexity
- ❌ More LLM calls (each supervisor is an LLM)
- ❌ Difficult to debug (many layers)

**When to use:** Enterprise systems with complex, multi-stage workflows.

---

#### **Pattern 4: Collaborative (Peer-to-Peer)**

Agents communicate directly with each other without a central supervisor.

```
┌────────┐ ←→ ┌────────┐
│Agent A │    │Agent B │
└────────┘    └────────┘
    ↕             ↕
┌────────┐ ←→ ┌────────┐
│Agent C │    │Agent D │
└────────┘    └────────┘
```

**Example:** Debate system

```
Proposition Agent: "Climate change requires immediate action"
  ↓
Opposition Agent: "Economic impacts must be considered"
  ↓
Proposition Agent: "Long-term costs outweigh short-term pain"
  ↓
Moderator Agent: "Proposition wins based on evidence"
```

**Pros:**

- ✅ Emergent behavior (agents discover solutions)
- ✅ No single point of failure
- ✅ Highly flexible

**Cons:**

- ❌ Can loop infinitely
- ❌ Unpredictable behavior
- ❌ Very hard to debug

**When to use:** Research, creative tasks, or when you want emergent solutions.

---

### **Subgraphs: Building Reusable Agents**

A subgraph is a complete LangGraph workflow that can be embedded as a single node in a larger graph.

#### **Why Subgraphs?**

**Without subgraphs (messy):**

```python
# All logic in one giant graph
main_graph.add_node("search_web", search_web)
main_graph.add_node("filter_results", filter_results)
main_graph.add_node("summarize", summarize)
main_graph.add_node("extract_data", extract_data)
main_graph.add_node("rank_data", rank_data)
main_graph.add_node("draft_report", draft_report)
# ... 20 more nodes
```

**With subgraphs (clean):**

```python
# Each agent is a self-contained subgraph
main_graph.add_node("researcher", research_agent)
main_graph.add_node("analyst", analysis_agent)
main_graph.add_node("writer", writing_agent)
```

#### **Creating a Subgraph**

```python
from langgraph.graph import StateGraph, END

# Define the subgraph's state
class ResearchState(TypedDict):
    query: str
    results: List[str]
    summary: str

# Build the subgraph
def create_research_agent():
    builder = StateGraph(ResearchState)

    builder.add_node("search", search_web_tool)
    builder.add_node("filter", filter_relevant_results)
    builder.add_node("summarize", create_summary)

    builder.add_edge(START, "search")
    builder.add_edge("search", "filter")
    builder.add_edge("filter", "summarize")
    builder.add_edge("summarize", END)

    return builder.compile()

# Use it in the main graph
research_agent = create_research_agent()
main_graph.add_node("researcher", research_agent)
```

#### **State Mapping Between Graphs**

The main graph and subgraph often have different state schemas. You need to map between them:

```python
# Main graph state
class MainState(TypedDict):
    user_query: str
    research_summary: str
    final_report: str

# Subgraph state
class ResearchState(TypedDict):
    query: str
    summary: str

# Map main state → subgraph state
def map_to_research(main_state: MainState) -> ResearchState:
    return {
        "query": main_state["user_query"]
    }

# Map subgraph state → main state
def map_from_research(research_state: ResearchState, main_state: MainState) -> MainState:
    return {
        **main_state,
        "research_summary": research_state["summary"]
    }

# Add subgraph with state mapping
main_graph.add_node(
    "researcher",
    research_agent,
    input=map_to_research,
    output=map_from_research
)
```

---

### **The Supervisor Pattern: Coordinating Agents**

The supervisor pattern is the most common multi-agent architecture. A supervisor agent acts as a coordinator, deciding which worker agent to call.

#### **How It Works**

```python
def supervisor_node(state: MultiAgentState):
    """Supervisor decides which agent to call next."""

    # Analyze current state
    messages = state["messages"]
    last_message = messages[-1].content

    # Use LLM to decide next agent
    prompt = f"""
    You are a supervisor coordinating a team of agents:
    - researcher: Searches the web for information
    - analyst: Analyzes data and extracts insights
    - writer: Writes reports and summaries
    - FINISH: Complete the task

    Current state: {last_message}

    Which agent should act next? Respond with just the agent name.
    """

    response = llm.invoke(prompt)
    next_agent = response.content.strip().lower()

    return {"next_agent": next_agent}

# Routing function
def route_to_agent(state: MultiAgentState) -> str:
    """Route to the agent specified by supervisor."""
    next_agent = state["next_agent"]

    if next_agent == "finish":
        return END
    else:
        return next_agent

# Build the graph
builder = StateGraph(MultiAgentState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", research_agent)
builder.add_node("analyst", analysis_agent)
builder.add_node("writer", writing_agent)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {
        "researcher": "researcher",
        "analyst": "analyst",
        "writer": "writer",
        END: END
    }
)

# All agents return to supervisor
builder.add_edge("researcher", "supervisor")
builder.add_edge("analyst", "supervisor")
builder.add_edge("writer", "supervisor")
```

#### **Supervisor Decision-Making**

The supervisor can use different strategies to decide which agent to call:

**Strategy 1: LLM-based (flexible but expensive)**

```python
# Supervisor uses an LLM to analyze the situation
response = llm.invoke(f"Which agent should handle: {user_query}")
```

**Strategy 2: Rule-based (fast but rigid)**

```python
# Supervisor uses if/else logic
if "search" in user_query.lower():
    return "researcher"
elif "analyze" in user_query.lower():
    return "analyst"
```

**Strategy 3: Hybrid (best of both)**

```python
# Use rules for common cases, LLM for edge cases
if is_simple_query(user_query):
    return apply_rules(user_query)  # Fast path
else:
    return llm_decide(user_query)   # Flexible path
```

---

### **Preventing Infinite Loops**

Multi-agent systems can loop forever if not designed carefully.

#### **Problem: Infinite Revision Loop**

```
Writer: "Here's a draft"
  ↓
Critic: "Needs improvement"
  ↓
Writer: "Here's a revision"
  ↓
Critic: "Still needs improvement"
  ↓
Writer: "Here's another revision"
  ↓
... (loops forever)
```

#### **Solution 1: Maximum Iterations**

```python
class MultiAgentState(TypedDict):
    draft: str
    revision_count: int  # Track iterations

def writer_agent(state):
    if state["revision_count"] >= 3:
        # Force completion after 3 revisions
        return {"next_agent": "FINISH"}

    # Otherwise, continue revising
    return {
        "draft": revised_draft,
        "revision_count": state["revision_count"] + 1,
        "next_agent": "critic"
    }
```

#### **Solution 2: Quality Threshold**

```python
def critic_agent(state):
    quality_score = evaluate_quality(state["draft"])

    if quality_score >= 0.8:  # Good enough
        return {"next_agent": "FINISH"}
    else:
        return {
            "feedback": "Needs improvement...",
            "next_agent": "writer"
        }
```

#### **Solution 3: LangGraph Recursion Limit**

```python
# Set a hard limit on total graph steps
graph = builder.compile(recursion_limit=20)

# If graph exceeds 20 steps, it will raise an error
```

---

### **Production Considerations**

#### **1. Cost Optimization**

Each agent call = 1 LLM invocation. Costs add up quickly.

**Problem:**

```
Supervisor (1 call) → Researcher (1 call) → Supervisor (1 call) →
Analyst (1 call) → Supervisor (1 call) → Writer (1 call)

Total: 6 LLM calls per request
```

**Optimization strategies:**

**A) Reduce supervisor calls:**

```python
# Instead of returning to supervisor after each agent,
# let agents decide the next step directly
def researcher_agent(state):
    return {
        "research_data": data,
        "next_agent": "analyst"  # Skip supervisor
    }
```

**B) Use cheaper models for routing:**

```python
# Use GPT-3.5 for supervisor (routing)
supervisor_llm = ChatOpenAI(model="gpt-3.5-turbo")

# Use GPT-4 for complex agents (analysis, writing)
analyst_llm = ChatOpenAI(model="gpt-4")
```

**C) Batch agent calls:**

```python
# Instead of sequential calls, run agents in parallel
async def run_parallel_agents(state):
    results = await asyncio.gather(
        researcher_agent(state),
        analyst_agent(state)
    )
    return combine_results(results)
```

#### **2. Observability**

Multi-agent systems are hard to debug. You need visibility into:

- Which agent is currently executing
- What each agent decided
- Why the supervisor routed to a specific agent

**Solution: Structured logging**

```python
import logging

def supervisor_node(state):
    decision = decide_next_agent(state)

    logging.info(
        "Supervisor decision",
        extra={
            "next_agent": decision,
            "reason": state.get("reasoning"),
            "state_summary": summarize_state(state)
        }
    )

    return {"next_agent": decision}
```

**Solution: LangSmith tracing**

```python
# LangSmith automatically traces multi-agent execution
# Shows: agent call tree, latency per agent, token usage per agent
```

#### **3. Error Handling**

If one agent fails, should the entire workflow fail?

**Strategy 1: Fail fast (strict)**

```python
def agent_wrapper(agent_func):
    def wrapper(state):
        try:
            return agent_func(state)
        except Exception as e:
            # Propagate error, stop execution
            raise
    return wrapper
```

**Strategy 2: Graceful degradation (resilient)**

```python
def agent_wrapper(agent_func, fallback_agent):
    def wrapper(state):
        try:
            return agent_func(state)
        except Exception as e:
            logging.error(f"Agent failed: {e}")
            # Route to fallback agent
            return {"next_agent": fallback_agent}
    return wrapper
```

**Strategy 3: Partial results (best UX)**

```python
def supervisor_node(state):
    if state.get("research_failed"):
        # Research failed, but we can still write a report
        # with available data
        return {"next_agent": "writer"}
    else:
        return {"next_agent": "researcher"}
```

---

### **Common Pitfalls & Best Practices**

#### **❌ Pitfall 1: Too Many Agents**

```python
# DON'T DO THIS - 10 agents for a simple task
agents = [
    "input_validator",
    "query_parser",
    "intent_classifier",
    "entity_extractor",
    "context_builder",
    "retriever",
    "ranker",
    "synthesizer",
    "formatter",
    "output_validator"
]
```

**Fix:** Start with 2-3 agents. Add more only when necessary.

#### **❌ Pitfall 2: Unclear Agent Responsibilities**

```python
# DON'T DO THIS - Overlapping responsibilities
researcher_agent  # Searches web AND analyzes data
analyst_agent     # Analyzes data AND writes reports
```

**Fix:** Each agent should have ONE clear responsibility.

#### **❌ Pitfall 3: No Loop Prevention**

```python
# DON'T DO THIS - Infinite loop possible
writer → critic → writer → critic → ... (forever)
```

**Fix:** Add iteration limits or quality thresholds.

#### **❌ Pitfall 4: Ignoring State Bloat**

```python
# DON'T DO THIS - State grows unbounded
class State(TypedDict):
    messages: List[BaseMessage]  # Grows to 1000+ messages
    all_research_data: List[str]  # Gigabytes of data
```

**Fix:** Prune state, summarize old data, or use external storage.

#### **✅ Best Practice 1: Start Simple**

```python
# Start with a sequential pipeline
graph.add_edge("researcher", "analyst")
graph.add_edge("analyst", "writer")

# Add supervisor later if needed
```

#### **✅ Best Practice 2: Test Agents Independently**

```python
# Test each agent in isolation before integrating
def test_researcher():
    state = {"query": "test query"}
    result = researcher_agent(state)
    assert "research_data" in result
```

#### **✅ Best Practice 3: Monitor Costs**

```python
# Track LLM calls per request
def track_cost(state):
    state["llm_calls"] = state.get("llm_calls", 0) + 1
    if state["llm_calls"] > 10:
        logging.warning("High LLM usage!")
    return state
```

---

## 🧠 Knowledge Check Questions

Before we start coding, think through these scenarios:

### **Question 1: Agent Specialization**

You're building a customer support system. Should you use:

A) One agent that handles all types of questions  
B) Separate agents for billing, technical, and general questions  
C) A supervisor that routes to specialized agents

**Why? What are the trade-offs?**

<details>
<summary>Click for answer</summary>

**Answer: C (Supervisor with specialized agents)**

**Why:**

- **Specialization:** Each agent can have domain-specific prompts and tools
- **Flexibility:** Easy to add new agent types (e.g., refunds agent)
- **Cost optimization:** Use cheaper models for simple questions

**Trade-offs:**

- More complex architecture
- Extra LLM call for supervisor routing
- Requires careful agent design

**A is wrong because:** One agent trying to do everything will be mediocre at all tasks.

**B is wrong because:** Without a supervisor, you need external routing logic.

</details>

---

### **Question 2: Infinite Loops**

You have a Writer → Critic → Writer loop. The Critic rejects drafts 30% of the time. You set a max of 3 revisions.

**What's the probability a report requires all 3 revisions?**

A) 30%  
B) 9%  
C) 2.7%  
D) Depends on the quality threshold

<details>
<summary>Click for answer</summary>

**Answer: C (2.7%)**

**Calculation:**

- Probability of 1st rejection: 30% = 0.3
- Probability of 2nd rejection: 30% = 0.3
- Probability of 3rd rejection: 30% = 0.3

Probability of ALL THREE rejections: 0.3 × 0.3 × 0.3 = 0.027 = 2.7%

**This shows:** Most reports will pass on the first or second try. The 3-revision limit is a safety net, not the norm.

</details>

---

### **Question 3: Parallel vs Sequential**

You have 3 research agents that each take 10 seconds. Running them sequentially takes 30 seconds. Running them in parallel takes 10 seconds.

**When should you NOT run them in parallel?**

A) When they're independent (don't need each other's results)  
B) When they share the same API (risk hitting rate limits)  
C) When you want to minimize latency  
D) When you have unlimited API quota

<details>
<summary>Click for answer</summary>

**Answer: B (When they share the same API)**

**Why:**

- If all 3 agents call the same API simultaneously, you might hit rate limits
- Example: OpenAI allows 10,000 RPM. 3 parallel calls might exceed this if you have other traffic

**Solutions:**

1. Add rate limiting
2. Stagger the calls (small delays)
3. Use different API keys for each agent

**A is wrong because:** Independence is a REASON to parallelize, not avoid it.

**C is wrong because:** Parallel execution MINIMIZES latency.

**D is wrong because:** Even with unlimited quota, shared resources can bottleneck.

</details>

---

### **Question 4: State Management**

You have 3 agents: Researcher, Analyst, Writer. The Analyst needs the Researcher's data. The Writer needs the Analyst's insights.

**How should you structure the state?**

A) Each agent has its own isolated state  
B) All agents share one state with all fields  
C) Use a database to pass data between agents  
D) Agents return data directly to each other

<details>
<summary>Click for answer</summary>

**Answer: B (Shared state with all fields)**

**Why:**

```python
class MultiAgentState(TypedDict):
    query: str
    research_data: Optional[str]  # Written by Researcher
    analysis: Optional[str]        # Written by Analyst
    draft: Optional[str]           # Written by Writer
```

- **Researcher** writes `research_data`
- **Analyst** reads `research_data`, writes `analysis`
- **Writer** reads `analysis`, writes `draft`

**A is wrong because:** Agents can't communicate without shared state.

**C is wrong because:** Adds unnecessary complexity and latency.

**D is wrong because:** LangGraph uses state-based communication, not direct returns.

</details>

---

### **Question 5: Cost Optimization**

You have a 4-agent workflow: Supervisor → Researcher → Supervisor → Analyst → Supervisor → Writer. Each agent makes 1 LLM call.

**How many LLM calls per request?**

A) 4 (one per agent)  
B) 6 (including supervisor calls)  
C) 7 (including initial supervisor call)  
D) Depends on the routing logic

<details>
<summary>Click for answer</summary>

**Answer: C (7 LLM calls)**

**Breakdown:**

1. Supervisor decides to call Researcher (1 call)
2. Researcher executes (1 call)
3. Supervisor decides to call Analyst (1 call)
4. Analyst executes (1 call)
5. Supervisor decides to call Writer (1 call)
6. Writer executes (1 call)
7. Supervisor decides to finish (1 call)

**Total: 7 calls**

**Optimization:** Let agents decide next steps directly, skipping supervisor:

```
Supervisor → Researcher → Analyst → Writer
Total: 4 calls (3 fewer!)
```

</details>

---

## 📝 Next Steps

Once you've thought through these questions, we'll move to the implementation phase where you'll build:

1. **Task 1:** Sequential multi-agent pipeline (Researcher → Analyst → Writer)
2. **Task 2:** Add a supervisor for dynamic routing
3. **Task 3:** Implement a revision loop with the Critic agent
4. **Task 4:** Convert agents to subgraphs for reusability
5. **Task 5:** Add parallel execution for independent agents

Ready to start coding? Let me know!
