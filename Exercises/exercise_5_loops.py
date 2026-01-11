from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from IPython.display import display, Image

# 1. Define State
class AgentState(TypedDict):
    draft: str
    feedback: str
    iterations: int

# 2. Nodes
def writer_node(state: AgentState) -> AgentState:
    """This node writes draft and save it to the state"""
    
    # Initialize iterations if missing (Safety check)
    if "iterations" not in state:
        state["iterations"] = 0
        
    print("Writing draft version", state["iterations"])
    
    # If feedback is "Too short", improve the draft and increment counter
    if state.get("feedback") == "Too short":
        state["iterations"] += 1
        state["draft"] = state["draft"] + " more content."
    else:
        # First pass or successful pass
        state["draft"] = "Draft saved"
        
    return state

def critic_node(state: AgentState) -> AgentState:
    """This node criticizes the draft and returns the feedback"""
    if len(state["draft"]) < 20:
        state["feedback"] = "Too short"
    else:
        state["feedback"] = "Perfect"
    return state

def router_node(state: AgentState) -> str:
    """This function routes the flow based on the feedback"""
    
    # Safety Valve: Prevent infinite loops
    if state.get("iterations", 0) > 5:
        print("Max iterations reached. Ending.")
        return "END"

    if state.get("feedback") == "Too short":
        return "writer_node"
    elif state.get("feedback") == "Perfect":
        return "END"
    return "END"

# 3. Build Graph
graph = StateGraph(AgentState)
graph.add_node("writer_node", writer_node)
graph.add_node("critic_node", critic_node)

graph.add_edge(START, "writer_node")
graph.add_edge("writer_node", "critic_node")

graph.add_conditional_edges(
    "critic_node",
    router_node,
    {
        "writer_node": "writer_node",
        "END": END
    }
)

app = graph.compile()

# 4. Run it
# We initialize with iterations=0 to be safe
inputs = {"draft": "Draft", "iterations": 0}
result = app.invoke(inputs)

print("--- FINAL RESULT ---")
print('Final draft:', result['draft'])
