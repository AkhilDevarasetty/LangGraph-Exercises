from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import display, Image

## Define the state/app schema
class AgentState(TypedDict):
    age: int
    status: str

## Define the nodes
def check_id_node(state: AgentState) -> AgentState:
    """This node just prints the welcome message and returns the state without any changes"""
    print("Checking ID...")
    return state

def router(state: AgentState) -> str:
    """This node routes the user to the appropriate node"""
    if state["age"] >= 21:
        return "allow_edge"
    else:
        return "deny_edge"

def allow_node(state: AgentState) -> AgentState:
    """This node allows the user to enter the club"""
    state["status"] = "Welcome to the club!"
    return state

def deny_node(state: AgentState) -> AgentState:
    """This node denies the user to enter the club"""
    state["status"] = "Sorry, you are too young."
    return state

## Build the graph
graph = StateGraph(AgentState)
graph.add_node("check_id_node", check_id_node)
graph.add_node("allow_node", allow_node)
graph.add_node("deny_node", deny_node)

## Build the edges and conditional edges
graph.add_edge(START, "check_id_node")
graph.add_conditional_edges(
    "check_id_node",
    router,
    {
        "allow_edge": "allow_node",
        "deny_edge": "deny_node"
    }
)
graph.add_edge("allow_node", END)
graph.add_edge("deny_node", END)

app = graph.compile()
display(Image(app.get_graph().draw_mermaid_png()))

initialState = AgentState(age=20)

result = app.invoke(initialState)

print(result["status"])


    