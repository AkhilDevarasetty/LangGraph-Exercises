from typing import TypedDict
from langgraph.graph import StateGraph
from IPython.display import display, Image

## Define the state
class AgentState(TypedDict):
    text: str

## Define the nodes
def shouter_node(state: AgentState) -> AgentState:
    """This node converts the text to uppercase"""
    state["text"] = state["text"].upper()
    return state

def reverser_node(state:AgentState) -> AgentState:
    """This node reverses the text"""
    state["text"] = state["text"][::-1]
    print(state["text"])
    return state

def star_wrapper_node(state:AgentState) -> AgentState:
    """This node adds stars to the text"""
    state["text"] = "*** " + state["text"] + " ***"
    return state

## Define and build the graph
graph = StateGraph(AgentState)
graph.add_node("shouter_node", shouter_node)
graph.add_node("reverser_node",reverser_node)
graph.add_node("star_wrapper_node",star_wrapper_node)

## Set the entry and finish point
graph.set_entry_point("shouter_node")
graph.set_finish_point("star_wrapper_node")

##Add edges
graph.add_edge("shouter_node","reverser_node")
graph.add_edge("reverser_node","star_wrapper_node")

## Always compile the graph before invoking it
app = graph.compile()

display(Image(app.get_graph().draw_mermaid_png()))

result = app.invoke({"text": "LangGraph Tutorial"})

print(result['text'])