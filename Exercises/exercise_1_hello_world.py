from typing import TypedDict, Dict
from langgraph.graph import StateGraph

## Defining the state
class AgentState(TypedDict):
    message: str

## Defining the node
def greeting_node(state: AgentState) -> AgentState:
    """This is a simple node that adds/updates the exitsing state message"""
    state["message"] = "Hey " + state["message"] + ",how is your day going ?"
    return state

## Defining the graph
graph = StateGraph(AgentState)
graph.add_node("greedting-node",greeting_node)
graph.set_entry_point("greedting-node")
graph.set_finish_point("greedting-node")

app = graph.compile()

from IPython.display import display, Image
display(Image(app.get_graph().draw_mermaid_png()))

result = app.invoke({"message": "Niha"})

print(result['message'])
