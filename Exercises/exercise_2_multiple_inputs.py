from typing import TypedDict, List
from langgraph.graph import StateGraph
from IPython.display import display, Image

## Define the state
class AgentState(TypedDict):
    value: List[int]
    name: str
    operation: str
    result: str

## Define the node
def process_input_node(state: AgentState) -> AgentState:
    """This node processes the input and updates the state"""
    if state["operation"] == "+":
        state["result"] = f"Hi {state['name']}, your answer is {sum(state['value'])}"
    elif state["operation"] == "*":
        product_result = map(lambda x:x*x, state['value'])
        state["result"] = f"Hi {state['name']}, your answer is {sum(product_result)}"
    else:
        state["result"] = f"Hi {state['name']}, your answer is {sum(state['value'])}"
    return state ## returning the updated state

## Define and build the graph
graph = StateGraph(AgentState)
graph.add_node("process-input-node", process_input_node)
graph.set_entry_point("process-input-node")
graph.set_finish_point("process-input-node")

## Always compile the graph before invoking it
app = graph.compile()

## Display the compiled graph
display(Image(app.get_graph().draw_mermaid_png()))

## Invoke the graph
result = app.invoke({"value": [999,45,50], "operation": "+", "name": "Akshay"})

print(result['result'])

