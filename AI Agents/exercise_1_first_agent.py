from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


load_dotenv(override=True)

openai_llm_model = ChatOpenAI(model_name="gpt-4o", temperature=0)

class AgentState(TypedDict):
    messages: List[HumanMessage]
    llm_response: str

def call_llm(state:AgentState) -> AgentState:
    response = openai_llm_model.invoke(state["messages"])
    state["llm_response"] = response.content
    return state

graph = StateGraph(AgentState)
graph.add_node("call_llm", call_llm)

graph.add_edge(START, "call_llm")
graph.add_edge("call_llm", END)

app = graph.compile()

user_input = input("Enter your message: ")

while user_input != "exit":
    result = app.invoke({"messages": [HumanMessage(content=user_input)]})
    print(result["llm_response"])
    user_input = input("Enter your message: ")



