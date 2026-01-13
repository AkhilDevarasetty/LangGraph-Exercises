import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv(override=True)

# --- State ---
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    llm_response: str

# --- LLM Model Init ---
model = ChatOpenAI(model_name="gpt-4o", temperature=0)

# --- Node Functions ---
def call_llm(state: AgentState) -> AgentState:
    response = model.invoke(state["messages"])
    state["llm_response"] = response.content
    state["messages"].append(AIMessage(content= response.content))
    return state

# --- Graph Construction ---
graph = StateGraph(AgentState)
graph.add_node("llm_process", call_llm)
graph.add_edge(START, "llm_process")
graph.add_edge("llm_process", END)
memory_agent = graph.compile()

user_input = input("Enter your message: ")

context_history = []

while user_input != "exit":
    context_history.append(HumanMessage(content=user_input))
    result = memory_agent.invoke({"messages": context_history})
    print(f"LLM Response: {result['llm_response']}")
    context_history = result["messages"]
    user_input = input("Enter your message: ")

with open("conversation.txt", "w") as f:
    f.write("Your Conversation Log:\n\n")
    for message in context_history:
        if isinstance(message, HumanMessage):
            f.write("User: " + message.content + "\n")
        elif isinstance(message, AIMessage):
            f.write("AI: " + message.content + "\n\n")
    
    f.write("End of Conversation")
    
print("Conversation saved to conversation.txt")

