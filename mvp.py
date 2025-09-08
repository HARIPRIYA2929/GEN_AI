from langgraph.graph import StateGraph, END
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_community.chat_models import init_chat_model
from mcp.integrations.langchain import MCPTool

# Load LLM from Groq
model = init_chat_model("xxxxxxxx", model_provider="groq")

# Connect to your custom MCP tool server
math_tools = MCPTool.from_stdio("Math", path="python math_server.py")


llm = model.bind_tools([math_tools])

#  Tool executor node
tool_node = ToolNode([math_tools])

#  LLM reply 
def call_llm(state):
    messages = state["messages"]
    ai_response = llm.invoke(messages)
    return {"messages": messages + [ai_response]}

#  Tool decision logic
def should_continue(state):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END

#  Build LangGraph
builder = StateGraph(dict)
builder.add_node("llm", call_llm)
builder.add_node("tools", tool_node)
builder.set_entry_point("llm")
builder.add_conditional_edges("llm", should_continue, {"tools": "tools", "__end__": END})
builder.add_edge("tools", "llm")

graph = builder.compile()

# User question
state = {
    "messages": [HumanMessage(content="What is 12 times 6?")]
}


final_state = graph.invoke(state)


print("\n--- Conversation Flow ---")
for m in final_state["messages"]:
    role = m.__class__.__name__.replace("Message", "")
    print(f"{role}: {m.content}")



