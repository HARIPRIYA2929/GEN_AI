import os
from dotenv import load_dotenv

load_dotenv()

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

from langgraph.prebuilt import create_react_agent

def get_weather(city: str) -> str:  
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_react_agent(
    model="groq:XXXX",  
    tools=[get_weather],  
    #prompt="You are a helpful assistant" 
    checkpointer=checkpointer 
)

# Run the agent
#response1=agent.invoke(
#    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
#)

#print(response1["messages"][-1].content)

#response = agent.invoke(
#   {"messages": [{"role": "user", "content": "who are you?"}]}
#)


#print(response)
# Run the agent
config = {"configurable": {"thread_id": "1"}}
sf_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    config  
)
print(sf_response["messages"][-1].content)
ny_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what about new york?"}]},
    config
)
print(ny_response["messages"][-1].content)

try:
   img = agent.get_graph().draw_mermaid_png()
   with open("agent.png", "wb") as f:
       f.write(img)
except Exception:
   pass