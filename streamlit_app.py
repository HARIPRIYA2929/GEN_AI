from dotenv import load_dotenv
load_dotenv()
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent


import streamlit as st


st.title("My GPT")


if "messages" not in st.session_state:
   st.session_state.messages = []




checkpointer = InMemorySaver()


agent = create_react_agent(
   model="groq:llama3-8b-8192", 
   tools=[], 
   checkpointer=checkpointer,
   prompt="You are a helpful assistant" 
)

def format_history():
    return [{"role": role, "content": content} for role, content in st.session_state.messages]


def stream_graph_updates(user_input : str):
   st.session_state.messages.append(("user", user_input))


   assistant_response = ""


   with st.chat_message("assistant"):
       message_placeholder = st.empty()
       for event in agent.stream({"messages": format_history()}, {"configurable": {"thread_id": "def"}}):
           for value in event.values():
               new_text = value["messages"][-1].content
               assistant_response += new_text
               message_placeholder.markdown(assistant_response)


   st.session_state.messages.append(("assistant", assistant_response))


for role, message in st.session_state.messages:
   with st.chat_message(role):
       st.markdown(message)


prompt = st.chat_input("What is your question?")
if prompt:
   # Display user input 
   with st.chat_message("user"):
       st.markdown(prompt)
   # Append user input to session state
   st.session_state.messages.append(("user", prompt))
  
   # response from the chatbot 
   response = stream_graph_updates(prompt)
