from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import random
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

# Streamlit UI 

st.set_page_config(page_title="My GPT", page_icon="🤖")
st.title("🤖 Welcome to My GPT!")
st.subheader("🌟 Your smart assistant, game buddy, and riddle master!")
st.divider()

# Puzzle Section 

if "current_puzzle" not in st.session_state:
    puzzle_list = [
        ("What comes once in a minute, twice in a moment, but never in a thousand years?", "m"),
        ("I speak without a mouth and hear without ears. I have nobody, but I come alive with the wind.", "echo"),
        ("I have keys but no locks, space but no room. You can enter but can’t go outside. What am I?", "keyboard"),
        ("What gets wetter the more it dries?", "towel"),
        ("The more you take, the more you leave behind. What are they?", "footsteps")
    ]
    st.session_state.current_puzzle = random.choice(puzzle_list)
    st.session_state.puzzle_solved = False

puzzle_question, puzzle_answer = st.session_state.current_puzzle

st.subheader(" Can You Crack This Riddle?")
st.markdown(f" Riddle: {puzzle_question}*")

if not st.session_state.puzzle_solved:
    puzzle_guess = st.text_input("Type your answer here:", key="puzzle_guess_input")

    if st.button("Submit Answer", key="submit_puzzle"):
        if puzzle_guess.strip().lower() == puzzle_answer.lower():
            st.success(" Correct! You cracked the puzzle.")
            st.session_state.puzzle_solved = True
        else:
            st.error(" Oops! That's not it. Try again or guess something else.")
else:
    st.info(" You already solved this puzzle! Refresh to get a new one.")

st.divider()

# Word Scramble Game 

WORDS = ["streamlit", "langchain", "assistant", "groq", "python", "chatbot", "prompt", "memory", "graph", "puzzle"]

def scramble(word):
    word = list(word)
    random.shuffle(word)
    return ''.join(word)

if "target_word" not in st.session_state:
    st.session_state.target_word = random.choice(WORDS)
    st.session_state.scrambled_word = scramble(st.session_state.target_word)

def reset_word_game():
    st.session_state.target_word = random.choice(WORDS)
    st.session_state.scrambled_word = scramble(st.session_state.target_word)

st.subheader(" Word Scramble Game")
st.markdown("Can you **unscramble** this word? Use your brain 🧠!")

st.info(f" Scrambled Word: **{st.session_state.scrambled_word}**")

user_guess = st.text_input("Your guess here:", key="guess_input")

if st.button("Check Answer"):
    if user_guess.lower().strip() == st.session_state.target_word:
        st.success(f" Correct! The word was **{st.session_state.target_word}**.")
        reset_word_game()
    else:
        st.error(" Nope! Try again!")

st.divider()

# LangGraph Assistant 

if "messages" not in st.session_state:
    st.session_state.messages = []

checkpointer = InMemorySaver()

agent = create_react_agent(
    model="groq:llama3-8b-8192",
    tools=[],
    checkpointer=checkpointer,
    prompt="You are a helpful assistant."
)

def format_history():
    return [{"role": role, "content": content} for role, content in st.session_state.messages]

def stream_graph_updates(user_input: str):
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

# Chatbot UI 

for role, message in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(message)

prompt = st.chat_input("💬 Ask your assistant anything!")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(("user", prompt))
    stream_graph_updates(prompt)
