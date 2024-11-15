# main.py

import os
import sys
import streamlit as st

from crewai import Task, Crew
from crewai.process import Process

# Update the import to fix deprecation warning
from langchain_community.chat_models import ChatOpenAI

from agents import create_agents
from stream_to_expander import StreamToExpander

# Set the OpenAI model name
os.environ["OPENAI_API_KEY"] = "sk-proj-"

OPENAI_MODEL_NAME = "gpt-3.5-turbo"

# Ensure the OpenAI API key is set
if "OPENAI_API_KEY" not in os.environ or os.environ["OPENAI_API_KEY"] == "your_openai_api_key_here":
    st.error("Please set your OpenAI API key as an environment variable.")
    st.stop()

# Setup Manager LLM
manager_llm = ChatOpenAI(model_name=OPENAI_MODEL_NAME, temperature=0)

# Streamlit App
st.set_page_config(page_title="Molecule Design Assistant", page_icon="🧪", layout="wide")

# Add GIF at the heading
st.markdown(
    """
    <div style='text-align: center;'>
        <img src='https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExaWNobDVjMjJpNTc1a2J6dm5oZW40eTdlbWMxcnN1NHZzOW43NWF1MiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/VcAt04a901woawqLv2/giphy.gif' alt='GIF Image' width='200'>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 style='text-align: center;'>🧪 Molecule Design Assistant</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Interact with AI agents to design and optimize molecules.</h3>", unsafe_allow_html=True)

# Initialize session state variables
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""
if "results" not in st.session_state:
    st.session_state["results"] = []

# Main Tabs
tab1, tab2 = st.tabs(["Main App", "Results Table"])

# Tab 1: Main Application
with tab1:
    # Create columns for layout
    col1, col2 = st.columns([3, 1])

    # User Input Section
    with col2:
        st.header("User Input")
        def get_text():
            input_text = st.text_area(
                label="Your Message",
                placeholder="Enter your request here...",
                value=st.session_state["input_text"],
                key="input",
                height=100,
            )
            return input_text

        user_input = get_text()
        submit_button = st.button(label="Submit", key="submit_btn")

        if submit_button:
            if user_input:
                st.session_state["past"].append(user_input)
                st.session_state["input_text"] = ""  # Clear input after submission

    # Agent Outputs Section
    with col1:
        st.header("Agent Outputs")
        agent_expander = st.expander("💬 Agent Conversations", expanded=False)

    # Conversation History Section
    with col2:
        if st.session_state["generated"]:
            st.subheader("Conversation History")
            for i in range(len(st.session_state["generated"])):
                with st.container():
                    st.markdown(f"**You:** {st.session_state['past'][i]}")
                    st.markdown(f"**Assistant:** {st.session_state['generated'][i]}")
                    st.markdown("---")

    # Function to Process User Request
    def process_user_request(input_text: str):
        try:
            sys.stdout = StreamToExpander(agent_expander)  # Redirect stdout to expander

            # Create agents with verbose output
            agents = create_agents(verbose=True)

            # Create Crew with Hierarchical Process
            crew = Crew(
                agents=agents,
                process=Process.hierarchical,
                manager_llm=manager_llm,
                memory=True,
                planning=True,
                verbose=True,
            )

            task = Task(
                description=input_text,
                expected_output="Provide the appropriate output based on the user's request.",
                agent=None,
            )
            crew.tasks = [task]
            result = crew.kickoff()

            sys.stdout = sys.__stdout__  # Reset stdout
            return result, agents
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            sys.stdout = sys.__stdout__  # Reset stdout
            return None, None

    # Processing User Input
    if "past" in st.session_state and len(st.session_state["past"]) > len(st.session_state["generated"]):
        user_input = st.session_state["past"][-1]
        with col1:  # Place the spinner in the Agent Outputs section
            with st.spinner("🤖 **Agents at work...**"):
                output, agents = process_user_request(user_input)
        if output:
            st.session_state["generated"].append(output.raw if hasattr(output, "raw") else str(output))
            with col1:
                st.subheader("Final Answer")
                st.markdown(output.raw if hasattr(output, "raw") else str(output))
            st.session_state["results"].append({
                "Input": user_input,
                "Agents Used": ", ".join([agent.role for agent in agents]),
                "Output": output.raw if hasattr(output, "raw") else str(output),
            })
with tab2:
    st.header("Results Table")
    if st.session_state["results"]:
        import pandas as pd
        df = pd.DataFrame(st.session_state["results"])
        # Convert Output to string to avoid serialization issues
        df['Output'] = df['Output'].astype(str)
        st.table(df)
    else:
        st.write("No results yet.")

# Custom CSS for better UI
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        width: 100%;
        background-color: #2196F3 !important;  /* Changed to blue */
        color: white !important;
        border: none;
        padding: 10px 20px;
        text-align: center;
        font-size: 16px;
    }
    .stTextArea textarea {
        height: 100px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

