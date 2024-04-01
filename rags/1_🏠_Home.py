import streamlit as st
from streamlit_pills import pills
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to sys.path
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from st_utils import (
    add_sidebar,
    get_current_state,
)

current_state = get_current_state()

####################
#### STREAMLIT #####
####################


st.set_page_config(
    page_title="Eval CowPilot with TruLens",
    page_icon="🦑",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("Build a Eval CowPilot with Trulens 🦑")
st.info(
    "Use this page to build a agent-based evaluation test suite over your RAG system! "
    "Once the agent is finished creating, check out the `RAG Test Suite Config` and "
    "`Generated RAG Agent` pages.\n"
    "To build a new agent, please make sure that 'Create a new agent' is selected.",
    icon="ℹ️",
)
if "metaphor_key" in st.secrets:
    st.info("**NOTE**: The ability to add web search is enabled.")


# add_builder_config()
add_sidebar()


st.info(f"Currently building/editing agent: {current_state.cache.agent_id}", icon="ℹ️")

# add pills
selected = pills(
    "Outline your task!",
    [
        "I want to analyze this PDF file (./data/invoices.pdf)",
        "I want to search and index over an entire directory (i.e. ./data/custom_dataset).",
        "I want to load the content of a URL (i.e. https://www.wikipedia.org/).",
    ],
    clearable=True,
    index=None,
)

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Let's build a RAG test suite - start with specifying your dataset."}
    ]


def add_to_message_history(role: str, content: str) -> None:
    message = {"role": role, "content": str(content)}
    st.session_state.messages.append(message)  # Add response to message history


for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# TODO: this is really hacky, only because st.rerun is jank
if prompt := st.chat_input(
    "Your question",
):  # Prompt for user input and save to chat history
    # TODO: hacky
    if "has_rerun" in st.session_state.keys() and st.session_state.has_rerun:
        # if this is true, skip the user input
        st.session_state.has_rerun = False
    else:
        add_to_message_history("user", prompt)
        with st.chat_message("user"):
            st.write(prompt)

        # If last message is not from assistant, generate a new response
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = current_state.builder_agent.chat(prompt)
                    st.write(str(response))
                    add_to_message_history("assistant", str(response))

        else:
            pass

        # check agent_ids again
        # if it doesn't match, add to directory and refresh
        agent_ids = current_state.agent_registry.get_agent_ids()
        # check diff between agent_ids and cur agent ids
        diff_ids = list(set(agent_ids) - set(st.session_state.cur_agent_ids))
        if len(diff_ids) > 0:
            # # clear streamlit cache, to allow you to generate a new agent
            # st.cache_resource.clear()
            st.session_state.has_rerun = True
            st.rerun()

else:
    # TODO: set has_rerun to False
    st.session_state.has_rerun = False
