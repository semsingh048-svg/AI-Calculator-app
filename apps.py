import streamlit as st
import os
import signal

from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, SystemMessage



# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Agentic Calculator & Assistant", layout='wide')
st.title("🧠 Agentic Calculator & General Assistant 🌐")
st.write("Enter a calculation (e.g., 'What is 12 * 99?') or a general question (e.g., 'What is AI?').")

# --- API Key Retrieval ---
try:
    # Read API key from environment variable
    groq_api_key = os.environ.get('GROQ_API_KEY')
    if not groq_api_key:
        st.error("Groq API key not found. Please ensure it's set as an environment variable `GROQ_API_KEY`.")
        st.stop()
except Exception as e:
    st.error(f"Error retrieving Groq API key: {e}.")
    st.stop()

# --- LLM Initialization ---
llm = ChatGroq(
    model="llama-3.3-70b-versatile", # Or your preferred model
    temperature=0,
    api_key=groq_api_key
)

# --- Tool Definitions ---
def add(x: float, y: float) -> float:
    """Adds two numbers."""
    return x + y

def subtract(x: float, y: float) -> float:
    """Subtracts the second number from the first."""
    return x - y

def multiply(x: float, y: float) -> float:
    """Multiplies two numbers."""
    return x * y

def divide(x: float, y: float) -> float:
    """Divides the first number by the second. Handles division by zero."""
    if y == 0:
        # st.error("Error: Division by zero is not allowed.") # Avoid Streamlit calls within tools if possible
        return float('nan') # Return Not a Number for error case
    return x / y

search = DuckDuckGoSearchRun() # General web search tool

# Combine all tools
tools = [add, subtract, multiply, divide, search]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

# --- LangGraph Agent Setup ---

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def assistant_node(state: State):
    current_messages = state["messages"]
    # Add a system message to guide the LLM for better tool use
    system_message_content = "You are a helpful assistant. Use the provided tools to answer questions, prioritize arithmetic tools for calculations. If you can't use a tool, respond directly. Perform only one tool call at a time. Do not nest tool calls." # More detailed system prompt
    response = llm_with_tools.invoke([SystemMessage(content=system_message_content)] + current_messages)
    return {"messages": [response]}

def should_continue(state: State) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# Build the LangGraph
builder = StateGraph(State)
builder.add_node("assistant", assistant_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", should_continue)
builder.add_edge("tools", "assistant")

# Compile the graph with memory
# Using MemorySaver for basic in-memory checkpointing
agent_graph = builder.compile(checkpointer=MemorySaver())

# --- Streamlit UI for Interaction ---

st.markdown("--- ")
st.subheader("🔢 Calculator Tools")

# Calculator Input Fields
col_num1, col_op, col_num2, col_equals = st.columns([2, 1, 2, 1])

with col_num1:
    calc_num1 = st.number_input("First Number", value=0.0, key="calc_num1")
with col_num2:
    calc_num2 = st.number_input("Second Number", value=0.0, key="calc_num2")

calc_result = None

calc_col_buttons = st.columns(4)

with calc_col_buttons[0]:
    if st.button("Add (+)", key="btn_add"):
        calc_result = add(calc_num1, calc_num2)
with calc_col_buttons[1]:
    if st.button("Subtract (-)", key="btn_subtract"):
        calc_result = subtract(calc_num1, calc_num2)
with calc_col_buttons[2]:
    if st.button("Multiply (*)", key="btn_multiply"):
        calc_result = multiply(calc_num1, calc_num2)
with calc_col_buttons[3]:
    if st.button("Divide (/)", key="btn_divide"):
        calc_result = divide(calc_num1, calc_num2)

if calc_result is not None:
    if calc_result == float('nan'): # Check for NaN explicitly for division by zero
        st.error("Error: Division by zero is not allowed.")
    else:
        st.success(f"**Calculator Result: {calc_result}**")

st.markdown("--- ")
st.subheader("💬 General Agent Query")

user_query = st.text_input("Your query:", key="agent_query_input")

if st.button("Ask Agent", key="ask_agent_button") and user_query:
    with st.spinner("Thinking..."):
        try:
            # Use a unique thread_id for each session or query for stateful interaction
            config = {"configurable": {"thread_id": "streamlit_agent_session"}}
            result = agent_graph.invoke(
                {"messages": [("user", user_query)]},
                config=config
            )
            
            # Display the final answer from the agent
            final_message = result["messages"][-1]
            if final_message.content:
                st.success("**Agent's Response:**")
                st.markdown(final_message.content)
            else:
                st.warning("The agent processed your request but returned no direct content.")

        except Exception as e:
            st.error(f"An error occurred while processing your request: {e}")
            st.info("Please ensure your Groq API key is correct and try again.")
elif st.button("Ask Agent", key="ask_agent_button_disabled") and not user_query:
    st.warning("Please enter a query to get a response.")
