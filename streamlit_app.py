import streamlit as st


st.set_page_config(page_title="KarenGV - SCAN-B exploration", page_icon=":dna:", layout="wide")
st.title('SCAN-B exploration')

openai_api_key = st.secrets.get('OPENAI_API')
gemini_api_key = st.secrets.get('GEMINI_API')

from langchain.llms import OpenAI


import os
import logging
from typing import Any, Dict, List, Literal, Optional, TypedDict

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('SCANB_AI.log'),
        logging.StreamHandler()
    ]
)

# Environment variables
from dotenv import load_dotenv

# LangChain and Google AI specific libraries
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage
)
from google.genai.types import GenerateContentResponse
from langchain_core.tools import tool

# LangGraph specific libraries
from langgraph.graph import END, StateGraph

# Application-specific imports
import agent.nodes as nodes
from agent.graph_state import GraphState
from agent.langraph_model import get_agent

import logging

# log to SCANB_AI.log
logging.basicConfig(
    filename='SCANB_AI.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
agent = get_agent()

def run_langraph(current_state, config={"recursion_limit": 100}):
  return agent.invoke(current_state, config)

def generate_response(input_text):
  llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
  st.info(llm(input_text))

with st.form('my_form'):
  text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
  submitted = st.form_submit_button('Submit')
  if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
  if submitted and openai_api_key.startswith('sk-'):
    initial_state = GraphState()
    initial_state['messages'] = [HumanMessage(content=text)]
    answer = run_langraph(initial_state)
    print (f"Answer: {answer}")
    logging.info(f"Answer: {answer}")
    anwer_text = answer['messages'].content
    st.write(anwer_text)
    


    
