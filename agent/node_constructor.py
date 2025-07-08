from altair import Chart
from langchain_core.messages import ( 
    AIMessage
    )
import streamlit as st
from google.genai.types import GenerateContentResponse
from langchain_google_genai import ChatGoogleGenerativeAI
from google.genai import types
from google import genai 
from dotenv import load_dotenv
import os
from agent.plot_functions import PlotFunctons
import base64
import io
from agent.instructions import Instructions
import re
import json
# save logs
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('SCAN-B_AI.log'),
        logging.StreamHandler()
    ]
)

# Get the API key
GOOGLE_API_KEY = st.secrets.get('GEMINI_API')


class node:
    def __init__(self, llm=None, instructions=None, functions=None, welcome=None, logging_key = None, api_key=GOOGLE_API_KEY):
        if api_key is None:
            api_key = GOOGLE_API_KEY
        self.llm = llm
        self.instructions = instructions
        self.functions = functions
        self.welcome = welcome
        self.client = genai.Client(api_key=api_key)
        self.logging_key = logging_key
    def get_node(self, state):
        return None
    def log_message(self, message):
        """Log the message to the console."""
        logging.info(self.logging_key + message)
        print(message)
    
