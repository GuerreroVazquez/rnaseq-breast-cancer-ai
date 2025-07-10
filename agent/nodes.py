
import os, sys
import pandas as pd
from agent.node_chatbot import ChatbotNode
from agent.node_plot import PlotNode
from agent.node_literature import LiteratureNode
from agent.instructions import Instructions
from agent.literature_functions import LiteratureTools
from agent.model_functions import *
from agent.node_model import ModelNode
from langgraph.prebuilt import ToolNode
import streamlit as st

vector_store = st.session_state.get('vector_store', None)
current_path = os.path.dirname(os.path.abspath(__file__))



# Load .env file
# Get the API key
GOOGLE_API_KEY = st.secrets.get("GEMINI_API")
# set the models
LOCATION = "europe-west1"
LLM_ROUTE = "gemini-2.5-flash"
LLM = "gemini-2.0-flash"
LLM_MODEL = "gemini-2.5-flash"
LLM_PLOT = "gemini-2.0-flash"
###### define instructions for nodes


ASSISTANT_SYSTEM_MESSAGE = Instructions.router.get_instruction()
COMPLETE_ANSWER = Instructions.format_answer.get_instruction()
PLOT_INSTRUCTIONS = Instructions.plot.get_instruction()
LITERATURE_INSTRUCTIONS = Instructions.literature.get_instruction()
MODEL_INSTRUCTIONS =  Instructions.model.get_instruction()
##### Specific for MODEL
model_functions = [validate_data, predict_clinical_features, predict_survival_outcome,
                 predict_survival_outcomes, get_column_names,
                 rename_columns, read_data_from_csv]


#### SQL connection

config = {
    'user': os.getenv('MIRKAT_USER'),
    'password': os.getenv('MIRKAT_PASSWORD'),
    'host': os.getenv('MIRKAT_HOST'),
    'database': os.getenv('MIRKAT_DATABASE'),
    'raise_on_warnings': True
}



##### Creating the nodes

master_node = ChatbotNode(llm=LLM, instructions=ASSISTANT_SYSTEM_MESSAGE)
literature_search_node = LiteratureNode(llm=LLM, functions=LiteratureTools, instructions=LITERATURE_INSTRUCTIONS)
plot_node = PlotNode(llm=LLM_PLOT, instructions=PLOT_INSTRUCTIONS)
model_node = ModelNode(llm=LLM_MODEL, instructions=MODEL_INSTRUCTIONS, functions=model_functions )
