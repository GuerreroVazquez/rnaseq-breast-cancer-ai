
import os, sys
import pandas as pd
from agent.node_chatbot import ChatbotNode
from agent.node_plot import PlotNode
from agent.node_literature import LiteratureNode
from agent.instructions import Instructions
from agent.literature_functions import LiteratureTools
from agent.model_functions import *
from agent.node_model import ModelNode
from agent.node_rag import RagNode
from langgraph.prebuilt import ToolNode
import streamlit as st
from agent.create_db import (HtmlVectorDatabaseManager, DEFAULT_CORPUS_DIR,
                             DEFAULT_CHROMA_DB_DIR, DEFAULT_EMBEDDINGS_MODEL,
                             DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
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
LLM_RAG = "gemini-2.0-flash"
###### define instructions for nodes


ASSISTANT_SYSTEM_MESSAGE = Instructions.router.get_instruction()
COMPLETE_ANSWER = Instructions.format_answer.get_instruction()
PLOT_INSTRUCTIONS = Instructions.plot.get_instruction()
LITERATURE_INSTRUCTIONS = Instructions.literature.get_instruction()
RAG_INSTRUCTIONS = Instructions.rag.get_instruction()
MODEL_INSTRUCTIONS =  Instructions.model.get_instruction()
##### Specific for MODEL
model_functions = [validate_data, predict_clinical_features, predict_survival_outcome,
                 predict_survival_outcomes, get_column_names,
                 rename_columns, read_data_from_csv]

## SCPECIFC FOR RAG

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
CORPUS_DIR = DEFAULT_CORPUS_DIR
CHROMA_DB_DIR = DEFAULT_CHROMA_DB_DIR
EMBEDDINGS_MODEL = DEFAULT_EMBEDDINGS_MODEL
LLM_MODEL_NAME ="gemini-2.0-flash"
CHUNK_SIZE = DEFAULT_CHUNK_SIZE
CHUNK_OVERLAP = DEFAULT_CHUNK_OVERLAP
K_RETRIEVAL = 5

if 'db_manager' not in st.session_state:
    print("Initializing HtmlVectorDatabaseManager...")
    st.session_state['db_manager'] = HtmlVectorDatabaseManager(
        google_api_key=GOOGLE_API_KEY,
        corpus_dir=CORPUS_DIR,
        chroma_db_dir=CHROMA_DB_DIR,
        embeddings_model_name=EMBEDDINGS_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    print("HtmlVectorDatabaseManager stored in session_state.")

if 'vector_store' not in st.session_state or st.session_state['vector_store'] is None:
    print("Initializing or loading vector store...")
    # initialize_vector_store returns the Chroma instance or None on failure
    st.session_state['vector_store'] = st.session_state['db_manager'].initialize_vector_store(force_reindex=False)

    if st.session_state['vector_store'] is None:
        st.error("Failed to initialize or load vector database. RAG functionality will be unavailable.")
        # You might want to stop the app or disable related features here
    else:
        print("Vector store initialized/loaded and stored in session_state.")



vector_store = st.session_state.get('vector_store', None)
##### Creating the nodes

master_node = ChatbotNode(llm=LLM, instructions=ASSISTANT_SYSTEM_MESSAGE)
literature_search_node = LiteratureNode(llm=LLM, functions=LiteratureTools, instructions=LITERATURE_INSTRUCTIONS)
plot_node = PlotNode(llm=LLM_PLOT, instructions=PLOT_INSTRUCTIONS)
model_node = ModelNode(llm=LLM_MODEL, instructions=MODEL_INSTRUCTIONS, functions=model_functions )
rag_node = RagNode(llm=LLM_RAG, instructions=RAG_INSTRUCTIONS, vector_store=vector_store, k=5)