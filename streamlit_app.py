import streamlit as st
import os
from langchain.llms import OpenAI
# LangChain and Google AI specific libraries
from langchain_core.messages import (
    HumanMessage,

)
from agent.graph_state import GraphState
from agent.langraph_model import get_agent
from agent.create_db import (HtmlVectorDatabaseManager, DEFAULT_CORPUS_DIR,
                             DEFAULT_CHROMA_DB_DIR, DEFAULT_EMBEDDINGS_MODEL,
                             DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)

import logging

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('SCANB_AI.log'),
        logging.StreamHandler()
    ]
)
st.set_page_config(page_title="KarenGV - SCAN-B exploration", page_icon=":dna:", layout="wide")
st.title('SCAN-B exploration')

openai_api_key = st.secrets.get('OPENAI_API')
gemini_api_key = st.secrets.get('GEMINI_API')

agent = get_agent()

CORPUS_DIR = os.getenv("CORPUS_DIR", DEFAULT_CORPUS_DIR)
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", DEFAULT_CHROMA_DB_DIR)
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", DEFAULT_EMBEDDINGS_MODEL)
LLM_MODEL_NAME = os.getenv("MODEL", "gemini-2.0-flash")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", DEFAULT_CHUNK_SIZE))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP))
K_RETRIEVAL = int(os.getenv("K_RETRIEVAL", 5))


if 'db_manager' not in st.session_state:
    print("Initializing HtmlVectorDatabaseManager...")
    st.session_state['db_manager'] = HtmlVectorDatabaseManager(
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




def run_langraph(current_state, config={"recursion_limit": 100,  "configurable": {"thread_id": "1"}}):
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
    


    
