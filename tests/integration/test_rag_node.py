# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from executing.executing import function_node_types
# mypy: disable-error-code="union-attr"

from google.genai.types import GenerateContentResponse
from google.genai.types import Candidate, Content,Part, GenerateContentConfig
import os, sys
current_path = os.path.dirname(os.path.abspath(__file__))
if current_path.endswith("agent"):
    sys.path.append("../tests")
#if current_path.endswith("tests/integration"):
#    sys.path.append("../")
#    sys.path.append("../app")
import logging

from google.genai.types import Candidate, Content,Part, GenerateContentConfig, GenerateContentResponse
from agent.node_rag import RagNode
from langchain_core.messages import ( # Grouped message types
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage
    # ToolMessage is implicitly handled by LangGraph/ToolNode
)
from unittest.mock import patch, MagicMock
import pickle
from  agent.instructions import Instructions
import streamlit  as st
from agent.create_db import (HtmlVectorDatabaseManager, DEFAULT_CORPUS_DIR,
                             DEFAULT_CHROMA_DB_DIR, DEFAULT_EMBEDDINGS_MODEL,
                             DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)

GOOGLE_API_KEY = st.secrets.get("GEMINI_API")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
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



def test_files_accesibility() -> None:
    """
    Check that the files are accessible.
    """
    files = [
        'models/estimator_survival_clinical.pkl',
        'models/scaler_survival_clinical.pkl',
        'models/multi_output_model_selected_.pkl',
        'models/scaler_selected.pkl',
        'models/selected_genes.txt',
        'dummy_files/clinical_data.csv'
    ]
    for file in files:
        assert os.path.exists(file), f"File {file} does not exist"
def test_agent_model(monkeypatch) -> None:
    """
    Integration test for the agent stream query functionality.
    Tests that the agent returns valid streaming responses.
    """
    input_dict = {
        "request": AIMessage(content="Hi"),
        "answer": None,
        "finished": False,
        "user_id": "test-user",
        "session_id": "test-session",
        "original_query": "Hi"
    }
    llm = "gemini-2.5-flash-preview-04-17"
    instructions  = Instructions.rag.get_instruction()

    vector_store = st.session_state.get('vector_store', None)
    ragNode = RagNode( llm=llm, instructions=instructions, welcome=None, vector_store=vector_store)
    response = ragNode.get_node(state = input_dict)
    print(response)
    assert response
    assert response['answer_source']=='RAG_NODE'


def test_agent_model_clinical(monkeypatch) -> None:
    """
    Integration test for the agent stream query functionality.
    Tests that the agent returns valid streaming responses.
    """
    input_dict = {
        "request": AIMessage(content="What is the GEO Series ID of the data?"),
        "answer": None,
        "finished": False,
        "user_id": "test-user",
        "session_id": "test-session",
        "original_query": "What is the data about?"
    }
    llm = "gemini-2.5-flash-preview-04-17"
    instructions = Instructions.rag.get_instruction()

    vector_store = st.session_state.get('vector_store', None)
    ragNode = RagNode(llm=llm, instructions=instructions, welcome=None, vector_store=vector_store)
    response = ragNode.get_node(state=input_dict)
    print(response)
    assert response
    assert response['answer_source']=='RAG_NODE'
    assert 'GSE96058' in response['answer'].content
