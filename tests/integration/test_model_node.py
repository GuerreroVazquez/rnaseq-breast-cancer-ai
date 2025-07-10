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

from agent.instructions import instructions

current_path = os.path.dirname(os.path.abspath(__file__))
if current_path.endswith("agent"):
    sys.path.append("../tests")
#if current_path.endswith("tests/integration"):
#    sys.path.append("../")
#    sys.path.append("../app")
import logging

from google.genai.types import Candidate, Content,Part, GenerateContentConfig, GenerateContentResponse
from agent.node_model import ModelNode
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
from agent.model_functions import *

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
        "request": AIMessage(content="Hi'?"),
        "answer": None,
        "finished": False,
        "user_id": "test-user",
        "session_id": "test-session",
        "original_query": "Hi"
    }
    llm = "gemini-2.5-flash-preview-04-17"
    instructions  = Instructions.model.get_instruction()
    functions = [validate_data, predict_clinical_features, predict_survival_outcome,
                 predict_survival_outcomes, get_column_names,
                 rename_columns, read_data_from_csv]
    modelNode = ModelNode( llm=llm, instructions=instructions, functions=functions, welcome=None)
    response = modelNode.get_node(state = input_dict)
    print(response)
    assert response
    assert response['answer_source']=='MODEL_NODE'


def test_agent_model_clinical(monkeypatch) -> None:
    """
    Integration test for the agent stream query functionality.
    Tests that the agent returns valid streaming responses.
    """
    input_dict = {
        "request": AIMessage(content=" What is the survival of the people which clinical data is in path 'dummy_files/clinical_data.csv'?"),
        "answer": None,
        "finished": False,
        "user_id": "test-user",
        "session_id": "test-session",
        "original_query": " What is the survival of the people which clinical data is in path 'dummy_files/clinical_data.csv'?"
    }
    llm = "gemini-2.5-flash-preview-04-17"
    instructions  = Instructions.model.get_instruction()
    functions = [validate_data, predict_clinical_features, predict_survival_outcome,
                 predict_survival_outcomes, get_column_names,
                 rename_columns, read_data_from_csv]
    modelNode = ModelNode( llm=llm, instructions=instructions, functions=functions, welcome=None)
    response = modelNode.get_node(state = input_dict)
    print(response)
    assert response
    assert response.answer_source=='MODEL_NODE'
    assert '98' in response.answer



def test_agent_model_genetic(monkeypatch) -> None:
    """
    Integration test for the agent stream query functionality.
    Tests that the agent returns valid streaming responses.
    """
    input_dict = {
        "request": AIMessage(content=" What are the clinical values of the people which clinical data is in path 'dummy_files/genetic_data.csv'?"),
        "answer": None,
        "finished": False,
        "user_id": "test-user",
        "session_id": "test-session",
        "original_query": "What is the clinical values of these people"
    }
    llm = "gemini-2.5-flash-preview-04-17"
    instructions  = Instructions.model.get_instruction()
    functions = [validate_data, predict_clinical_features, predict_survival_outcome,
                 predict_survival_outcomes, get_column_names,
                 rename_columns, read_data_from_csv]
    modelNode = ModelNode( llm=llm, instructions=instructions, functions=functions, welcome=None)
    response = modelNode.get_node(state = input_dict)
    print(response)
    assert response
    assert response.answer_source=='MODEL_NODE'
    assert '98' in response.answer

