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
"""
You can add your unit tests here.
This is where you test your business logic, including agent functionality,
data processing, and other core components of your application.
"""
import pytest

from agent.node_constructor import (
    node
)
from agent.node_chatbot import ChatbotNode
from agent.node_plot import PlotNode
from agent.node_literature import LiteratureNode
import agent.plot_functions as plot_functions
import agent.model_functions as model_functions
from agent.node_model import ModelNode

from langchain_core.messages import (  # Grouped message types
    AIMessage, HumanMessage
)
from langchain_google_genai import ChatGoogleGenerativeAI

from google.genai.client import Client
from google.genai.chats import Chat

from google.genai.types import GenerateContentResponse
from google.genai.types import Candidate, Content,Part, GenerateContentConfig
import pickle
import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
if current_path.endswith("tests/unit"):
    # change the path to the parent directory
    sys.path.append("../../tests")
    
if current_path.endswith("agent"):
    sys.path.append("../tests")
## Test class node:
def test_create_node() -> None:
    """Check that the node is created correctly."""
    node_obj = node()
    assert node_obj is not None
    assert node_obj.llm is None
    assert node_obj.instructions is None
    assert node_obj.functions is None
    assert node_obj.welcome is None
    assert type(node_obj.client) == Client

#### Test class SQLNode:


def test_model_node() -> None:
    """Check that the node is created correctly."""
    model_node = ModelNode(instructions="you can run clinical features predictors", functions=[min, max])
    assert model_node is not None
    assert model_node.llm is None
    assert model_node.welcome is None
    assert type(model_node.client) == Client
    assert model_node.chat is not None
    assert type(model_node.chat) == Chat
    assert model_node.instructions == "you can run clinical features predictors"
    assert model_node.functions == [min, max]



def test_model_without_messages() -> None:
    """Check that the node is a responsive llm node"""
    model_node = ModelNode(instructions="you can run clinical features predictors", functions=[min, max])
    status = {
        "messages": [None],
        "request": None,
        "table": None,
        "answer": "",
        "finished": False}
    result = model_node.get_node(status)
    assert result['messages'] == [None]


def test_model_with_messages_str(monkeypatch) -> None:
    """Check that the node is a responsive llm node"""
    model_node = ModelNode(instructions="you can run clinical features predictors", functions=[min, max])
    status = {
        "messages": [HumanMessage(content="Hi")],
        "request":  AIMessage(content="hi"),
        "table": None,
        "answer": "",
        "original_query": "Hi",
        "finished": False}
    def mock_run_model(*args, **kwargs):
        fake_response = GenerateContentResponse
        fake_response.text = "Risotto alla Milanese"
        return fake_response

    monkeypatch.setattr(model_node, "run_model", 
                        mock_run_model)
    
    result = model_node.get_node(status)

    # check messages is AIMessage
    assert isinstance(result['messages'], AIMessage)
    assert result['request'].content == "Risotto alla Milanese"
    assert result['answer'] == AIMessage(content='Risotto alla Milanese')
    assert result['finished'] is False




def test_model_with_messages_GenerateContentResponse(monkeypatch) -> None:
    """Check that the node is a responsive llm node"""
    model_node = ModelNode(instructions="you are a SQL expert", functions=[min, max])
    status = {"messages":GenerateContentResponse, "request":"Run a model", "original_query":"question"}
    def mock_run_model(*args, **kwargs):
        fake_response = GenerateContentResponse
        fake_response.text = "Risotto alla Milanese"
        return fake_response

    monkeypatch.setattr(model_node, "run_model", 
                        mock_run_model)
    
    result = model_node.get_node(status)

    # check messages is AIMessage
    assert isinstance(result['messages'], AIMessage)
    assert result['request'].content == "Risotto alla Milanese"
    assert result['answer'] == AIMessage(content='Risotto alla Milanese')
    assert result['finished'] is False

