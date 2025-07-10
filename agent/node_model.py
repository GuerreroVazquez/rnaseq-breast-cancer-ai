import io
import json
import os
import random
import string
import traceback

import pandas as pd
from google.genai import Client
from google.genai import types
from langchain_core.messages import AIMessage
import streamlit as st
from google.genai.types import GenerateContentResponse


from agent.node_constructor import node
from agent.model_functions import (
    load_models_and_data,
    validate_data,
    predict_clinical_features,
    predict_survival_outcomes,
    recommend_treatment,
    get_column_names,
    rename_columns
)
import agent.model_functions as mf

# Get the API key
GOOGLE_API_KEY = st.secrets.get("GEMINI_API")

class ModelNode(node):
    def __init__(self, llm=None, instructions=None, functions=None, welcome=None):
        super().__init__(llm, instructions, functions, welcome, logging_key="Model Node.- ")
        self.set_model()
    def set_model(self):
            config_tools = types.GenerateContentConfig(
                system_instruction=self.instructions,
                tools=self.functions,
                temperature=0.0,
                )

            # Start a chat with automatic function calling enabled.
            self.chat = self.client.chats.create(
                model=self.llm,
                config=config_tools,
            )

    def run_model(self, messages):
        """Run the model with the given messages."""
        print(f"--- Message entering run model: {messages}---")
        self.log_message(f"Message entering run model: {messages}")
        text = messages.content
        print (f"--- Message going to the sql model: {text}---")
        self.log_message(f"Message going to the sql model: {text}")
        response = self.chat.send_message(text)
        return response


    def get_node(self,state):
        """The model llm that can run the models for prediction of clinical features,
          survival outcomes and treatment recommendations and give the information back to the user.."""

        self.log_message("Calling Model Node")
        self.log_message(f"State: {state}")
        history = state.get('history', [])
        # If history is empty, use the last message
        messages = state['request']
        if not messages:
            self.log_message("Model called with no messages.")
            return state

        print("The type of the message is: ", type(messages))
        self.log_message(f"The type of the message is: {type(messages)}")
        # check if it is GenerateContentResponse
        if isinstance(messages, GenerateContentResponse):
            print("The message is GenerateContentResponse, changing to AIMessage")
            self.log_message("The message is GenerateContentResponse, changing to AIMessage")
            messages = AIMessage(content=messages.candidates[0].content)
        elif isinstance(messages, str):
            print("The message is str, changing to AIMessage")
            self.log_message("The message is str, changing to AIMessage")
            messages = AIMessage(content=messages)
        elif isinstance(messages, AIMessage):
            pass
        else:
            print("The message is not str or AIMessage, changing to AIMessage")
            self.log_message("The message is not str or AIMessage, changing to AIMessage")
            print("The type of the message is: ", type(messages))
            self.log_message(f"The type of the message is: {type(messages)}")

        print("The message sent to the Model node is: ", messages)
        self.log_message(f"The message sent to the Model node is: {messages}")
        response = self.run_model(messages)
        self.log_message(f"Model LLM Response: {response}")
        new_answer = AIMessage(content=response.text)
        history = state.get("history", [])

        return {
            #"messages": response.content,
            "original_query": state["original_query"], # Add the router's decision/response
            "messages": AIMessage(content=""), # Add the router's decision/response
            "request": AIMessage(content=response.text), # Add the router's decision/response
            "answer": new_answer, # Return the potentially updated answer
            "finished": state.get("finished", False), # Use .get for safety
            "answer_source": 'MODEL_NODE',
            "trys": state.get("trys", 0) + 1, # Use .get for safety
            "history": history + [messages], # Update history with the new message
        }

