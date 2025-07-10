from typing import Optional, Dict, Any

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import pandas as pd
from pathlib import Path
import streamlit as st
from agent.node_constructor import node
from google.genai import types
from langchain_core.messages import ( 
    AIMessage
    ) 


# Get the API key
GOOGLE_API_KEY = st.secrets.get("GEMINI_API")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

class RagNode(node):
    """
    A LangGraph node for Retrieval Augmented Generation using a pre-built index.
    Inherits from the base 'node' class.
    """
    def __init__(self, llm=None, instructions=None, functions=None, welcome=None, vector_store=None, k=None):
        super().__init__(llm=llm, instructions=instructions, functions=functions, welcome=welcome, logging_key="RAG node.- ")
        self.set_model()
        self.vector_store = vector_store
        if k is None:
            self.k=5
        else:
            self.k=k

    def set_model(self):
        config_tools = types.GenerateContentConfig(
            system_instruction=self.instructions,
            tools=self.functions,
            temperature=0.0,
            )

        # Start a chat with automatic function calling enabled.
        self.chat = ChatGoogleGenerativeAI(model=self.llm,
                                            temperature=0.5)

    def get_retriever(self, vector_store, k) -> Runnable:
        """
        Retrieves the vector store from the state and returns it as a retriever.
        This method is expected to be overridden in subclasses if needed.
        """
        if not vector_store:
            raise ValueError("Vector store not found in state.")
        return vector_store.as_retriever(search_kwargs={"k": k})
    
    def run_model(self, messages):
        """Run the model with the given messages."""
        response = self.chat.invoke(messages)
        return response

    def get_node(self, state):
        """
        Get RAG model response based on the provided state.
        """
        retriever = self.get_retriever(self.vector_store, self.k)
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.instructions),
            ("human", "{input}")
        ])
        combine_docs_chain = create_stuff_documents_chain(
            self.chat,
            prompt_template
        )
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        question = state.get("request", "").content
        response = retrieval_chain.invoke({"input":question})
        self.log_message(f"RAG node response: {response}")
        return {
                #"messages": response.content,
                "original_query": state["original_query"], # Add the router's decision/response
                "messages": AIMessage(content=""), # Add the router's decision/response
                "request": AIMessage(content=response['answer']), # Add the router's decision/response
                "answer": AIMessage(content=response['answer']), # Return the potentially updated answer
                "finished": state.get("finished", False), # Use .get for safety
                "answer_source": 'RAG_NODE',
                "trys": state.get("trys", 0) + 1, # Use .get for safety
                "history": state.get("history", 0), # Update history with the new message
            }
    
