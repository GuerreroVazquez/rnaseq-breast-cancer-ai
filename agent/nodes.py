
import os, sys
import pandas as pd
from agent.node_chatbot import ChatbotNode
from agent.node_plot import PlotNode
from agent.node_literature import LiteratureNode
from dotenv import load_dotenv
from agent.instructions import Instructions
from agent.literature_functions import LiteratureTools
from langgraph.prebuilt import ToolNode


current_path = os.path.dirname(os.path.abspath(__file__))



# Load .env file
load_dotenv()
# Get the API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# set the models
LOCATION = "europe-west1"
LLM_ROUTE = "gemini-1.5-flash"
LLM = "gemini-2.0-flash"
LLM_SQL = "gemini-2.5-flash-preview-04-17"
LLM_PLOT = "gemini-2.0-flash"
###### define instructions for nodes


MIRNA_ASSISTANT_SYSTEM_MESSAGE = Instructions.router.get_instruction()
MIRNA_COMPLETE_ANSWER = Instructions.format_answer.get_instruction()
PLOT_INSTRUCTIONS = Instructions.plot.get_instruction()
LITERATURE_INSTRUCTIONS = Instructions.literature.get_instruction()

##### Specific for SQL NODE


#### SQL connection

config = {
    'user': os.getenv('MIRKAT_USER'),
    'password': os.getenv('MIRKAT_PASSWORD'),
    'host': os.getenv('MIRKAT_HOST'),
    'database': os.getenv('MIRKAT_DATABASE'),
    'raise_on_warnings': True
}



##### Creating the nodes

master_node = ChatbotNode(llm=LLM, instructions=MIRNA_ASSISTANT_SYSTEM_MESSAGE)
literature_search_node = LiteratureNode(llm=LLM, functions=LiteratureTools, instructions=LITERATURE_INSTRUCTIONS)
plot_node = PlotNode(llm=LLM_PLOT, instructions=PLOT_INSTRUCTIONS)

