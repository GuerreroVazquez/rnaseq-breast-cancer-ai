from typing import List, Literal
import logging
import agent.nodes as nodes
from agent.graph_state import GraphState
from langgraph.graph import END, StateGraph
from google.genai.types import GenerateContentResponse
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('SCAN_B_AI.log'),
        logging.StreamHandler()
    ]
)


# Nodes

master_node = nodes.master_node
literature_search_node = nodes.literature_search_node
plot_node = nodes.plot_node
model_node = nodes.model_node
rag_node = nodes.rag_node

# start defininf graph

# Define node names for clarity
CHATBOT_NODE = "chatbot_router"
LITERATURE_NODE = "literature_search_node"
TOOL_NODE = "execute_tools" # Name for the ToolNode instance
PLOT_NODE = "plot_node"
MODEL_NODE = "model_node"
RAG_NODE = "rag_node"



### Routing criteria

# edges
# Router: After the Main Chatbot/Router (`chatbot_with_tools`)
def route_chatbot_decision(state: GraphState) -> Literal["sql_processor_node", "literature_search_node","human_node", "chatbot_router",  "__end__"]:
    """
    Inspects the last message from the main chatbot (`chatbot_with_tools`)
    and decides where to route the conversation next.
    """
    print("\n--- ROUTING: route_chatbot_decision ---")
    logging.info("Routing decision based on the last message from the chatbot.")
    print (f"--- Current state: {state} ---")
    logging.info(f"--- Current state: {state} ---")
    messages: List[BaseMessage] = state['request']
    logging.info(f"--- Messages in state: {messages} ---")
    if not messages:
        #print("--- Routing Error: No messages found in route_chatbot_decision ---")
        return END # Or raise error
    if isinstance(messages, list):
        last_message = messages[-1]
    else:
        last_message = messages
    if not isinstance(last_message, AIMessage) :
        print(f"--- Routing Warning: Expected AIMessage, got {type(last_message)}. Routing to Human. ---")
        #return HUMAN_NODE
        content = last_message
    else:
        content = last_message.content.strip()

    
    
    if "***ROUTE_TO_LITERATURE***" in content:
        #print("--- Routing: Master Router to Literature Searcher ---")
        # state['messages'][-1].content = "Okay, I need to search the literature for that."
        return LITERATURE_NODE
    elif "***PLOT***" in content:
        #print("---- Routing to plot node ----")
        return PLOT_NODE
    elif "***MODEL_NODE***" in content:
        return MODEL_NODE
    elif "***RAG_NODE***" in content:
        return RAG_NODE
    
    elif "***ANSWER_DIRECTLY***" in content:
        content = content.replace("***ANSWER_DIRECTLY***", "")
        print(f"--- The answer directly was: {content}")
        state['messages'].content = str(content)
        #  #print (f"--- The answer directly was: {answer}")
        state['answer'] = str(content)#.response.candidates[0].content.parts[0].text
        print(f"\n\n\n After Answer directly before returning to finihs \n\n\n\n")
        return CHATBOT_NODE
    elif "***FINISH***" in content or state.get("finished"): # Check flag too
        # remove the finish keyword
        content = content.replace("***FINISH***", "")
        return END
    else:
        logging.warning(f"\n\n\n Returning to CHATBOT NODE, key word unrecognized \n\n\n\n")
        return CHATBOT_NODE

# Router 3: After a Specialist Processor Node (`sql_processor_node`, `literature_search_node`)
def route_processor_output(state: GraphState) -> Literal["chatbot_router","human_node", "__end__"]:
    """
    Inspects the last message from a specialist processor node.
    Routes to 'tools' if a tool call was made (e.g., query_database, ground_search).
    Routes to 'human_node' if a final synthesized answer was provided.
    """
    logging.info("\n--- ROUTING: route_processor_output ---")
    messages: List[BaseMessage] = state['messages']
    if not messages:
        #print("--- Routing Error: No messages found in route_processor_output ---")
        return END

    last_message = messages
    return CHATBOT_NODE
        



# --- Build the Graph ---
workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node(CHATBOT_NODE, master_node.get_node)
workflow.add_node(LITERATURE_NODE, literature_search_node.get_node)
workflow.add_node(PLOT_NODE, plot_node.get_node)
workflow.add_node(MODEL_NODE, model_node.get_node)
workflow.add_node(RAG_NODE, rag_node.get_node)

# --- Define Edges ---

# 1. Entry Point (Where the graph starts)
workflow.set_entry_point(CHATBOT_NODE) # Start with a hello input

# Add direct edges
workflow.add_edge(PLOT_NODE, CHATBOT_NODE)
workflow.add_edge(LITERATURE_NODE, CHATBOT_NODE)
workflow.add_edge(MODEL_NODE, CHATBOT_NODE)
workflow.add_edge(RAG_NODE, CHATBOT_NODE)
# 2. From Human Node


# 3. From Main Chatbot Node
workflow.add_conditional_edges(
    CHATBOT_NODE,
    route_chatbot_decision, # Function to decide based on chatbot output
    {
        LITERATURE_NODE: LITERATURE_NODE,   # Route to Literature searcher
        PLOT_NODE: PLOT_NODE,              # Route to plot node
        CHATBOT_NODE: CHATBOT_NODE,         # Route back to chatbot for further processing
        RAG_NODE: RAG_NODE,                # Route to rag node
        MODEL_NODE: MODEL_NODE,            # Route to model
        END: END                           # Route to end (though usually handled via human)
    }
)

# define state

# Initial state with a welcome message
initial_state = {
    "messages": [], # <-- Empty list
    "table": None,
    "answer": "",
    "finished": False,
    "request": None,
    "original_query": "",
    "answer_source": "Human",
    "trys": 0,
    "history": [],
    "thread_id": "1" ,
    "file_path": ""
}
current_state = initial_state
config = {"recursion_limit": 100, "configurable": {"thread_id": "1"}}

checkpointer = InMemorySaver()
agent = workflow.compile(checkpointer=checkpointer)


def get_agent():
    """
    Returns the compiled agent workflow.
    """
    return agent

def run_langraph(current_state, config={"recursion_limit": 100,  "configurable": {"thread_id": "1"}}):
  return agent.invoke(current_state, config)