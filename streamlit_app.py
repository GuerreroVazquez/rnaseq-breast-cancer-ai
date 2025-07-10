import streamlit as st
import os
from langchain.llms import OpenAI
# LangChain and Google AI specific libraries
from langchain_core.messages import (
    HumanMessage,

)
from agent.graph_state import GraphState
from agent.langraph_model import run_langraph
from agent.create_db import (HtmlVectorDatabaseManager, DEFAULT_CORPUS_DIR,
                             DEFAULT_CHROMA_DB_DIR, DEFAULT_EMBEDDINGS_MODEL,
                             DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)

import logging
from agent.instructions import Instructions

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
os.environ["GOOGLE_API_KEY"] = gemini_api_key






uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    # Ensure 'user' directory exists
    user_folder = "user"
    os.makedirs(user_folder, exist_ok=True)

    # Save the uploaded file
    file_path = os.path.join(user_folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Saved file to {file_path}")
    


if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.markdown(f"**You:** {msg.content}")
    else:
        st.markdown(f"**AI:** {msg.content}")



user_input = st.text_input("Your message:", value=st.session_state.user_input, key="input")

if st.button("Send"):

    if user_input.strip() != "":
        # Add user message
        st.session_state.messages.append(HumanMessage(content=user_input))
        print(f"User input: {user_input}")
        logging.info(f"User input: {user_input}")
        # Run agent with full history
        state = GraphState()
        state ={
                  "messages": [],
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
        state['history'] = st.session_state.messages
        state["messages"] = [HumanMessage(content=user_input)]
        state["file_path"] = file_path if uploaded_file else None

        answer = run_langraph(state)
        if answer:
          logging.info(f"Answer: {answer}")
          final_answer = answer['messages']
          if isinstance(final_answer, str):
              model_reply = final_answer
          else:
              model_reply = final_answer.content
          if "<image>" in model_reply:
              # get the image adress in the labels <image>...</image>
              image= model_reply.split("<image>")[1].split("</image>")[0]
              st.image(image, caption="Generated Image", use_column_width=True)
          # Add model reply
          st.session_state.messages.append(final_answer)

        # Clear input box
        st.session_state.user_input = ""


        # Rerun to show new messages
        st.rerun()




