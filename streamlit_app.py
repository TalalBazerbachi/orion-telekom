import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
# import chromadb
# from llama_index.vector_stores.chroma import ChromaVectorStore


st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)

openai.api_key = st.secrets["OPEN_API_KEY"]
# st.logo("image001.png")
st.title("Chat with the Orion's AI Agent")
st.info("This agent has knowledge about information on this website and about Orion")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about Streamlit's open-source Python library!",
        }
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    reader = SimpleDirectoryReader(input_dir="./orion-kb", recursive=True)
    docs = reader.load_data()
    Settings.llm = OpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        system_prompt="""You are an expert customer service representative for Orion Telekom. 
        Your job is to assist customers by providing accurate and relevant information based on 
        the knowledge base created from the data on the company's website. 
        Your responses should be factual, clear, and directly related to the content available in the vector store collection. 
        Do not generate information beyond what is provided in the knowledge base. 
        If you do not have enough information to answer a question, suggest contacting customer support for further assistance.""",
    )
    
    # chroma_client = chromadb.PersistentClient(path="/Users/talalbazerbachi/Documents/dictionary-extractor/customer-service")
    # chroma_collection = chroma_client.get_collection("orion-kb")
    # vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    # print(f"Collection: {chroma_collection}")
    # print(f"Attributes: {dir(chroma_collection)}")

    index = VectorStoreIndex.from_documents(docs)
    return index


index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )

if prompt := st.chat_input(
    "Ask a question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        # Add response to message history
        st.session_state.messages.append(message)
