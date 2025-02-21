import streamlit as st
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import gdown
import os
import logging

# Fix for sqlite3 version issue
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules["pysqlite3"]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit app configuration
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="💬",
    layout="wide"
)

# Sidebar for downloads
with st.sidebar:
    st.title("Medical Chatbot")
    st.write("This chatbot answers medical-related questions based on a pre-trained model.")

    # Get file IDs from environment variables
    dataset_file_id = st.secrets["DATASET_FILE_ID"]
    model_file_id = st.secrets["MODEL_FILE_ID"]

    # Download the dataset from Google Drive if it doesn't exist
    dataset_path = "dataset.csv"
    if not os.path.exists(dataset_path):
        st.write("Downloading the dataset...")
        dataset_url = f"https://drive.google.com/uc?export=download&id={dataset_file_id}"
        try:
            logger.info("Downloading dataset...")
            gdown.download(dataset_url, dataset_path, quiet=False)
            st.success("Dataset downloaded successfully!")
        except Exception as e:
            logger.error(f"Failed to download the dataset: {e}")
            st.error(f"Failed to download the dataset: {e}")

    # Download the model from Google Drive if it doesn't exist
    model_path = "medical_chatbot_model"
    if not os.path.exists(model_path):
        st.write("Downloading the model...")
        model_url = f"https://drive.google.com/uc?export=download&id={model_file_id}"
        try:
            logger.info("Downloading model...")
            gdown.download(model_url, model_path, quiet=False)
            st.success("Model downloaded successfully!")
        except Exception as e:
            logger.error(f"Failed to download the model: {e}")
            st.error(f"Failed to download the model: {e}")

# Load the dataset and initialize components
@st.cache_resource
def load_data_and_model():
    # Load dataset
    if not os.path.exists("dataset.csv"):
        logger.error("Dataset not found. Please ensure it is downloaded.")
        st.error("Dataset not found. Please ensure it is downloaded.")
        st.stop()

    logger.info("Loading dataset...")
    df = pd.read_csv("dataset.csv")
    df.dropna(subset=["input_text", "target_text"], inplace=True)
    queries = df["input_text"].tolist()

    # Embeddings and VectorStore
    logger.info("Creating embeddings and vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(queries, embeddings)
    retriever = vectorstore.as_retriever()

    # Load the pre-trained T5 model and tokenizer
    if not os.path.exists("medical_chatbot_model"):
        logger.error("Model not found. Please ensure it is downloaded.")
        st.error("Model not found. Please ensure it is downloaded.")
        st.stop()

    logger.info("Loading model and tokenizer...")
    model = T5ForConditionalGeneration.from_pretrained("medical_chatbot_model")
    tokenizer = T5Tokenizer.from_pretrained("medical_chatbot_model")

    # Define the prompt template
    prompt_template = """
    You are a medical expert. Based on the context provided below, answer the following question.

    Context: {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    # Set up the ConversationalRetrievalChain
    logger.info("Setting up ConversationalRetrievalChain...")
    qa_chain = ConversationalRetrievalChain.from_llm(
        model=model,
        retriever=retriever,
        prompt=prompt
    )

    return qa_chain

# Load the model and data
logger.info("Loading model and data...")
qa_chain = load_data_and_model()

# Main Streamlit app
st.title("Medical Chatbot")
st.write("Ask any medical-related question below:")

# Input field for user query
user_query = st.text_input("Enter your question:")

# Button to submit the query
if st.button("Submit"):
    if user_query:
        with st.spinner("Generating response..."):
            try:
                logger.info("Running query through conversational chain...")
                # Run the query through the conversational chain
                response = qa_chain.run({"question": user_query})
                st.success("Response:")
                st.write(response)
            except Exception as e:
                logger.error(f"An error occurred: {e}")
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")
