import streamlit as st
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import gdown
import os
import zipfile
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
    dataset_file_id = "1A2B3C4D5E6F7G8H9I0J"  # Replace with your dataset file ID
    model_folder_zip_id = "1sziMRE691psjZEYZaudz0Y8Jg0lX59PF"  # Replace with your model folder .zip file ID

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

    # Download the model folder as a .zip file and extract it
    model_folder_path = "medical_chatbot_model"
    if not os.path.exists(model_folder_path):
        st.write("Downloading the model folder...")
        model_folder_zip_url = f"https://drive.google.com/uc?export=download&id={model_folder_zip_id}"
        try:
            logger.info("Downloading model folder...")
            gdown.download(model_folder_zip_url, "model_folder.zip", quiet=False)
            st.success("Model folder downloaded successfully!")

            # Extract the .zip file
            logger.info("Extracting model folder...")
            with zipfile.ZipFile("model_folder.zip", "r") as zip_ref:
                zip_ref.extractall(model_folder_path)
            st.success("Model folder extracted successfully!")
        except Exception as e:
            logger.error(f"Failed to download or extract the model folder: {e}")
            st.error(f"Failed to download or extract the model folder: {e}")

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
    model_folder_path = "medical_chatbot_model"
    if not os.path.exists(model_folder_path):
        logger.error("Model folder not found. Please ensure it is downloaded and extracted.")
        st.error("Model folder not found. Please ensure it is downloaded and extracted.")
        st.stop()

    logger.info("Loading model and tokenizer...")
    model = T5ForConditionalGeneration.from_pretrained(model_folder_path)
    tokenizer = T5Tokenizer.from_pretrained(model_folder_path)

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
