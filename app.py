import streamlit as st
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import gdown
import os

# Debugging: Print installed libraries
st.write("Installed libraries:")
st.write(os.popen("pip list").read())

# Step 1: Download Dataset and Model from Google Drive
def download_from_drive():
    # Google Drive file IDs
    dataset_file_id = "12hwhc4G0BC1pkRQ0KjgbNKMgLmJ9aa5_"  # Dataset file ID
    model_file_id = "1sziMRE691psjZEYZaudz0Y8Jg0lX59PF"  # Model file ID

    # Download dataset
    dataset_url = f"https://drive.google.com/uc?id={dataset_file_id}"
    dataset_path = "dataset.csv"
    if not os.path.exists(dataset_path):
        try:
            gdown.download(dataset_url, dataset_path, quiet=False)
            st.success("Dataset downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download dataset: {e}")
            st.stop()

    # Download model (assuming it's a zip file)
    model_url = f"https://drive.google.com/uc?id={model_file_id}"
    model_zip_path = "medical_chatbot_model.zip"
    model_dir = "medical_chatbot_model"
    if not os.path.exists(model_dir):
        try:
            gdown.download(model_url, model_zip_path, quiet=False)
            st.success("Model downloaded successfully!")
            # Unzip the model
            import zipfile
            with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
                zip_ref.extractall(model_dir)
            os.remove(model_zip_path)  # Clean up the zip file
            st.success("Model extracted successfully!")
        except Exception as e:
            st.error(f"Failed to download or extract model: {e}")
            st.stop()

# Step 2: Initialize Embeddings and VectorStore
@st.cache_resource
def initialize_vectorstore(queries):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(queries, embeddings)
    return vectorstore.as_retriever()

# Step 3: Load the Pre-trained T5 Model and Tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model_dir = "medical_chatbot_model"  # Ensure this path is correct
    try:
        model = T5ForConditionalGeneration.from_pretrained(model_dir)
        tokenizer = T5Tokenizer.from_pretrained(model_dir)
        st.success("Model and tokenizer loaded successfully!")
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load model and tokenizer: {e}")
        st.stop()

# Step 4: Define the Prompt Template
def create_prompt_template():
    prompt_template = """
    You are a medical expert. Based on the context provided below, answer the following question.

    Context: {context}

    Question: {question}

    Answer:
    """
    return PromptTemplate(input_variables=["context", "question"], template=prompt_template)

# Step 5: Set up the ConversationalRetrievalChain
@st.cache_resource
def initialize_qa_chain(model, retriever, prompt):
    return ConversationalRetrievalChain.from_llm(
        model=model,
        retriever=retriever,
        prompt=prompt
    )

# Streamlit App
def main():
    st.set_page_config(page_title="Medical Chatbot", page_icon="💬", layout="wide")
    st.title("Medical Chatbot")
    st.write("Ask any medical-related question below:")

    # Download dataset and model from Google Drive
    download_from_drive()

    # Load dataset
    try:
        df = pd.read_csv("dataset.csv")
        if "input_text" not in df.columns:
            st.error("The dataset does not contain an 'input_text' column.")
            st.stop()
        queries = df["input_text"].tolist()
        if len(queries) == 0:
            st.error("The dataset is empty or contains no valid queries.")
            st.stop()
    except FileNotFoundError:
        st.error("The dataset file 'dataset.csv' was not found.")
        st.stop()

    # Initialize embeddings and vectorstore
    retriever = initialize_vectorstore(queries)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Create prompt template
    prompt = create_prompt_template()

    # Initialize QA chain
    qa_chain = initialize_qa_chain(model, retriever, prompt)

    # Input field for user query
    user_query = st.text_input("Enter your question:")

    # Button to submit the query
    if st.button("Submit"):
        if user_query:
            with st.spinner("Generating response..."):
                try:
                    response = qa_chain.run({"question": user_query})
                    st.success("Response:")
                    st.write(response)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
