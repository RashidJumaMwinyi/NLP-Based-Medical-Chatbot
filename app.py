import streamlit as st
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import gdown
import os

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

    # Download the dataset from Google Drive if it doesn't exist
    dataset_path = "chatbot_data/dataset.csv"
    if not os.path.exists(dataset_path):
        st.write("Downloading the dataset...")
        dataset_url ="https://drive.google.com/uc?export=download&id=1A2B3C4D5E6F7G8H9I0J"  # Replace with your dataset FILE_ID
        try:
            gdown.download(dataset_url, dataset_path, quiet=False)
            st.success("Dataset downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download the dataset: {e}")

    # Download the model from Google Drive if it doesn't exist
    model_path = "medical_chatbot_model"
    if not os.path.exists(model_path):
        st.write("Downloading the model...")
        model_url = "https://drive.google.com/uc?export=download&id=1A2B3C4D5E6F7G8H9I0J"  # Replace with your model FILE_ID
        try:
            gdown.download(model_url, model_path, quiet=False)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download the model: {e}")

# Load the dataset and initialize components
@st.cache_resource
def load_data_and_model():
    # Load dataset
    df = pd.read_csv("dataset.csv")
    df.dropna(subset=["input_text", "target_text"], inplace=True)
    queries = df["input_text"].tolist()

    # Embeddings and VectorStore
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(queries, embeddings)
    retriever = vectorstore.as_retriever()

    # Load the pre-trained T5 model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    # Define the prompt template
    prompt_template = """
    You are a medical expert. Based on the context provided below, answer the following question.

    Context: {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    # Set up the ConversationalRetrievalChain
    qa_chain = ConversationalRetrievalChain.from_llm(
        model=model,
        retriever=retriever,
        prompt=prompt
    )

    return qa_chain

# Load the model and data
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
                # Run the query through the conversational chain
                response = qa_chain.run({"question": user_query})
                st.success("Response:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")
