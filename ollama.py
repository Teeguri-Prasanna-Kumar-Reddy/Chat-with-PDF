import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Function to process a PDF and return search results
def process_pdf(query):
    embeddings = OllamaEmbeddings(model="llama3")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    results = db.similarity_search(query)
    return results

# Streamlit App
st.title("PDF Search App")

query = st.text_input("Enter your search query")

if st.button("Search"):
    results = process_pdf(query)

    if results is not None:
        st.subheader("Search Results:")
        st.write("Result :", results[0].page_content)
