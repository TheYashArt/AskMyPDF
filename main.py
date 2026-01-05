import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", type="password")
    
uploaded_files = st.file_uploader("Upload PDF files", type="pdf")

if uploaded_files:
    # Save uploaded files to a temporary location
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_files.getbuffer())
    
    # Load the PDF document
    loader = PyPDFLoader("temp.pdf")
    data = loader.load()
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    chunks=text_splitter.split_documents(data)
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    
    # Initialize the language model
    llm = ChatGroq(
            model = "llama-3.1-8b-instant",
            api_key = groq_api_key,
            temperature=0
        )
    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    
    # Ask a question about the document
    user_question = st.text_input("Ask a question about the document:")
    if user_question:
        response = qa_chain.run(user_question)
        st.write("Answer:\n", response)
else:
    st.write("Please upload PDF files and provide your LLM key.")