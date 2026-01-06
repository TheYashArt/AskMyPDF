import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
import os


st.header("Document Exploration with PDF Uploads")
st.text("Upload PDF files and ask questions about their content.")    
uploaded_files = st.file_uploader("Upload PDF files", type="pdf")

if uploaded_files:
    # Save uploaded files to a temporary location
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_files.getbuffer())
    
    # Load the PDF document
    loader = PyPDFLoader("temp.pdf")
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    
    # Initialize the language model
    llm = ChatGroq(
            model = "llama-3.1-8b-instant",
            api_key = st.secrets["GROQ_API_KEY"],
            temperature=0
        )
    response = None
    user_question = ""
    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Summurize Document", icon="📝"):
            summurize = "Provide a concise summary of the document."
            user_question = summurize
            response = qa_chain.run(summurize)
            
    with col2:
        if st.button("Generate Questions", icon="❓"):
            questions = "Generate five insightful questions based on the document content."
            user_question = questions
            response = qa_chain.run(questions)
    with col3:
        if st.button("Key Takeaways", icon="📌"):
            takeaways = "List the key takeaways from the document."
            user_question = takeaways
            response = qa_chain.run(takeaways) 
    with st.expander("Ask your own question"):
        user_question = st.text_input("Enter your question about the document:")
        if st.button("Get Answer", icon="💡"):
            if user_question:
                response = qa_chain.run(user_question)
    
            
    # Ask a question about the document

    if response:
        st.write("Response:\n", response)
else:
    st.write("Please Upload PDF File")