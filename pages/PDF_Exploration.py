import os
import streamlit as st
from typing import Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_classic.chains import ConversationalRetrievalChain

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

if "history_store" not in st.session_state:
    st.session_state.history_store = {}

def get_chat_history(session_id: str):
    if session_id not in st.session_state.history_store:
        st.session_state.history_store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.history_store[session_id]

# if "history_store" not in st.session_state:
#     st.session_state.history_store = {}

def get_chat_history(session_id: str):
    if session_id not in st.session_state.history_store:
        st.session_state.history_store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.history_store[session_id]

import os
from typing import TypedDict, List

st.header("Document Exploration with PDF Uploads")
st.text("Upload PDF files and ask questions about their content.")    
uploaded_files = st.file_uploader("Upload PDF files", type="pdf")

st.subheader("Quick Actions")
col1, col2, col3 = st.columns(3)
prompt = None


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
    
    with col1:
        if st.button("Summarize Document", icon="📝"):
            if uploaded_files:
                prompt = "Provide a concise summary of the document."
    with col2:
        if st.button("Generate Questions", icon="❓"):
            if uploaded_files:
                prompt = "Generate five insightful questions based on the document content."
    with col3:
        if st.button("Key Takeaways", icon="📌"):
            if uploaded_files:
                prompt = "List the key takeaways from the document."

    # Initialize the language model
    llm = ChatGoogleGenerativeAI(
            model = "gemini-2.5-flash-lite",
            api_key = st.secrets["GEMINI_API_KEY"],
            temperature=0
        )
    # Create the RetrievalQA chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff"
    )
    rag_with_history = RunnableWithMessageHistory(
        qa_chain,
        get_chat_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    session_id = "default_user"  # you can make this dynamic later

    if prompt or (user_input := st.chat_input("Enter your question about the document")):
        input_text = prompt if prompt else user_input

        with st.chat_message("user"):
            st.markdown(input_text)

        with st.chat_message("assistant"):
            response = rag_with_history.invoke(
                {"question": input_text},
                config={"configurable": {"session_id": session_id}}
            )
            st.markdown(response["answer"])
else:
    st.write("Please Upload PDF File")