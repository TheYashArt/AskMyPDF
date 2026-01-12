import streamlit as st
import os
import json
import hashlib
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

if "history_store" not in st.session_state:
    st.session_state.history_store = {}
    
        # Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

VECTOR_DB_DIR = "vector_db"

os.makedirs(VECTOR_DB_DIR, exist_ok=True)
def get_pdf_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def load_chat_history(file_path="./chats/chat.json"):
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        with open(file_path, "w") as f:
            json.dump([], f)
        return []
    
    with open(file_path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []
    
def session_initialization(chats, session_id):
    for session in chats:
        if session["session_id"] == session_id:
            return session
    new_session = {
        "session_id" : session_id,
        "messages": []
    }
    chats.append(new_session)
    return new_session 

session_id = "default_user"
chats = load_chat_history()
session = session_initialization(chats, session_id)
        
def load_chats(session_id, file_path="./chats/chat.json"):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return []

    with open(file_path, "r") as f:
        try:
            chats = json.load(f)
        except json.JSONDecodeError:
            return []

    for session in chats:
        if session["session_id"] == session_id:
            return session["messages"]

    return []

def build_full_context(session_id, question):
    # Get last 3 chat messages
    chat_history = load_chats(session_id)[-3:]
    chat_text = "\n".join(f"{m['role']}: {m['content']}" for m in chat_history)

    # Get relevant PDF chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    pdf_docs = retriever.invoke(question)  # returns list of Document objects
    pdf_text = "\n\n".join(d.page_content for d in pdf_docs)

    # Combine chat + PDF
    context = ""
    if chat_text:
        context += f"Conversation history:\n{chat_text}\n\n"
    if pdf_text:
        context += f"Relevant PDF content:\n{pdf_text}"

    return context

def load_chats_as_string(session_id, last_n=3):
    """Load last N chats and return as a formatted string."""
    messages = load_chats(session_id)[-last_n:]
    chat_str = ""
    for msg in messages:
        role = "User" if msg["role"] == "user" else "AI"
        chat_str += f"{role}: {msg['content']}\n"
    return chat_str

def get_pdf_content(query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    # The retriever is a Runnable-like object now
    return retriever.invoke(query)  # Pass as string

def to_langchain_messages(messages):
    lc_messages = []
    for msg in messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))
    return lc_messages
    
def get_chat_history(session_id):
    if session_id not in st.session_state.history_store:
        history = InMemoryChatMessageHistory()
        
        stored_msg = load_chats(session_id)
        history.add_messages(to_langchain_messages(stored_msg))
        st.session_state.history_store[session_id] = history
    return st.session_state.history_store[session_id]
    
def save_chat_history(session_id,role, content, file_path = "./chats/chat.json"):
    chats = load_chat_history()
    for session in chats:
        if session["session_id"] == session_id:
            session["messages"].append({"role": role, "content": content})
            break
    else:
        chats.append({"session_id": session_id, "messages": [{"role": role, "content": content}]})
    with open(file_path, "w") as f:
        json.dump(chats, f, indent=4)
    
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
    

    
    file_bytes = uploaded_files.getbuffer()
    file_hash = get_pdf_hash(file_bytes)
    vector_db_path = os.path.join(VECTOR_DB_DIR, f"{file_hash}.faiss")

    embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(vector_db_path):
        vectorstore = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    else:
        loader = PyPDFLoader("temp.pdf")
        data = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(data)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(vector_db_path)
    
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
    llm= ChatGroq(
        model = "llama-3.1-8b-instant",
        api_key = st.secrets["GROQ_API_KEY"],
        temperature=0
    )

    llama = ChatOllama(
        model = "llama3.1:8b",
        temperature=0,
        verbose=True
    )
    # Display chat messages from history on app rerun
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"]) 


    if prompt or (user_input := st.chat_input("Enter your question about the document")):
        
        
        input_text = prompt if prompt else user_input
        prompt_chat = ChatPromptTemplate.from_messages(
            [   ("system", 
                        "You are a helpful AI assistant. Answer the question using only the information from the context below.\n\nContext:\n{context}"
                        ),
                # MessagesPlaceholder(variable_name = "chat_history"),
                ("user", "{question}")
            ]
        )
        rag_chain = (
            {
                "question": RunnableLambda(lambda x: x["question"]),
                "context": RunnableLambda(lambda x: build_full_context(session_id, x["question"])),
            } |
            prompt_chat |
            llama
        )
        rag_with_history = RunnableWithMessageHistory(
            rag_chain,
            get_chat_history,
            input_messages_key="question",  
            history_messages_key="chat_history",
        )
        
        with st.chat_message("user"):
            st.markdown(input_text)
            st.session_state.messages.append({"role": "user", "content": input_text})
            save_chat_history(session_id, "user", input_text)

        with st.chat_message("assistant"):
            response = rag_with_history.invoke(
                {"question": input_text},
                config={"configurable": {"session_id": session_id}}
            )
            st.session_state.messages.append({"role": "assistant", "content": response.content})
            save_chat_history(session_id, "assistant", response.content)
            st.rerun()

else:
    st.write("Please Upload PDF File")