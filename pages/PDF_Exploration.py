"""
PDF Exploration Module - A Streamlit application for uploading and querying PDF documents.
Uses LangChain for document processing, embedding generation, and RAG (Retrieval-Augmented Generation).
Supports multiple LLM backends (Groq, Ollama) and maintains persistent chat history.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import streamlit as st
import os
import json
import hashlib
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

# ============================================================================
# STREAMLIT SESSION STATE INITIALIZATION
# ============================================================================
# Initialize chat history store for managing multiple session conversations
if "history_store" not in st.session_state:
    st.session_state.history_store = {}

# Initialize messages list for displaying chat history in the UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================================================
# CONSTANTS
# ============================================================================
# Directory path for storing Chroma vector databases
VECTOR_DB_DIR = "vector_db"

# Create vector database directory if it doesn't exist
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_pdf_hash(file_bytes):
    """
    Generate an MD5 hash of PDF file bytes for caching vector databases.
    
    Args:
        file_bytes: Binary content of the PDF file
    
    Returns:
        Hexadecimal hash string used as vector database identifier
    """
    return hashlib.md5(file_bytes).hexdigest()


def load_chat_history(file_path="./chats/chat.json"):
    """
    Load chat history from JSON file, creating it if it doesn't exist.
    
    Args:
        file_path: Path to the JSON file storing chat history
    
    Returns:
        List of chat sessions or empty list if file doesn't exist/is invalid
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Create empty file if it doesn't exist
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        with open(file_path, "w") as f:
            json.dump([], f)
        return []
    
    # Load and parse JSON, returning empty list on error
    with open(file_path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []
    
def session_initialization(chats, session_id):
    """
    Initialize or retrieve a chat session by ID.
    Creates a new session if it doesn't exist.
    
    Args:
        chats: List of existing chat sessions
        session_id: Unique identifier for the session
    
    Returns:
        Dictionary containing session_id and messages list
    """
    # Check if session already exists
    for session in chats:
        if session["session_id"] == session_id:
            return session
    
    # Create new session if not found
    new_session = {
        "session_id" : session_id,
        "messages": []
    }
    chats.append(new_session)
    return new_session

# ============================================================================
# SESSION INITIALIZATION
# ============================================================================
session_id = "default_user"
chats = load_chat_history()
session = session_initialization(chats, session_id)
        
def load_chats(session_id, file_path="./chats/chat.json"):
    """
    Load chat messages for a specific session from the JSON file.
    
    Args:
        session_id: Unique identifier for the chat session
        file_path: Path to the JSON file storing chat history
    
    Returns:
        List of messages for the session or empty list if not found
    """
    # Return empty list if file doesn't exist
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return []

    # Load chat file and find matching session
    with open(file_path, "r") as f:
        try:
            chats = json.load(f)
        except json.JSONDecodeError:
            return []

    # Return messages from matching session
    for session in chats:
        if session["session_id"] == session_id:
            return session["messages"]

    return []

def build_full_context(session_id, question):
    """
    Build RAG context by combining recent chat history and relevant PDF chunks.
    
    Args:
        session_id: Unique identifier for the chat session
        question: User's current question to retrieve relevant PDF content
    
    Returns:
        String containing combined conversation history and relevant PDF content
    """
    # Retrieve last 3 chat messages for conversation context
    chat_history = load_chats(session_id)[-3:]
    chat_text = "\n".join(f"{m['role']}: {m['content']}" for m in chat_history)

    # Query vector store to get top 5 most relevant PDF chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    pdf_docs = retriever.invoke(question)  # Returns list of Document objects
    pdf_text = "\n\n".join(d.page_content for d in pdf_docs)

    # Combine chat history and PDF content into single context string
    context = ""
    if chat_text:
        context += f"Conversation history:\n{chat_text}\n\n"
    if pdf_text:
        context += f"Relevant PDF content:\n{pdf_text}"

    return context

def load_chats_as_string(session_id, last_n=3):
    """
    Load last N chat messages and format as a human-readable string.
    
    Args:
        session_id: Unique identifier for the chat session
        last_n: Number of recent messages to retrieve (default: 3)
    
    Returns:
        Formatted string with conversation history
    """
    messages = load_chats(session_id)[-last_n:]
    chat_str = ""
    for msg in messages:
        role = "User" if msg["role"] == "user" else "AI"
        chat_str += f"{role}: {msg['content']}\n"
    return chat_str

def to_langchain_messages(messages):
    """
    Convert message dictionaries to LangChain message objects.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
    
    Returns:
        List of HumanMessage or AIMessage objects for LangChain processing
    """
    lc_messages = []
    for msg in messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))
    return lc_messages
    
def get_chat_history(session_id):
    """
    Retrieve or initialize chat history for a session from Streamlit session state.
    Loads previous messages from storage into LangChain-compatible format.
    
    Args:
        session_id: Unique identifier for the chat session
    
    Returns:
        InMemoryChatMessageHistory object containing session messages
    """
    # Create new history if session not in state
    if session_id not in st.session_state.history_store:
        history = InMemoryChatMessageHistory()
        
        # Load previous messages from file storage
        stored_msg = load_chats(session_id)
        history.add_messages(to_langchain_messages(stored_msg))
        st.session_state.history_store[session_id] = history
    
    return st.session_state.history_store[session_id]
    
def save_chat_history(session_id, role, content, file_path="./chats/chat.json"):
    """
    Save a message to the persistent chat history file.
    
    Args:
        session_id: Unique identifier for the chat session
        role: Message role ('user' or 'assistant')
        content: Message content text
        file_path: Path to the JSON file for storing chat history
    """
    # Load existing chat history
    chats = load_chat_history()
    
    # Find existing session and append message
    for session in chats:
        if session["session_id"] == session_id:
            session["messages"].append({"role": role, "content": content})
            break
    else:
        # Create new session if not found
        chats.append({"session_id": session_id, "messages": [{"role": role, "content": content}]})
    
    # Write updated history to file
    with open(file_path, "w") as f:
        json.dump(chats, f, indent=4)
    
# ============================================================================
# STREAMLIT UI - HEADER AND FILE UPLOAD
# ============================================================================
st.header("Document Exploration with PDF Uploads")
st.text("Upload PDF files and ask questions about their content.")

# File uploader widget for PDF selection
uploaded_files = st.file_uploader("Upload PDF files", type="pdf")

# Quick action buttons for common operations
st.subheader("Quick Actions")
col1, col2, col3 = st.columns(3)
prompt = None  # Will be set if any quick action button is clicked

# ============================================================================
# PDF PROCESSING AND VECTOR STORE MANAGEMENT
# ============================================================================
if uploaded_files:
    # Save uploaded PDF to temporary location for processing
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_files.getbuffer())
    
    # Generate hash of PDF content for vector store caching
    file_bytes = uploaded_files.getbuffer()
    file_hash = get_pdf_hash(file_bytes)
    vector_db_path = os.path.join(VECTOR_DB_DIR, f"{file_hash}_chroma")

    # Initialize embeddings model for vectorization
    embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

    # Load cached vector store if it exists, otherwise create new one
    if os.path.exists(vector_db_path):
        # Use cached embeddings from SQLite-backed Chroma
        vectorstore = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
    else:
        # Process new PDF: load, split, embed, and store
        loader = PyPDFLoader("temp.pdf")
        data = loader.load()  # Load PDF pages
        
        # Split documents into chunks for semantic search
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(data)
        
        # Create vector store from document chunks and persist to disk
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=vector_db_path
        )
    
    # ====================================================================
    # QUICK ACTION BUTTONS
    # ====================================================================
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
    
    # ====================================================================
    # LLM MODEL INITIALIZATION
    # ====================================================================
    # Initialize Groq LLM (not currently used, but available for alternative models)
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=st.secrets["GROQ_API_KEY"],
        temperature=0
    )

    # Initialize Ollama LLM as the primary model for RAG
    llama = ChatOllama(
        model="llama3.1:8b",
        temperature=0,
        verbose=True
    )
    
    # ====================================================================
    # CHAT HISTORY DISPLAY
    # ====================================================================
    # Display all previous messages from current session
    for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"]) 


    # ====================================================================
    # CHAT INPUT AND RAG PROCESSING
    # ====================================================================
    if prompt or (user_input := st.chat_input("Enter your question about the document")):
        # Use quick action prompt or user input
        input_text = prompt if prompt else user_input
        
        # Build RAG prompt template with system instructions and context placeholder
        prompt_chat = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant. Answer the question using only the information from the context below.\n\nContext:\n{context}"
                ),
                ("user", "{question}")
            ]
        )
        
        # Build RAG chain: extracts question, retrieves context, passes through prompt and LLM
        rag_chain = (
            {
                "question": RunnableLambda(lambda x: x["question"]),
                "context": RunnableLambda(lambda x: build_full_context(session_id, x["question"])),
            } |
            prompt_chat |
            llm
        )
        
        # Wrap RAG chain with chat history management
        rag_with_history = RunnableWithMessageHistory(
            rag_chain,
            get_chat_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(input_text)
            st.session_state.messages.append({"role": "user", "content": input_text})
            save_chat_history(session_id, "user", input_text)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            response = rag_with_history.invoke(
                {"question": input_text},
                config={"configurable": {"session_id": session_id}}
            )
            st.session_state.messages.append({"role": "assistant", "content": response.content})
            save_chat_history(session_id, "assistant", response.content)
            st.rerun()
    
# Display message when no PDF is uploaded
else:
    st.write("Please Upload PDF File")