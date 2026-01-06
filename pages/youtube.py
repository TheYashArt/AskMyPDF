import streamlit as st
import os

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain # Fixed import

st.header("YouTube Video Summarizer with Groq LLM")
video_url = st.text_input("Enter YouTube Video URL", key="youtube_url_input")
from pytubefix import YouTube

if video_url:
        with st.spinner("Loading video and extracting transcript..."):
            # Added 'en-IN' for Indian tech videos
            yt = YouTube(video_url)
            # This accesses the caption tracks directly from the YouTube object
            captions = yt.captions.get('en') or yt.captions.get('a.en') or yt.captions.get('en-IN')
            srt_text = captions.generate_srt_captions()
            docs = [Document(page_content=srt_text)]
            if not docs:
                st.error("No transcript found. This video might have captions disabled or restricted.")
            else:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                chunks = text_splitter.split_documents(docs)
        # Move LLM setup inside the conditional to save resources
            llm = ChatGoogleGenerativeAI(
                model="gemini-3-flash-preview",
                api_key=st.secrets["GEMINI_API_KEY"],
                temperature=0,
                max_tokens=None,
                Timeout=None,
                max_retries=3
            )
            chain = load_summarize_chain(llm, chain_type="stuff")
        
        if st.button("Summarize Video"):
            with st.spinner("Generating summary..."):
                summary = chain.invoke(chunks)
                st.subheader("Video Summary")
                st.write(summary['output_text'])
        