import streamlit as st
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.summarize import load_summarize_chain # Fixed import

st.header("YouTube Video Summarizer")
video_url = st.text_input("Enter YouTube Video URL", key="youtube_url_input")
from pytubefix import YouTube

preferred_codes = [
    "en",
    "a.en",
    "en-IN",
    "a.en-IN",
    "hi",
    "a.hi"
]
srt_text = None
if video_url:
        with st.spinner("Loading video and extracting transcript..."):
            try:
                yt = YouTube(video_url)
                captions = (
                    yt.captions.get("en")
                    or yt.captions.get("en-US")
                    or yt.captions.get("a.en-US")
                    or yt.captions.get("a.en")
                    or yt.captions.get("en-IN")
                    or yt.captions.get("a.en-IN")
                    or yt.captions.get("hi")
                    or yt.captions.get("a.hi")
                )
                srt_text = captions.generate_srt_captions()
                docs = [Document(page_content=srt_text)]
                if not docs:
                    st.write("No transcript available for this video.")
                else:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200, length_function=len)
                    chunks = text_splitter.split_documents(docs)
            except AttributeError as e:
                st.write("Error retrieving transcript. Please ensure the video has captions available.")
                chunks = []
                
            llm = ChatGoogleGenerativeAI(
                model="gemini-3-flash-preview",
                api_key=st.secrets["GEMINI_API_KEY"],
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=3
            )
            chain = load_summarize_chain(llm, chain_type="stuff")
        
        if st.button("Summarize Video", icon ="📝"):
            with st.spinner("Generating summary..."):
                summary = chain.run(chunks)
                st.subheader("Video Summary")
                st.write(summary)
        