import streamlit as st
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.summarize import load_summarize_chain # Fixed import

st.header("YouTube Video Summarizer")
video_url = st.text_input("Enter YouTube Video URL", key="youtube_url_input")
from pytubefix import YouTube

if video_url:
        with st.spinner("Loading video and extracting transcript..."):
            # Added 'en-IN' for Indian tech videos
            yt = YouTube(video_url)
            yt.bypass_age_gate()
            # This accesses the caption tracks directly from the YouTube object
            caption_track = yt.captions.get('en') or yt.captions.get('a.en') or yt.captions.get('en-IN')
            if caption_track:
                srt_text = caption_track.generate_srt_captions()
            else:
                from youtube_transcript_api import YouTubeTranscriptApi
                video_id = yt.video_id
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                srt_text = " ".join([t['text'] for t in transcript])
                
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
                summary = chain.invoke(chunks)
                st.subheader("Video Summary")
                st.write(summary['output_text'])
        