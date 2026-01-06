import streamlit as st

# 1. Define the pages
pdf_page = st.Page("pages/PDF_Exploration.py", title="PDF Chat", icon="📄")
yt_page = st.Page("pages/Youtube_Summarizer.py", title="YouTube Summary", icon="🎥")

# 2. Setup Navigation
# Only the pages in this list will show up in the sidebar
pg = st.navigation([pdf_page, yt_page])

# 3. Set page config (Optional but recommended)
st.set_page_config(page_title="Ask My AI", page_icon="🤖", layout="wide")

# 4. Run the selected page
pg.run()