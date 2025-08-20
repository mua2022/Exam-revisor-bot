import sys
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
from dotenv import load_dotenv
import streamlit as st
from pathlib import Path
from ingest import load_docs_from_folder, load_docs_from_urls
from build_index import build_chroma_index
from rag_chain import build_chain

# Load environment variables
load_dotenv()

# ==============================
# STREAMLIT PAGE CONFIG
# ==============================
st.set_page_config(page_title="DocGenius RAG", page_icon="📚", layout="wide")

st.title("📚 DocGenius RAG Chatbot")
st.caption("Document Analyzer Project – End-to-end RAG with LangChain + FAISS")

# ==============================
# INITIALIZE SESSION STATE
# ==============================
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # stores (role, content) tuples

# ==============================
# SIDEBAR – INGESTION + HISTORY
# ==============================
with st.sidebar:
    st.header("⚡ Ingestion")
    uploaded = st.file_uploader("Upload PDFs or text files", accept_multiple_files=True, type=["pdf", "txt", "md"])
    urls = st.text_area("Web URLs (one per line)").strip().splitlines()

    if st.button("🔄 Build / Rebuild Index"):
        data_dir = Path("data/source")
        data_dir.mkdir(parents=True, exist_ok=True)

        # Save uploads locally
        for f in uploaded or []:
            (data_dir / f.name).write_bytes(f.read())

        # Load docs
        docs = load_docs_from_folder(str(data_dir)) + load_docs_from_urls([u for u in urls if u])
        n = build_faiss_index(docs)
        st.success(f"✅ Indexed {n} chunks successfully")

    st.markdown("---")

    # Clear chat history button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat history cleared!")

# ==============================
# DOCUMENT UPLOAD
# ==============================
uploaded_file = st.file_uploader("📂 Upload a PDF for analysis", type="pdf")

if uploaded_file is not None:
    with open("uploaded_doc.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.chain = build_chain("uploaded_doc.pdf")
    st.success("✅ Document processed and chain initialized!")
# ==============================
# CHAT INTERFACE
# ==============================
st.subheader("💬 Chat with your documents")

# Bubble styles
bubble_style = """
<style>
.user-bubble {
    background-color: #DCF8C6; /* WhatsApp-like green */
    color: black;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 5px 0;
    text-align: right;
    max-width: 70%;
    margin-left: auto;
    box-shadow: 1px 1px 5px rgba(0,0,0,0.1);
}
.bot-bubble {
    background-color: #E4E6EB; /* light grey */
    color: black;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 5px 0;
    text-align: left;
    max-width: 70%;
    margin-right: auto;
    box-shadow: 1px 1px 5px rgba(0,0,0,0.1);
}
</style>
"""
st.markdown(bubble_style, unsafe_allow_html=True)

# Display chat history
for role, content in st.session_state.messages:
    if role == "user":
        st.markdown(f"<div class='user-bubble'>👤 {content}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>🤖 {content}</div>", unsafe_allow_html=True)

# Use chat input (auto-clears)
if st.session_state.chain:
    query = st.chat_input("Ask a question about the document...")
    if query:
        # Save user query
        st.session_state.messages.append(("user", query))

        # Generate answer
        answer = st.session_state.chain(query)

        # Save bot response
        st.session_state.messages.append(("assistant", answer))

        # Refresh to show new messages instantly
        st.experimental_rerun()
else:
    st.info("⚠️ Please upload and process a document first to start chatting.")


# ==============================
# FOOTER
# ==============================
footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #0E1117;
    color: white;
    text-align: center;
    padding: 10px;
    font-size: 14px;
}
.footer a {
    color: #00BFFF;
    margin: 0 10px;
    text-decoration: none;
}
.footer a:hover {
    color: #1E90FF;
}
</style>

<div class="footer">
    <p>DocGenius © 2025 | Developed with ❤️ by <b>Mua Emmanuel @Mua254</b> </p>
    <p>
        <a href="https://github.com/mua2022" target="_blank"><i class="fa fa-github"></i></a>
        <a href="https://www.linkedin.com/in/mua-emmanuel-a9a134278/" target="_blank"><i class="fa fa-linkedin"></i></a>
        <a href="https://x.com/AnnuitCoep88551" target="_blank"><i class="fa fa-twitter"></i></a>
    </p>
</div>

<!-- Load Font Awesome -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
"""
st.markdown(footer, unsafe_allow_html=True)
