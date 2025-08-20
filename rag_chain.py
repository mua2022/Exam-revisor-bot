import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

# === Load environment variables === #
load_dotenv()  # Local dev

def get_secret(key: str, default=None):
    """Get secret from Streamlit Cloud or .env/local environment."""
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

# === Configuration === #
GROQ_API_KEY = get_secret("GROQ_API_KEY")
GROQ_MODEL = get_secret("GROQ_MODEL", "llama3-70b-8192")
EMBEDDINGS_MODEL = get_secret("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(get_secret("TOP_K", 6))
CHUNK_SIZE = int(get_secret("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(get_secret("CHUNK_OVERLAP", 120))

# === Initialize Embeddings === #
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

# === LLM Wrapper for Groq === #
class GroqLLM:
    def __init__(self, model=GROQ_MODEL):
        if not GROQ_API_KEY:
            raise ValueError("❌ Missing GROQ_API_KEY. Please set it in .env or Streamlit secrets.")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = model

    def __call__(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

# === Create Vectorstore === #
def create_vectorstore(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="./chroma_db"
    )
    return vectordb

# === Build RAG Chain === #
def build_chain(pdf_path: str):
    vectordb = create_vectorstore(pdf_path)
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})
    llm = GroqLLM(model=GROQ_MODEL)

    def rag_pipeline(query: str) -> str:
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""You are a helpful assistant. 
Use the following context to answer the question accurately.

Context:
{context}

Question: {query}

Answer:"""

        return llm(prompt)

    return rag_pipeline
