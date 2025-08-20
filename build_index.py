import os
from dotenv import load_dotenv
from pathlib import Path
from ingest import load_docs_from_folder, load_docs_from_urls, chunk_documents
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()

EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
INDEX_DIR = Path("data/chroma_index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)


def build_chroma_index(docs):
    chunks = chunk_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

    # Create or load persistent Chroma DB
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(INDEX_DIR)
    )

    vectordb.persist()  # ensures data is saved
    return len(chunks)


if __name__ == "__main__":
    # Configure your sources here (or call via Streamlit/app)
    folder_docs = load_docs_from_folder("data/source")
    web_docs = load_docs_from_urls([])  # add URLs if needed
    total_chunks = build_chroma_index(folder_docs + web_docs)
    print(f"Indexed {total_chunks} chunks -> {INDEX_DIR}")
