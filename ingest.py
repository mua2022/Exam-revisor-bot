import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP =int(os.getenv("CHUNK_OVERLAP",120))

def load_docs_from_folder(folder: str):
    folder_path  =Path(folder)
    docs =[]
    for p in folder_path.rglob("*"):
        if p.suffix.lower()  == ".pdf":
            docs.extend(PyPDFLoader(str(p)).load())
        elif p.suffix.lower()  in {".txt", ".md"}:
            docs.extend(TextLoader(str(p), encoding="utf-8").load())

    return docs

def load_docs_from_urls(urls: list[str]):
    if not urls:
        return []
    loader = WebBaseLoader(urls)
    return loader.load()


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", ".", " "]
    )
    return splitter.split_documents(documents)


if __name__ == "__main__":
    # Quick manual test
    folder_docs = load_docs_from_folder("data/source")
    web_docs = load_docs_from_urls(["https://langchain.readthedocs.io/"])
    chunks = chunk_documents(folder_docs + web_docs)
    print(f"Loaded {len(folder_docs)+len(web_docs)} docs -> {len(chunks)} chunks")                         
                                       
