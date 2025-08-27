# rag/ingest.py
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def ingest_pdfs(
    file_paths: List[str],
    persist_directory: str = ".chroma/student-rag",
    collection_name: str = "pdf-chat"
):
    # 1) Load pages from all PDFs
    docs = []
    for p in file_paths:
        loader = PyPDFLoader(p)
        docs.extend(loader.load())

    # 2) Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, add_start_index=True
    )
    chunks = splitter.split_documents(docs)

    # 3) Embed + store
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    os.makedirs(persist_directory, exist_ok=True)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
