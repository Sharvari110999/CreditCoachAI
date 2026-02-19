"""RAG (Retrieval Augmented Generation) module for the Credit System."""

import os
import glob
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.creditsystem.config import (
    EMBEDDING_MODEL,
    DB_NAME,
    RETRIEVAL_K_EXPLANATION,
    RETRIEVAL_K_SIMULATION,
    RETRIEVAL_K_DEFAULT,
)


# Initialize embeddings globally (lazy initialization)
_embeddings = None
_vectorstore = None


def get_embeddings():
    """Get or create the embeddings model."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def get_vectorstore():
    """Get or create the vector store."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=DB_NAME,
            embedding_function=get_embeddings()
        )
    return _vectorstore


def get_retriever():
    """Get the retriever from the vector store."""
    return get_vectorstore().as_retriever()


def load_documents(data_dir: str = "data"):
    """Load all markdown files from the data folder."""
    markdown_files = glob.glob(f"{data_dir}/*.md")
    
    documents = []
    for file_path in markdown_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            documents.append({"content": content, "source": file_path})
    
    print(f"Loaded {len(documents)} markdown files from data folder")
    return documents


def create_chunks(documents: list, chunk_size: int = 500, chunk_overlap: int = 150):
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = []
    for doc in documents:
        doc_chunks = text_splitter.create_documents(
            texts=[doc["content"]],
            metadatas=[{"source": doc["source"]}]
        )
        chunks.extend(doc_chunks)
    
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def initialize_vectorstore(data_dir: str = "data", chunk_size: int = 500, chunk_overlap: int = 150):
    """Initialize the vector store with documents from the data folder."""
    # Delete existing DB if it exists
    if os.path.exists(DB_NAME):
        shutil.rmtree(DB_NAME)
        print("Old DB deleted.")
    else:
        print("No existing DB found.")
    
    # Load documents
    documents = load_documents(data_dir)
    
    # Create chunks
    chunks = create_chunks(documents, chunk_size, chunk_overlap)
    
    # Create vector store
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_NAME
    )
    
    global _vectorstore
    _vectorstore = vectorstore
    
    print("New DB created successfully.")
    return vectorstore


def retrieve_context(question: str, intent: str = None) -> tuple[str, float]:
    """
    Retrieve context for a question with confidence scoring.
    
    Args:
        question: The question to retrieve context for
        intent: The intent category (affects k value)
    
    Returns:
        Tuple of (context_string, confidence_score)
    """
    # Determine k based on intent
    if intent == "simulation":
        k = RETRIEVAL_K_SIMULATION
    elif intent == "explanation":
        k = RETRIEVAL_K_EXPLANATION
    else:
        k = RETRIEVAL_K_DEFAULT
    
    # Get docs with similarity scores
    vectorstore = get_vectorstore()
    docs_with_scores = vectorstore.similarity_search_with_score(question, k=k)

    docs = [doc for doc, score in docs_with_scores]
    scores = [score for doc, score in docs_with_scores]
    
    context = "\n\n".join([d.page_content for d in docs])
    
    # Confidence = inverse of distance (Chroma returns distance, lower is better)
    # Use exponential decay to convert distance to confidence (handles distances > 1)
    if scores:
        distance = scores[0]
        confidence = 1 / (1 + distance)  # Exponential decay: 1/(1+distance)
    else:
        confidence = 0
    
    return context, confidence


def simple_retrieve(question: str, k: int = 4) -> str:
    """
    Simple retrieval without confidence scoring.
    
    Args:
        question: The question to retrieve context for
        k: Number of documents to retrieve
    
    Returns:
        Context string
    """
    docs = get_retriever().invoke(question)
    context = "\n\n".join([d.page_content for d in docs])
    return context
