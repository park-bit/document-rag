"""
Document ingestion pipeline.
Handles text extraction, chunking, embedding, and FAISS indexing.
"""
from sentence_transformers import SentenceTransformer
from utils import load_text, chunk_text
from config import INDEX_PATH, TEXT_PATH, INDEX_DIR
import faiss
import numpy as np
import os
import pickle

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

os.makedirs(INDEX_DIR, exist_ok=True)


def ingest_document(file_path: str) -> None:
    """
    Ingest a document into the vector index.
    
    Args:
        file_path: Path to the document file (PDF or TXT)
    """
    extracted_text = load_text(file_path)
    print(f"Extracted {len(extracted_text)} characters")
    if extracted_text:
        print(f"First 300 chars: {extracted_text[:300]}")
    
    text_chunks = chunk_text(extracted_text)
    if not text_chunks:
        print("Warning: No text chunks generated. Skipping indexing.")
        return
    
    print(f"Created {len(text_chunks)} chunks")
    
    chunk_embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True)
    
    if chunk_embeddings.ndim == 1:
        chunk_embeddings = chunk_embeddings.reshape(1, -1)
    
    embedding_dim = chunk_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    faiss_index.add(chunk_embeddings)
    
    with open(TEXT_PATH, "wb") as f:
        pickle.dump(text_chunks, f)
    
    faiss.write_index(faiss_index, INDEX_PATH)
    
    print(f"Successfully indexed {len(text_chunks)} chunks at {INDEX_PATH}")
