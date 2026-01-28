"""
Document retrieval and QA pipeline.
Handles semantic search and answer generation.
"""
import faiss
import pickle
import numpy as np
import os
import time
from sentence_transformers import SentenceTransformer
from config import INDEX_PATH, TEXT_PATH
from llm import generate_answer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def answer_question(question: str) -> str:
    """
    Answer a question based on indexed documents.
    
    Args:
        question: The user's question
        
    Returns:
        The generated answer from the QA model
    """
    if not os.path.exists(INDEX_PATH):
        return "No documents have been indexed yet. Please upload a document first."
    
    faiss_index = faiss.read_index(INDEX_PATH)
    with open(TEXT_PATH, "rb") as f:
        stored_chunks = pickle.load(f)
    
    start = time.time()
    
    question_vector = embedding_model.encode([question])
    
    chunk_indices = faiss_index.search(np.array(question_vector), k=5)
    
    relevant_chunks = [stored_chunks[idx] for idx in chunk_indices[0]]
    
    answer_text = generate_answer(question, relevant_chunks)
    
    elapsed = time.time() - start
    print(f"Retrieval + QA time: {elapsed:.2f}s")
    
    return answer_text
