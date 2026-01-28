import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(DATA_DIR, "index")

INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
TEXT_PATH = os.path.join(INDEX_DIR, "chunks.pkl")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
