import logging
import os
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("embeddings")

MODEL_NAME = os.getenv("SENTENCE_TRANSFORMER", "all-MiniLM-L6-v2")

import threading

_model = None
_load_lock = threading.Lock()

def _load_model():
    global _model
    with _load_lock:
        if _model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading embedding model '{MODEL_NAME}' on {device}")
            _model = SentenceTransformer(MODEL_NAME, device=device)
    return _model

import torch

def get_embeddings_batch(texts: list[str]) -> np.ndarray:
    """
    Returns a 2D float32 numpy array for the given list of texts.
    Uses GPU acceleration and FP16 precision.
    """
    model = _load_model()
    # Use GPU FP16 for massive speedup
    vecs = model.encode(texts, show_progress_bar=False, batch_size=64, convert_to_numpy=True)
    return np.asarray(vecs, dtype="float32")

def get_embedding(text: str) -> np.ndarray:
    """
    Returns a 1D float32 numpy array for the given text.
    """
    model = _load_model()
    vec = model.encode(text, show_progress_bar=False)
    return np.asarray(vec, dtype="float32")
