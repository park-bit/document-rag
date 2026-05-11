import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging

logger = logging.getLogger("model")

# Use a small, efficient model that can run on CPU
MODEL_ID = "google/flan-t5-large"

import threading

_tokenizer = None
_model = None
_load_lock = threading.Lock()

def load_model():
    global _tokenizer, _model
    with _load_lock:
        if _model is None:
            logger.info(f"Loading local model: {MODEL_ID} in FP16...")
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            # Use float16 for massive speedup on GPU
            _model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_ID, 
                torch_dtype=torch.float16,
                device_map="auto"
            )
            logger.info(f"Model loaded on {next(_model.parameters()).device}")
    return _tokenizer, _model

def generate_local(prompt: str, max_new_tokens: int = 512) -> str:
    """
    Generate text using a local HuggingFace model.
    """
    import time
    start = time.time()
    try:
        tokenizer, model = load_model()
        device = next(model.parameters()).device
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True,
                top_p=0.9
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generation complete in {time.time() - start:.2f}s")
        return response.strip()
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return f"Error in local generation: {str(e)}"
