# backend/main.py
import os
import logging
import hashlib
import time
import re
import json
from fastapi import FastAPI, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from io import BytesIO

from backend.rag import build_vector_store, retrieve_chunks
from backend.model import generate_local
from backend.ocr import extract_text_from_pdf_bytes, ocr_pdf_bytes, clean_ocr_text
from backend.schemas import QueryPayload, FillFormPayload

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(title="DocMeant Backend")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming {request.method} request to {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class DocumentState:
    def __init__(self):
        self.vector_index = None
        self.doc_chunks = []
        self.filename = None
        self.file_hash = None

state = DocumentState()

@app.on_event("startup")
def startup_event():
    logger.info("--- Neural Engine Startup: Initializing Models ---")
    from backend.model import load_model
    from backend.embeddings import _load_model
    load_model()
    _load_model()
    logger.info("--- Neural Engine Ready: GPU Accelerated ---")

@app.get("/")
def root():
    return {"status": "ok", "message": "DocMeant System Active"}

@app.post("/upload")
async def upload(file: UploadFile):
    raw = await file.read()
    filename = (file.filename or "").lower()
    
    current_hash = hashlib.md5(raw).hexdigest()
    if state.file_hash == current_hash:
        logger.info(f"Skipping indexing: File '{filename}' cached.")
        return {"message": "Success (Cached)", "num_chunks": len(state.doc_chunks), "filename": filename}

    start_time = time.time()
    logger.info(f"--- Starting Upload: {filename} ---")
    
    pages = []
    if filename.endswith(".pdf"):
        logger.info(f"Extracting text via PyMuPDF...")
        pages = extract_text_from_pdf_bytes(raw)
        total_chars = sum(len(p.get("text", "")) for p in pages)
        
        if total_chars < 200:
            logger.info("Falling back to OCR...")
            try:
                pages = ocr_pdf_bytes(raw)
            except Exception as e:
                logger.warning(f"OCR failed: {e}")
    else:
        try:
            txt = raw.decode("utf-8", errors="ignore")
            pages = [{"page": 1, "text": txt}]
        except:
            raise HTTPException(status_code=400, detail="Decode failed")

    if not pages:
        raise HTTPException(status_code=400, detail="Empty document")

    for p in pages:
        p["text"] = clean_ocr_text(p.get("text", ""))

    try:
        state.vector_index, state.doc_chunks = build_vector_store(pages)
        state.filename = filename
        state.file_hash = current_hash
        logger.info(f"Indexing successful in {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.exception(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail="Processing error")

    return {"message": "Success", "num_chunks": len(state.doc_chunks), "filename": filename}

@app.post("/query")
def query(payload: QueryPayload):
    if state.vector_index is None:
        raise HTTPException(status_code=400, detail="No document")

    retrieved = retrieve_chunks(state.vector_index, state.doc_chunks, payload.question, payload.top_k)
    context = "\n\n".join([f"[Page {r.get('page')}]: {r['text']}" for r in retrieved])

    prompt = (
        f"Task: Answer the question using the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {payload.question}\n\n"
        "Answer:"
    )
    
    answer = generate_local(prompt)
    return {
        "answer": answer,
        "sources": [{"page": r.get("page"), "excerpt": r["text"][:300] + "..."} for r in retrieved]
    }

@app.post("/analyze/certificate")
def analyze_certificate():
    if state.vector_index is None:
        raise HTTPException(status_code=400, detail="No document")

    query = "document title, subject, summary"
    retrieved = retrieve_chunks(state.vector_index, state.doc_chunks, query, k=5)
    context = "\n\n".join([r["text"] for r in retrieved])

    prompt = (
        "Task: Extract metadata from the text below.\n\n"
        f"Text:\n{context}\n\n"
        "Format your answer EXACTLY as follows:\n"
        "Title: [Document Title]\n"
        "Subject: [Document Subject]\n"
        "Topics: [Key Topics]\n"
        "Summary: [Short Summary]\n\n"
        "Answer:"
    )
    
    raw_response = generate_local(prompt, max_new_tokens=150)
    logger.info(f"Scan Response: {raw_response}")
    
    parsed = {}
    lines = [l.strip() for l in raw_response.split('\n') if l.strip()]
    for line in lines:
        if ":" in line:
            parts = line.split(":", 1)
            parsed[parts[0].strip()] = parts[1].strip()
    
    if not parsed and lines:
        keys = ["Title", "Subject", "Topics", "Summary"]
        for i, l in enumerate(lines[:len(keys)]):
            parsed[keys[i]] = l

    return {"parsed": parsed, "raw": raw_response}

@app.post("/fill-form")
def fill_form(payload: FillFormPayload):
    if state.vector_index is None:
        raise HTTPException(status_code=400, detail="No document")

    retrieved = retrieve_chunks(state.vector_index, state.doc_chunks, " ".join(payload.fields), k=5)
    context = "\n\n".join([r["text"] for r in retrieved])

    fields_str = "\n".join([f"- {f}" for f in payload.fields])
    prompt = (
        "Task: Extract the following fields from the text.\n\n"
        f"Fields:\n{fields_str}\n\n"
        f"Text:\n{context}\n\n"
        "Format: Field Name: Value\n"
        "Answer:"
    )
    
    raw_response = generate_local(prompt, max_new_tokens=200)
    logger.info(f"Fill Form Response: {raw_response}")
    
    result = {}
    lines = raw_response.split('\n')
    for line in lines:
        if ":" in line:
            parts = line.split(":", 1)
            k, v = parts[0].strip().lower(), parts[1].strip()
            for req_f in payload.fields:
                if req_f.lower() in k:
                    result[req_f] = v
                    break
    
    return {"result": result, "raw": raw_response}
