from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
import os
from ingestion import ingest_document
from retrieval import answer_question
from config import UPLOAD_DIR

app = FastAPI(title="Document QA API")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def health_check():
    """Basic health check endpoint"""
    return {"status": "ok"}


@app.post("/upload")
def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a document for ingestion.
    Processes the file in the background.
    """
    dest_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save uploaded file
    with open(dest_path, "wb") as dest_file:
        content = file.file.read()
        dest_file.write(content)
    
    # Process document asynchronously
    background_tasks.add_task(ingest_document, dest_path)
    
    return {
        "message": "File uploaded and queued for processing",
        "filename": file.filename
    }


@app.post("/query")
@limiter.limit("5/minute")
def query_document(request: Request, question: str):
    """
    Query the indexed documents.
    Limited to 5 requests per minute per IP.
    """
    response = answer_question(question)
    return {"answer": response}

