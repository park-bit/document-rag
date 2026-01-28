# Document Question-Answering System

A lightweight RAG (Retrieval-Augmented Generation) based question answering system that allows users to upload documents and ask questions about their content made as an internship project submission. It's a different version of my [previous repo](https://github.com/park-bit/llm-rag-document-qna).

## Features

- **Text Extraction**: Automatically extracts text from PDF and TXT files
- **Scanned PDF Support**: Uses OCR (Tesseract) to handle scanned or image-based PDFs
- **Semantic Search**: Leverages sentence transformers for semantic similarity search
- **Local LLM**: Uses DistilBERT fine-tuned on SQuAD for efficient question answering
- **Rate Limiting**: Built-in rate limiting (5 requests/minute per IP)
- **Async Processing**: Background task processing for document ingestion

## How It Works

1. **Document Upload**: Users upload PDF or TXT files
2. **Text Extraction**: The system extracts text (with automatic OCR fallback for scanned PDFs)
3. **Chunking**: Text is split into overlapping chunks (500 characters with 50-char overlap)
4. **Embedding**: Each chunk is converted to a vector embedding using SentenceTransformer
5. **Indexing**: Embeddings are stored in FAISS for fast similarity search
6. **Question Answering**: When a question is asked, it's converted to an embedding and similar chunks are retrieved; then a QA model generates the answer

## Technical Decisions

### Chunk Size Optimization
- Tested chunk sizes of 300, 500, and 800 characters
- Smaller chunks (300) often lost important context
- Larger chunks (800) reduced retrieval precision
- **Optimal choice**: 500 characters with 50-character overlap for balance

### Hybrid Text Extraction
- Standard PDF extraction (PyPDF2) works well for searchable PDFs
- Automatically falls back to OCR if extraction yields minimal text
- OCR uses Tesseract for scanned documents

### Known Limitations
- Documents with complex layouts (tables, lists) may not chunk optimally
- OCR is slower than standard text extraction
- LLM context is limited to top 3 retrieved chunks

### Performance Metrics
- End-to-end query latency: ~2-5 seconds (depending on system)
- Retrieval time: ~0.5-1 second
- QA generation time: ~1-4 seconds

## Installation

### Prerequisites
- Python
- Tesseract OCR (for scanned PDF support)

### Installing Tesseract on Windows
1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer (default installation path is fine)

### Python Dependencies
```bash
pip install -r requirements.txt
```

## Running the Application

```bash
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000`

### Endpoints

**Health Check**
```
GET /
```

**Upload Document**
```
POST /upload
- file: PDF or TXT file
```

**Query Document**
```
POST /query?question=<your_question>
- Limit: 5 requests per minute
```

### Using the API

1. Open `http://127.0.0.1:8000/docs` in your browser
2. Click "Try it out" on the `/upload` endpoint
3. Upload a PDF or TXT file
4. Once uploaded, use the `/query` endpoint to ask questions
5. Get answers based on the document content

## Example Usage

```bash
# Upload a document
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "file=@your_document.pdf"

# Ask a question
curl "http://127.0.0.1:8000/query?question=What%20is%20this%20document%20about"
```

## Project Background

Initially developed as an exploration of RAG systems with OpenAI APIs. This version focuses on local execution without external LLM dependencies, emphasizing understanding of core concepts: document chunking, embeddings, retrieval, and question answering.
The original previous repo can be found on the profile page or  [this link](https://github.com/park-bit/llm-rag-document-qna)
