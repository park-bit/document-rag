# DocMeant: Local Neural Document Intelligence

**DocMeant** is a high-performance, private, and local RAG system designed for deep document analysis. It runs entirely on your local hardware (optimized for NVIDIA GPUs) ensuring that no sensitive data ever leaves your machine.

![Version](https://img.shields.io/badge/version-3.0.0-white?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-white?style=flat-square)
![Engine](https://img.shields.io/badge/Engine-PyMuPDF%20|%20FAISS%20|%20Flan--T5--Large-white?style=flat-square)

<img width="1918" height="911" alt="image" src="https://github.com/user-attachments/assets/7864b5ca-9ca9-4ece-a4d4-8370be1ee30c" />


### Key Features
- **Neural Query Console**: Natural language interface to ask complex questions across 50+ page documents.
- **Autonomous Document Scan**: Deep neural identification of metadata, topics, and subjects.(Little buggy, PR the fix if u can).
- **Entity Target Extraction**: Specify target fields for high-precision data extraction.(and this one is buggy too :)
- **Local Neural VRAM**: Fully GPU-accelerated (FP16) using your local RTX hardware.(My laptop was burning)
- **Offline First**: Zero cloud dependencies. Works completely offline using local weights.
```mermaid
flowchart TD

  subgraph Primary_App["Primary App"]
    web_ui["Web UI<br/>Next.js frontend<br/>page.js"]
    web_layout["App shell<br/>Next.js layout<br/>layout.js"]
    api_main["API server<br/>FastAPI entrypoint<br/>main.py"]
    api_schemas["Contracts<br/>Pydantic schemas<br/>schemas.py"]
    api_limits{"Rate limit<br/>request control<br/>rate_limit.py"}

    doc_ocr["Text extraction<br/>OCR + parsing<br/>ocr.py"]
    embedder["Embeddings<br/>sentence-transformers<br/>embeddings.py"]
    retriever["RAG core<br/>retrieval orchestration<br/>rag.py"]
    generator["LLM model<br/>Flan-T5 generator<br/>model.py"]
  end

  subgraph Mirror_Copy["Packaged Mirror"]
    mirror_backend["Mirror backend<br/>packaged FastAPI copy<br/>main.py"]
    mirror_frontend["Mirror UI<br/>packaged Next.js copy<br/>page.js"]
  end

  web_layout -->|"renders"| web_ui
  web_ui -->|"API calls"| api_main

  api_main -->|"validates"| api_schemas
  api_main -->|"guards"| api_limits

  api_main -->|"ingests"| doc_ocr
  doc_ocr -->|"chunks"| embedder
  embedder -->|"indexes"| retriever

  api_main -->|"queries"| retriever
  retriever -->|"context"| generator
  api_main -->|"answers"| generator

  mirror_backend -.->|"mirrors"| api_main
  mirror_frontend -.->|"mirrors"| web_ui

  classDef blue fill:#dbeafe,stroke:#2563eb,stroke-width:1.5px,color:#172554
  classDef amber fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#78350f

  class web_ui,web_layout,api_main,api_schemas,api_limits,doc_ocr,embedder,retriever,generator blue
  class mirror_backend,mirror_frontend amber
```
##  Tech Stack
- **Frontend**: Next.js (React), Vanilla CSS (Noir System)
- **Backend**: FastAPI (Python 3.11)
- **Neural Brain**: 
  - **Generation**: Google Flan-T5-Large (770M)
  - **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Extraction**: PyMuPDF (fitz) & Parallel OCR (Tesseract)

##  Quick Start

### 1. Prerequisites
- **Python 3.11+**
- **Node.js 18+**
- **NVIDIA GPU** (RTX 30 series or higher recommended)
- **CUDA 12.1** installed

### 2. Backend Setup
```bash
# Navigate to backend
cd backend
# Create and activate venv
python -m venv venv
./venv/Scripts/activate
# Install optimized dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# Start the engine
uvicorn main:app --port 8000
```

### 3. Frontend Setup
```bash
# Navigate to web
cd web
# Install dependencies
npm install
# Start the noir interface
npm run dev
```

## Privacy & Security
DocMeant is built with a "Privacy by Design" philosophy. All neural processing, embedding generation, and vector storage happen locally in your VRAM and filesystem. 

## License
Distributed under the MIT License. See `LICENSE` for more information.

---
Built by [park-bit](https://github.com/park-bit) ; [support](https://www.chai4.me/park-bit)
