# backend/ocr.py
import re
import os
from io import BytesIO

def clean_ocr_text(text: str) -> str:
    """
    Basic post-processing to improve OCR output quality.
    Drop weird characters, normalize spaces, remove repeated underscores.
    """
    if not text:
        return ""
    text = text.replace("_", " ")
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()

def extract_text_from_pdf_bytes(raw_bytes: bytes):
    """
    Try PyMuPDF (fitz) text extraction first.
    Extremely fast and accurate.
    Returns list of pages dicts: {"page": int, "text": str}
    """
    pages = []
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
        for i, page in enumerate(doc):
            t = page.get_text() or ""
            if t.strip():
                pages.append({"page": i+1, "text": t})
        doc.close()
    except Exception as e:
        print(f"PyMuPDF extraction failed: {e}")
        return []

    return pages

def ocr_pdf_bytes(raw_bytes: bytes, dpi=200, psm=6):
    """
    Uses pdf2image + pytesseract to OCR each page in parallel.
    Returns list of pages dicts: {"page": int, "text": str}
    """
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
        from concurrent.futures import ThreadPoolExecutor
    except Exception as e:
        raise RuntimeError("pdf2image/pytesseract required for OCR: " + str(e))

    # Use local poppler path
    poppler_path = r"d:\llm-rag-document-qna\poppler\Library\bin"
    
    # Lower DPI for faster processing without losing much accuracy for standard docs
    images = convert_from_bytes(raw_bytes, dpi=dpi, poppler_path=poppler_path)
    pages = []
    
    def process_page(args):
        i, img = args
        txt = pytesseract.image_to_string(img, config=f"--psm {psm}")
        txt = clean_ocr_text(txt)
        return {"page": i+1, "text": txt}

    # Use ThreadPoolExecutor to run Tesseract in parallel
    # Tesseract itself is single-threaded per call, so this helps a lot on multi-core
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        results = list(executor.map(process_page, enumerate(images)))
    
    # Filter out empty pages
    pages = [r for r in results if r["text"].strip()]
    return pages

