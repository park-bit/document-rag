"""
Utility functions for document processing.
Supports both text extraction and OCR for scanned documents.
"""
import os
from PyPDF2 import PdfReader
import pytesseract
from pdf2image import convert_from_path
from PIL import Image


def load_text(file_path: str, use_ocr: bool = False) -> str:
    """
    Load text from a file (PDF or TXT).
    Supports OCR for scanned PDFs.
    
    Args:
        file_path: Path to the file
        use_ocr: If True, use OCR for PDF extraction (slower but works on scanned PDFs)
        
    Returns:
        Extracted text from the file
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    
    elif file_extension == ".pdf":
        if not use_ocr:
            extracted_text = _extract_text_from_pdf(file_path)
            if len(extracted_text.strip()) < 50:
                print("Standard extraction yielded minimal text. Attempting OCR...")
                return _extract_text_with_ocr(file_path)
            return extracted_text
        else:
            return _extract_text_with_ocr(file_path)
    
    return ""


def _extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from PDF using PyPDF2 (works on searchable PDFs).
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text
    """
    try:
        pdf_reader = PdfReader(file_path)
        extracted_text = ""
        
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_text += page_text + "\n"
        
        return extracted_text
    except Exception as e:
        print(f"Error during standard PDF extraction: {e}")
        return ""


def _extract_text_with_ocr(file_path: str) -> str:
    """
    Extract text from PDF using OCR (works on scanned PDFs).
    This is slower but handles scanned documents.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text via OCR
    """
    try:
        print("Starting OCR extraction (this may take a while)...")
        
        images = convert_from_path(file_path)
        extracted_text = ""
        
        for idx, image in enumerate(images):
            print(f"Processing page {idx + 1}/{len(images)}...")
            page_text = pytesseract.image_to_string(image)
            if page_text:
                extracted_text += page_text + "\n"
        
        print("OCR extraction completed")
        return extracted_text
        
    except Exception as e:
        print(f"Error during OCR extraction: {e}")
        print("Make sure Tesseract OCR is installed on your system.")
        return ""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Split text into overlapping chunks.
    
    Args:
        text: The text to chunk
        chunk_size: Size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text or len(text.strip()) == 0:
        return []
    
    chunks = []
    pos = 0
    
    while pos < len(text):
        chunk_end = pos + chunk_size
        chunks.append(text[pos:chunk_end])
        pos = chunk_end - overlap
    
    return chunks
