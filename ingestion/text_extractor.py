"""
text_extractor.py

Implements text extraction from PDF files using PyMuPDF (fitz).
"""

import fitz
import io
import os
from typing import Optional

# Enable python implementation fallback for protobuf to allow PaddleOCR execution under protobuf 5.x
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


_paddle_ocr = None
_easy_ocr = None

def _ocr_page(page: fitz.Page) -> str:
    """
    Renders a PDF page to an image pixmap and performs OCR.
    Tries PaddleOCR first; if unavailable or fails on C++ backend, falls back to EasyOCR.
    """
    global _paddle_ocr, _easy_ocr
    
    try:
        from PIL import Image
        import numpy as np

        # Render page at 200 DPI for high OCR accuracy
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(image)
    except Exception as e:
        print(f"Warning: Failed to render page {page.number + 1} to image: {e}")
        return ""

    # Engine 1: Try PaddleOCR
    try:
        if _paddle_ocr is None:
            from paddleocr import PaddleOCR
            _paddle_ocr = PaddleOCR(use_textline_orientation=True, lang='en')
            
        result = _paddle_ocr.ocr(img_np)
        extracted_lines = []
        if result and isinstance(result, list):
            for res_block in result:
                if res_block and isinstance(res_block, list):
                    for line in res_block:
                        if line and len(line) >= 2 and line[1] and line[1][0]:
                            extracted_lines.append(line[1][0].strip())
        if extracted_lines:
            return "\n".join(extracted_lines)
    except Exception as e:
        print(f"PaddleOCR notice on page {page.number + 1}: {e}. Trying EasyOCR fallback...")

    # Engine 2: Fallback to EasyOCR
    try:
        if _easy_ocr is None:
            import easyocr
            _easy_ocr = easyocr.Reader(['en'], gpu=False, verbose=False)
            
        results = _easy_ocr.readtext(img_np)
        extracted_lines = [res[1].strip() for res in results if res and len(res) >= 2 and res[1]]
        if extracted_lines:
            return "\n".join(extracted_lines)
    except Exception as e:
        print(f"Warning: EasyOCR fallback failed for page {page.number + 1}: {e}")


    return ""



def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts complete text from a single PDF document.
    Iterates page-by-page: if a page has fewer than 50 characters,
    it falls back to PaddleOCR to extract text from scanned content.

    Args:
        pdf_path: The absolute or relative path to the PDF file.

    Returns:
        The extracted text as a string.
    """
    text_content = []
    
    # Open the PDF using fitz (PyMuPDF)
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text() or ""
            
            # If the extracted text on this page has fewer than 50 characters, run PaddleOCR
            if len(page_text.strip()) < 50:
                print(f"Page {page_num}: Low character count ({len(page_text.strip())} chars). Running PaddleOCR...")
                ocr_text = _ocr_page(page)
                if ocr_text.strip():
                    page_text = ocr_text
                    
            if page_text.strip():
                text_content.append(page_text.strip())
                
    return "\n".join(text_content)

