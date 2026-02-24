import fitz  # PyMuPDF

def load_pdf_text(pdf_path: str) -> list[dict]:
    """
    Loads text from a PDF and returns a list of dictionaries with page number and text.
    """
    pages = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages.append({"page": i + 1, "text": text})
    
    if not pages:
        raise ValueError("PDF appears to contain no extractable text (may be scanned/image-based).")
    
    return pages