import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os

# Path to tesseract - update if your install path is different
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file.
    - If PDF has text → directly extract
    - If PDF is scanned → use OCR
    """
    doc = fitz.open(pdf_path)
    full_text = ""

    for page_num, page in enumerate(doc):
        text = page.get_text().strip()

        if text:
            full_text += f"\n--- Page {page_num + 1} ---\n{text}"
        else:
            print(f"Page {page_num + 1} appears scanned, applying OCR...")
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            ocr_text = pytesseract.image_to_string(img)
            full_text += f"\n--- Page {page_num + 1} (OCR) ---\n{ocr_text}"

    doc.close()
    return full_text


def load_all_documents(data_folder: str) -> list:
    """
    Loads all PDFs from the data folder.
    Returns a list of dicts with filename + extracted text.
    """
    documents = []

    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(data_folder, filename)
            print(f"Processing: {filename}")
            text = extract_text_from_pdf(pdf_path)
            documents.append({
                "filename": filename,
                "text": text
            })
            print(f"✅ Done: {filename} — {len(text)} characters extracted")

    return documents


# Quick test
if __name__ == "__main__":
    docs = load_all_documents("data")
    for doc in docs:
        print(f"\nFile: {doc['filename']}")
        print(f"Preview: {doc['text'][:300]}")