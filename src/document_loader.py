import pymupdf4llm


def extract_content(pdf_path: str) -> str:
    """
    Extract raw text from a PDF and return it as Markdown string.
    """
    return pymupdf4llm.to_markdown(pdf_path)