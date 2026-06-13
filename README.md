# Pharma RAG Chatbot

A RAG-based chatbot that answers questions from pharmaceutical documents (PDFs + scanned images) related to Cetagliptin production.

## Tech Stack
- LLM: Mistral 7B (Hugging Face)
- Embeddings: sentence-transformers
- Vector Store: FAISS
- OCR: pytesseract + PyMuPDF
- UI: Streamlit

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```