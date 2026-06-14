"""
analyze_docs.py
───────────────
Orchestrator: loads PDFs → cleans text → deduplicates sentences
              → chunks → extracts keywords.

Run directly to inspect pipeline output:
    python analyze_docs.py
"""

import os
import nltk
from keybert import KeyBERT

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

from document_loader import extract_content
from text_cleaner import remove_tables, remove_markdown, find_start_position, find_end_position, remove_noise
from chunker import chunk_text, extract_keywords

# ── Config ───────────────────────────────────────────────────────
DATA_FOLDER = "../data"

SKIP_FILES = [
    "sitigliptan-tabs-ira-20240223.pdf"
]


# ── Pipeline ─────────────────────────────────────────────────────
def run_pipeline(data_folder: str = DATA_FOLDER) -> list[dict]:
    kw_model = KeyBERT()
    merged_text = ""

    pdf_files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDFs\n{'=' * 60}")

    for pdf_file in sorted(pdf_files):          # sorted for deterministic order
        if pdf_file in SKIP_FILES:
            print(f"⏭️  Skipping: {pdf_file}")
            continue

        pdf_path = os.path.join(data_folder, pdf_file)
        print(f"\n📄 Processing: {pdf_file}")

        raw_text = extract_content(pdf_path)
        text = remove_tables(raw_text)
        text = remove_markdown(text)

        start = find_start_position(text)
        end = find_end_position(text)
        text = text[start:end]

        text = remove_noise(text)
        merged_text += text + "\n"
        print(f"✅ Done: {len(text)} characters")

    # ── Deduplicate sentences ─────────────────────────────────────
    print("\n⏳ Deduplicating sentences...")
    all_sentences = sent_tokenize(merged_text)
    seen = set()
    unique_sentences = []

    for sentence in all_sentences:
        cleaned = sentence.strip().lower()
        if cleaned not in seen and len(sentence.split()) >= 5:
            seen.add(cleaned)
            unique_sentences.append(sentence)

    print(f"✅ Unique sentences: {len(unique_sentences)}")

    # ── Chunk ─────────────────────────────────────────────────────
    final_text = " ".join(unique_sentences)
    chunks = chunk_text(final_text)
    print(f"✅ Total chunks: {len(chunks)}")

    # ── Keywords ──────────────────────────────────────────────────
    print("\n⏳ Extracting keywords...")
    tagged_chunks = extract_keywords(chunks, kw_model)
    print(f"✅ Done!\n{'=' * 60}")

    return tagged_chunks


# ── Entrypoint ───────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_pipeline()
    for chunk in results:
        print(f"\nChunk {chunk['chunk_id']}")
        print(f"Keywords: {chunk['keywords']}")
        print(f"Text: {chunk['text']}")
        print("-" * 60)