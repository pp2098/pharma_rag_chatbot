"""
rag_pipeline.py  —  Phase 4
Embeddings + FAISS store + Retrieval
Phase 5 will add Mistral LLM
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
from dotenv import load_dotenv
load_dotenv()

# ── Config ───────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_DIR       = os.path.join(BASE_DIR, "models")
INDEX_PATH      = os.path.join(FAISS_DIR, "faiss_index.bin")
METADATA_PATH   = os.path.join(FAISS_DIR, "metadata.json")
_model    = None
_index    = None
_metadata = None


# ── Step 1: Load embedding model (once) ──────────────────────────
def _get_model():
    global _model
    if _model is None:
        print(f"⏳ Loading embedding model...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print("✅ Model ready.")
    return _model


# ── Step 2: Convert text to vectors ──────────────────────────────
def _embed(texts):
    embeddings = _get_model().encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=64
    )
    return embeddings.astype(np.float32)


# ── Step 3: Embed chunks and save to disk ────────────────────────
def embed_and_store(tagged_chunks):
    os.makedirs(FAISS_DIR, exist_ok=True)

    texts = [c["text"] for c in tagged_chunks]
    print(f"\n⏳ Embedding {len(texts)} chunks...")
    embeddings = _embed(texts)

    dim   = embeddings.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
    ids   = np.array([c["chunk_id"] for c in tagged_chunks], dtype=np.int64)
    index.add_with_ids(embeddings, ids)

    faiss.write_index(index, INDEX_PATH)
    print(f"✅ FAISS index saved → {INDEX_PATH} ({index.ntotal} vectors)")

    metadata = [
        {
            "chunk_id": c["chunk_id"],
            "text":     c["text"],
            "keywords": c["keywords"]
        }
        for c in tagged_chunks
    ]
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"✅ Metadata saved → {METADATA_PATH}")

    global _index, _metadata
    _index    = index
    _metadata = {m["chunk_id"]: m for m in metadata}
    print("\n🎉 FAISS store is ready!")


# ── Step 4: Load store from disk (use this on chatbot startup) ───

def load_store():
    global _index, _metadata

    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"No saved store found at {INDEX_PATH}. Run embed_and_store() first."
        )

    print("⏳ Loading FAISS index from disk...")
    _index = faiss.read_index(INDEX_PATH)
    print(f"✅ Index loaded ({_index.ntotal} vectors)")

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)
    _metadata = {m["chunk_id"]: m for m in metadata_list}
    print(f"✅ Metadata loaded ({len(_metadata)} chunks)")
    
    return _index, _metadata


# ── Step 5: Retrieve top-k chunks for a query ────────────────────
def retrieve(query, top_k=5):
    if _index is None or _metadata is None:
        raise RuntimeError("Call load_store() first.")

    query_vec            = _embed([query])
    distances, chunk_ids = _index.search(query_vec, top_k)

    results = []
    for dist, cid in zip(distances[0], chunk_ids[0]):
        if cid == -1:
            continue
        chunk          = _metadata[int(cid)].copy()
        chunk["score"] = round(float(dist), 4)
        results.append(chunk)

    return results

from huggingface_hub import InferenceClient

from huggingface_hub import InferenceClient

from huggingface_hub import InferenceClient

def generate_answer(query, context_chunks):
    try:
        import streamlit as st
        HF_TOKEN = st.secrets["HF_TOKEN"].strip()

        print("HF_TOKEN loaded from Streamlit")
        print("Length:", len(HF_TOKEN))
        print("Prefix:", HF_TOKEN[:10])

    except Exception as e:
        print("Streamlit secret error:", e)

        HF_TOKEN = os.getenv("HF_TOKEN")

        print("HF_TOKEN loaded from env")
        print("Length:", len(HF_TOKEN) if HF_TOKEN else 0)

    context = "\n\n".join(
        f"[Chunk {c['chunk_id']}]: {c['text']}"
        for c in context_chunks
    )

    prompt = f"""You are a pharmaceutical research assistant.
Answer the question using ONLY the context provided.
Keep answer professional, crisp and under 5 lines.
If answer not in context, say "I don't have enough information."

Context:
{context}

Question: {query}

Answer:"""

    try:
        response = requests.post(
            "https://router.huggingface.co/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta-llama/Meta-Llama-3-8B-Instruct:together",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a pharmaceutical research assistant. Answer professionally and concisely."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 512,
                "temperature": 0.3
            },
            timeout=30
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"⚠️ Error {response.status_code}: {response.text[:200]}"

    except Exception as e:
        return f"⚠️ Error: {str(e)[:200]}"
    
def ask(query, top_k=5):
    global _index, _metadata
    
    # Load store if not loaded yet
    if _index is None or _metadata is None:
        load_store()
    
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return "No relevant context found."

    print(f"\n🔍 Retrieved {len(chunks)} chunks")
    for c in chunks:
        print(f"   Chunk {c['chunk_id']} | score={c['score']}")

    return generate_answer(query, chunks)


# ── Run this file directly to build the store ────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from analyze_docs import run_pipeline

    print("=" * 60)
    print("Step 1 — Building chunks from PDFs...")
    tagged_chunks = run_pipeline()
    print(f"→ {len(tagged_chunks)} chunks ready")

    print("\nStep 2 — Embedding and storing in FAISS...")
    embed_and_store(tagged_chunks)

    print("\nStep 3 — Loading store...")
    load_store()

    questions = [
        "What is the mechanism of action of sitagliptin?",
        "What is the role of metformin in diabetes treatment?",
        "How does sitagliptin affect the JAK/STAT pathway?",
        "What are the side effects of sitagliptin?",
        "What is DPP-4 inhibitor?",
        "How does sitagliptin help in cardiomyopathy?"
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        answer = ask(question)
        print(f"A: {answer}")