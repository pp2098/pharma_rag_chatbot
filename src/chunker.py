import nltk
from keybert import KeyBERT

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

PHARMA_KEYWORDS = [
    "sitagliptin", "metformin", "DPP-4", "diabetes",
    "insulin", "HbA1c", "tablet", "formulation",
    "dissolution", "bioavailability", "excipient",
    "pharmacokinetics", "antidiabetic", "glucose",
    "pancreas", "biguanide", "phosphate", "monohydrate",
    "granulation", "compression", "coating", "hardness",
    "friability", "disintegration", "assay", "enzyme",
    "inhibitor", "glycemic", "hypoglycemic", "plasma",
    "absorption", "renal", "hepatic", "clinical",
    "efficacy", "safety", "mechanism", "pathway",
    "receptor", "therapeutic", "drug", "JAK", "STAT",
    "cardiomyopathy", "cardiac", "oxidative", "stress",
    "inflammation", "cytokine", "hyperglycemia"
]


def chunk_text(text: str, max_chunk_size: int = 400) -> list[dict]:
    """
    Split text into sentence-aware chunks of roughly max_chunk_size characters.
    Returns a list of dicts with 'chunk_id' and 'text'.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    chunk_id = 1

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_chunk_size:
            if current_chunk.strip():
                chunks.append({"chunk_id": chunk_id, "text": current_chunk.strip()})
                chunk_id += 1
            current_chunk = sentence + " "
        else:
            current_chunk += sentence + " "

    if current_chunk.strip():
        chunks.append({"chunk_id": chunk_id, "text": current_chunk.strip()})

    return chunks


def extract_keywords(chunks: list[dict], kw_model: KeyBERT) -> list[dict]:
    """
    Tag each chunk with auto-extracted keywords (KeyBERT) merged with
    domain-specific PHARMA_KEYWORDS found in the chunk text.
    Returns the same list of chunks with a 'keywords' key added.
    """
    tagged_chunks = []

    for chunk in chunks:
        auto_keywords = kw_model.extract_keywords(
            chunk["text"],
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=5
        )
        auto_kw_list = [kw[0] for kw in auto_keywords]
        manual_kw_list = [
            kw for kw in PHARMA_KEYWORDS
            if kw.lower() in chunk["text"].lower()
        ]
        chunk["keywords"] = list(set(auto_kw_list + manual_kw_list))
        tagged_chunks.append(chunk)

    return tagged_chunks