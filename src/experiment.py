import pymupdf4llm
import re
import os
import nltk
from keybert import KeyBERT

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

data_folder = "../data"

SKIP_FILES = [
    "sitigliptan-tabs-ira-20240223.pdf"
]

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


# ── Extract ──────────────────────────────────────────────────────
def extract_content(pdf_path):
    return pymupdf4llm.to_markdown(pdf_path)


# ── Remove Tables ────────────────────────────────────────────────
def remove_tables(text):
    text = re.sub(r'(\|.*\|[\r\n]*)+', '', text)
    text = re.sub(r'(\|[-:]+\|[\r\n]*)+', '', text)
    return text


# ── Remove Markdown ──────────────────────────────────────────────
def remove_markdown(text):
    text = re.sub(r'#+\s+', '', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'---+', '', text)
    return text


# ── Find Start ───────────────────────────────────────────────────
def find_start_position(text):
    for pattern in [
        r'\bABSTRACT\b', r'\bAbstract\b',
        r'\bINTRODUCTION\b', r'\bIntroduction\b'
    ]:
        match = re.search(pattern, text)
        if match:
            return match.end()
    return 0


# ── Find End → cut at earliest section ──────────────────────────
def find_end_position(text):
    candidates = []
    for pattern in [
        r'\bMaterials\s+and\s+[Mm]ethods\b',
        r'\bMATERIALS\s+AND\s+METHODS\b',
        r'\bMethods\b', r'\bMETHODS\b',
        r'\bResults\b', r'\bRESULTS\b',
        r'\bDiscussion\b', r'\bDISCUSSION\b',
        r'\bConclusion\b', r'\bCONCLUSION\b',
        r'\bConclusions\b', r'\bCONCLUSIONS\b',
        r'\bReferences\b', r'\bREFERENCES\b',
        r'\bAcknowledgment\b', r'\bAcknowledgments\b',
        r'\bAcknowledgement\b', r'\bAcknowledgements\b',
        r'\bDisclosure\b',
    ]:
        match = re.search(pattern, text)
        if match:
            candidates.append(match.start())
    return min(candidates) if candidates else len(text)


# ── Resolve Pronouns ─────────────────────────────────────────────
def resolve_pronouns(text):
    text = re.sub(r'\bThis study\b', 'The study on sitagliptin-metformin', text)
    text = re.sub(r'\bThis combination\b', 'The sitagliptin-metformin combination', text)
    text = re.sub(r'\bThis drug\b', 'Sitagliptin', text)
    text = re.sub(r'\bThis compound\b', 'The compound', text)
    text = re.sub(r'\bThis medicine\b', 'Sitagliptin-metformin', text)
    text = re.sub(r'\bThis tablet\b', 'The tablet formulation', text)
    text = re.sub(r'\bThis condition\b', 'Type 2 diabetes', text)
    text = re.sub(r'\bThis treatment\b', 'The treatment', text)
    text = re.sub(r'\bThis agent\b', 'Sitagliptin', text)
    text = re.sub(r'\bThis enzyme\b', 'The enzyme', text)
    text = re.sub(r'\bThis inhibitor\b', 'The DPP-4 inhibitor', text)
    text = re.sub(r'\bThat condition\b', 'Type 2 diabetes', text)
    text = re.sub(r'(?<=\. )It is also known as', 'Sitagliptin is also known as', text)
    text = re.sub(r'(?<=\. )It works by', 'Sitagliptin works by', text)
    text = re.sub(r'(?<=\. )It helps', 'Sitagliptin helps', text)
    text = re.sub(r'(?<=\. )It reduces', 'Sitagliptin reduces', text)
    text = re.sub(r'(?<=\. )It was', 'Sitagliptin was', text)
    text = re.sub(r'(?<=\. )It is', 'Sitagliptin is', text)
    return text


# ── Improve Quality ──────────────────────────────────────────────
def improve_quality(text):
    # Remove citation numbers like [1], [13], [13,14], [1-6]
    text = re.sub(r'\[\d+(?:[,\-–]\d+)*\]', '', text)

    # Remove lines starting with numbers like "13,14"
    text = re.sub(r'^\d+[,\d]*\s', '', text, flags=re.MULTILINE)

    # Fix multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)

    # Fix space before punctuation
    text = re.sub(r'\s+([.,;:])', r'\1', text)

    # Remove orphan brackets
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\[\s*\]', '', text)

    # Remove lines with less than 5 words
    lines = text.split('\n')
    lines = [l for l in lines if len(l.split()) >= 5]
    text = '\n'.join(lines)

    return text

def make_professional(text):

    # ── Fix sentence starters ────────────────────────────────────

    # "Either the..." → remove dangling either
    text = re.sub(r'\bEither the\b', 'The', text)
    text = re.sub(r'\bEither of those events\b', 'Both conditions', text)
    text = re.sub(r'\bEither\b', 'Both conditions', text)

    # "Those events" → "These conditions"
    text = re.sub(r'\bthose events\b', 'these conditions', text, flags=re.IGNORECASE)

    # "Such as" at start
    text = re.sub(r'^Such as\b', 'These include', text, flags=re.MULTILINE)

    # "Also," at start of sentence
    text = re.sub(r'(?<=\. )Also,', 'Additionally,', text)
    text = re.sub(r'^Also,', 'Additionally,', text, flags=re.MULTILINE)

    # "But" at start of sentence
    text = re.sub(r'(?<=\. )But\b', 'However,', text)
    text = re.sub(r'^But\b', 'However,', text, flags=re.MULTILINE)

    # "And" at start of sentence
    text = re.sub(r'(?<=\. )And\b', 'Furthermore,', text)
    text = re.sub(r'^And\b', 'Furthermore,', text, flags=re.MULTILINE)

    # "So," at start
    text = re.sub(r'(?<=\. )So,', 'Therefore,', text)
    text = re.sub(r'^So,', 'Therefore,', text, flags=re.MULTILINE)

    # "The use of this" → "The use of sitagliptin-metformin"
    text = re.sub(r'\bThe use of this\b', 'The use of sitagliptin-metformin', text, flags=re.IGNORECASE)

    # "These studies" → "Clinical studies"
    text = re.sub(r'\bThese studies\b', 'Clinical studies', text)

    # "Such studies" → "Clinical studies"
    text = re.sub(r'\bSuch studies\b', 'Clinical studies', text)

    # "This was" → "Sitagliptin was"
    text = re.sub(r'(?<=\. )This was\b', 'Sitagliptin was', text)

    # "The latter" → "Sitagliptin"
    text = re.sub(r'\bThe latter\b', 'Sitagliptin', text)

    # "The former" → "Metformin"
    text = re.sub(r'\bThe former\b', 'Metformin', text)

    # ── Fix informal language ────────────────────────────────────

    # "a lot of" → "a significant number of"
    text = re.sub(r'\ba lot of\b', 'a significant number of', text, flags=re.IGNORECASE)

    # "get" → "achieve/obtain"
    text = re.sub(r'\bget better\b', 'improve', text, flags=re.IGNORECASE)
    text = re.sub(r'\bgets worse\b', 'deteriorates', text, flags=re.IGNORECASE)

    # "kids" → "children"
    text = re.sub(r'\bkids\b', 'children', text, flags=re.IGNORECASE)

    # "stay high" → "remain elevated"
    text = re.sub(r'\bstay high\b', 'remain elevated', text, flags=re.IGNORECASE)

    # "blood sugar" → "blood glucose"
    text = re.sub(r'\bblood sugar\b', 'blood glucose', text, flags=re.IGNORECASE)

    # "make enough" → "produce sufficient"
    text = re.sub(r'\bmake enough\b', 'produce sufficient', text, flags=re.IGNORECASE)

    # "fight" → "combat"
    text = re.sub(r'\bfights\b', 'combats', text, flags=re.IGNORECASE)
    text = re.sub(r'\bdirectly fights\b', 'directly combats', text, flags=re.IGNORECASE)

    # "chance of getting" → "risk of developing"
    text = re.sub(r'\bchance of getting\b', 'risk of developing', text, flags=re.IGNORECASE)

    # "taken by mouth" → "administered orally"
    text = re.sub(r'\btaken by mouth\b', 'administered orally', text, flags=re.IGNORECASE)

    # "help people gain weight" → "cause weight gain"
    text = re.sub(r'\bhelp people gain weight\b', 'cause weight gain', text, flags=re.IGNORECASE)

    # "cause their blood" → "causing blood"
    text = re.sub(r'\bcause their blood glucose levels to remain elevated\b',
                  'resulting in persistently elevated blood glucose levels', text, flags=re.IGNORECASE)

    # "works well and is safe" → "demonstrates efficacy and safety"
    text = re.sub(r'\bworks well and is safe\b', 'demonstrates established efficacy and safety', text, flags=re.IGNORECASE)

    return text


# ── Remove Noise ─────────────────────────────────────────────────
def remove_noise(text):
    # Image placeholders
    text = re.sub(r'==>.*?<==', '', text)

    # Paper title repeating
    text = re.sub(r'Sitagliptin Phosphate Monohydrate.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Formulation Development.*?\n', '', text, flags=re.IGNORECASE)

    # Lines starting with colon
    text = re.sub(r'^\s*:.*?\n', '', text, flags=re.MULTILINE)

    # Step labels
    text = re.sub(r'\bStep\s*\d+\b\s*:?\s*', '', text, flags=re.IGNORECASE)

    # URLs and emails
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)

    # DOI
    text = re.sub(r'doi:.*?\n', '', text, flags=re.IGNORECASE)

    # Journal metadata
    text = re.sub(r'How to cite.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Source of support.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Conflict of interest.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\*?Author for Correspondence.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[A-Z]+,?\s*Vol(ume)?\s*\d+.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Page\s*\d+', '', text, flags=re.IGNORECASE)

    # Copyright lines
    text = re.sub(r'.*Creative Commons.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'.*All rights reserved.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'.*Dove Medical Press.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'.*submit your manuscript.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'.*licensed by.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'.*permission.*?\n', '', text, flags=re.IGNORECASE)

    # Figure and table references
    text = re.sub(r'(Fig\.|Figure|FIGURE)\s*\d+.*?\n', '', text)
    text = re.sub(r'\b(table|fig)\s*\d+\b', '', text, flags=re.IGNORECASE)

    # All citation formats
    text = re.sub(r'\[\d+(?:[,\-–]\d+)*\]', '', text)
    # Remove (Author and Author, 2002) style
    text = re.sub(r'\([A-Z][a-z]+\s+and\s+[A-Z][a-z]+,\s*\d{4}\)', '', text)

    # Remove (Author et al., 2019) style
    text = re.sub(r'\([A-Z][a-z]+(?:\s+et\s+al\.)?(?:,\s*\d{4})?\)', '', text)

    # Remove (Author, 2019) lowercase style
    text = re.sub(r'\([a-z][a-z]+(?:\s+et\s+al\.)?(?:,\s*\d{4})?\)', '', text)

    # Remove any remaining (Name Year) patterns
    text = re.sub(r'\([A-Za-z]+.*?\d{4}\)', '', text)
    text = re.sub(r'\(\d{4}\)', '', text)

    # Author name patterns
    text = re.sub(r'\b[A-Z][a-z]+\s+[A-Z]{1,3},?\s+[A-Z][a-z]+\s+[A-Z]{1,3}.*?\.\s*', '', text)

    # Lines with only numbers or symbols
    text = re.sub(r'^\s*[^a-zA-Z\s]+\s*$', '', text, flags=re.MULTILINE)

    # Improve quality
    text = improve_quality(text)

    # Resolve pronouns
    text = resolve_pronouns(text)

    # Clean whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()
    # Make language professional
    text = make_professional(text)

    return text


# ── Chunk ────────────────────────────────────────────────────────
def chunk_text(text):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    chunk_id = 1

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > 400:
            if current_chunk.strip():
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": current_chunk.strip()
                })
                chunk_id += 1
            current_chunk = sentence + " "
        else:
            current_chunk += sentence + " "

    if current_chunk.strip():
        chunks.append({
            "chunk_id": chunk_id,
            "text": current_chunk.strip()
        })

    return chunks


# ── Keywords ─────────────────────────────────────────────────────
def extract_keywords(chunks, kw_model):
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
        all_keywords = list(set(auto_kw_list + manual_kw_list))
        chunk["keywords"] = all_keywords
        tagged_chunks.append(chunk)

    return tagged_chunks


# ── Main ─────────────────────────────────────────────────────────
kw_model = KeyBERT()
merged_text = ""

pdf_files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]
print(f"Found {len(pdf_files)} PDFs\n{'='*60}")

for pdf_file in pdf_files:
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

# Deduplicate sentences
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

# Chunk once
final_text = " ".join(unique_sentences)
chunks = chunk_text(final_text)
print(f"✅ Total chunks: {len(chunks)}")

# Keywords once
print("\n⏳ Extracting keywords...")
tagged_chunks = extract_keywords(chunks, kw_model)
print(f"✅ Done!\n{'='*60}")

# Print results
for chunk in tagged_chunks:
    print(f"\nChunk {chunk['chunk_id']}")
    print(f"Keywords: {chunk['keywords']}")
    print(f"Text: {chunk['text']}")
    print("-" * 60)