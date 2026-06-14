import re


# ── Remove Tables ────────────────────────────────────────────────
def remove_tables(text: str) -> str:
    text = re.sub(r'(\|.*\|[\r\n]*)+', '', text)
    text = re.sub(r'(\|[-:]+\|[\r\n]*)+', '', text)
    return text


# ── Remove Markdown ──────────────────────────────────────────────
def remove_markdown(text: str) -> str:
    text = re.sub(r'#+\s+', '', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'---+', '', text)
    return text


# ── Find Start ───────────────────────────────────────────────────
def find_start_position(text: str) -> int:
    for pattern in [
        r'\bABSTRACT\b', r'\bAbstract\b',
        r'\bINTRODUCTION\b', r'\bIntroduction\b'
    ]:
        match = re.search(pattern, text)
        if match:
            return match.end()
    return 0


# ── Find End → cut at earliest section ──────────────────────────
def find_end_position(text: str) -> int:
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
def resolve_pronouns(text: str) -> str:
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
def improve_quality(text: str) -> str:
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


# ── Make Language Professional ───────────────────────────────────
def make_professional(text: str) -> str:
    text = re.sub(r'\bEither the\b', 'The', text)
    text = re.sub(r'\bEither of those events\b', 'Both conditions', text)
    text = re.sub(r'\bEither\b', 'Both conditions', text)
    text = re.sub(r'\bthose events\b', 'these conditions', text, flags=re.IGNORECASE)
    text = re.sub(r'^Such as\b', 'These include', text, flags=re.MULTILINE)
    text = re.sub(r'(?<=\. )Also,', 'Additionally,', text)
    text = re.sub(r'^Also,', 'Additionally,', text, flags=re.MULTILINE)
    text = re.sub(r'(?<=\. )But\b', 'However,', text)
    text = re.sub(r'^But\b', 'However,', text, flags=re.MULTILINE)
    text = re.sub(r'(?<=\. )And\b', 'Furthermore,', text)
    text = re.sub(r'^And\b', 'Furthermore,', text, flags=re.MULTILINE)
    text = re.sub(r'(?<=\. )So,', 'Therefore,', text)
    text = re.sub(r'^So,', 'Therefore,', text, flags=re.MULTILINE)
    text = re.sub(r'\bThe use of this\b', 'The use of sitagliptin-metformin', text, flags=re.IGNORECASE)
    text = re.sub(r'\bThese studies\b', 'Clinical studies', text)
    text = re.sub(r'\bSuch studies\b', 'Clinical studies', text)
    text = re.sub(r'(?<=\. )This was\b', 'Sitagliptin was', text)
    text = re.sub(r'\bThe latter\b', 'Sitagliptin', text)
    text = re.sub(r'\bThe former\b', 'Metformin', text)
    text = re.sub(r'\ba lot of\b', 'a significant number of', text, flags=re.IGNORECASE)
    text = re.sub(r'\bget better\b', 'improve', text, flags=re.IGNORECASE)
    text = re.sub(r'\bgets worse\b', 'deteriorates', text, flags=re.IGNORECASE)
    text = re.sub(r'\bkids\b', 'children', text, flags=re.IGNORECASE)
    text = re.sub(r'\bstay high\b', 'remain elevated', text, flags=re.IGNORECASE)
    text = re.sub(r'\bblood sugar\b', 'blood glucose', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmake enough\b', 'produce sufficient', text, flags=re.IGNORECASE)
    text = re.sub(r'\bfights\b', 'combats', text, flags=re.IGNORECASE)
    text = re.sub(r'\bdirectly fights\b', 'directly combats', text, flags=re.IGNORECASE)
    text = re.sub(r'\bchance of getting\b', 'risk of developing', text, flags=re.IGNORECASE)
    text = re.sub(r'\btaken by mouth\b', 'administered orally', text, flags=re.IGNORECASE)
    text = re.sub(r'\bhelp people gain weight\b', 'cause weight gain', text, flags=re.IGNORECASE)
    text = re.sub(
        r'\bcause their blood glucose levels to remain elevated\b',
        'resulting in persistently elevated blood glucose levels',
        text, flags=re.IGNORECASE
    )
    text = re.sub(
        r'\bworks well and is safe\b',
        'demonstrates established efficacy and safety',
        text, flags=re.IGNORECASE
    )
    return text


# ── Remove Noise ─────────────────────────────────────────────────
def remove_noise(text: str) -> str:
    text = re.sub(r'==>.*?<==', '', text)
    text = re.sub(r'Sitagliptin Phosphate Monohydrate.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Formulation Development.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*:.*?\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'\bStep\s*\d+\b\s*:?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'doi:.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'How to cite.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Source of support.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Conflict of interest.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\*?Author for Correspondence.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[A-Z]+,?\s*Vol(ume)?\s*\d+.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Page\s*\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'.*Creative Commons.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'.*All rights reserved.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'.*Dove Medical Press.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'.*submit your manuscript.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'.*licensed by.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'.*permission.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(Fig\.|Figure|FIGURE)\s*\d+.*?\n', '', text)
    text = re.sub(r'\b(table|fig)\s*\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\d+(?:[,\-–]\d+)*\]', '', text)
    text = re.sub(r'\([A-Z][a-z]+\s+and\s+[A-Z][a-z]+,\s*\d{4}\)', '', text)
    text = re.sub(r'\([A-Z][a-z]+(?:\s+et\s+al\.)?(?:,\s*\d{4})?\)', '', text)
    text = re.sub(r'\([a-z][a-z]+(?:\s+et\s+al\.)?(?:,\s*\d{4})?\)', '', text)
    text = re.sub(r'\([A-Za-z]+.*?\d{4}\)', '', text)
    text = re.sub(r'\(\d{4}\)', '', text)
    text = re.sub(r'\b[A-Z][a-z]+\s+[A-Z]{1,3},?\s+[A-Z][a-z]+\s+[A-Z]{1,3}.*?\.\s*', '', text)
    text = re.sub(r'^\s*[^a-zA-Z\s]+\s*$', '', text, flags=re.MULTILINE)

    text = improve_quality(text)
    text = resolve_pronouns(text)

    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()

    text = make_professional(text)
    return text