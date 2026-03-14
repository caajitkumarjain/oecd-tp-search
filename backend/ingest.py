"""
Ingest OECD Transfer Pricing Guidelines PDF into ChromaDB.

Workflow:
1. Parse PDF with PyMuPDF (fitz)
2. Split into paragraph-level chunks with metadata (page, chapter, para_ref)
3. Generate embeddings using sentence-transformers (all-MiniLM-L6-v2)
4. Store in ChromaDB persisted to backend/data/
"""

import os
import re
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(SCRIPT_DIR, "..", "docs", "OECD_Guidelines.pdf")
CHROMA_DIR = os.path.join(SCRIPT_DIR, "data")
COLLECTION_NAME = "oecd_tp_guidelines"

# Chunking config
MIN_CHUNK_LENGTH = 50
MAX_CHUNK_LENGTH = 2000

# Paragraph reference pattern: "1.29." or "10.5." on its own line or at start
PARA_REF_PATTERN = re.compile(r"^(\d{1,2}\.\d{1,3})\.?\s", re.MULTILINE)

# Canonical chapter mapping — roman numerals to standard names
ROMAN_TO_CHAPTER = {
    "I": "Chapter I",
    "II": "Chapter II",
    "III": "Chapter III",
    "IV": "Chapter IV",
    "V": "Chapter V",
    "VI": "Chapter VI",
    "VII": "Chapter VII",
    "VIII": "Chapter VIII",
    "IX": "Chapter IX",
    "X": "Chapter X",
}


def extract_text_by_page(pdf_path: str) -> list[dict]:
    """Extract text from each page of the PDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages.append({"page": page_num + 1, "text": text})
    doc.close()
    return pages


def detect_chapter(text: str) -> str:
    """Detect chapter from page header/text using multiple strategies."""
    # Strategy 1: Match "CHAPTER IX" or "Chapter IX" in headers
    # Full roman numeral pattern: I, II, III, IV, V, VI, VII, VIII, IX, X
    roman_pattern = r"(?:X|IX|VIII|VII|VI|V|IV|III|II|I)"
    for pattern in [
        rf"CHAPTER\s+({roman_pattern})\b",
        rf"Chapter\s+({roman_pattern})\b",
        rf"CHAPTER\s+(\d{{1,2}})\b",
        rf"Chapter\s+(\d{{1,2}})\b",
    ]:
        match = re.search(pattern, text)
        if match:
            val = match.group(1)
            if val in ROMAN_TO_CHAPTER:
                return ROMAN_TO_CHAPTER[val]
            if val.isdigit():
                # Map numeric to roman
                num_map = {
                    "1": "I", "2": "II", "3": "III", "4": "IV", "5": "V",
                    "6": "VI", "7": "VII", "8": "VIII", "9": "IX", "10": "X",
                }
                roman = num_map.get(val, val)
                return ROMAN_TO_CHAPTER.get(roman, f"Chapter {val}")
            return f"Chapter {val}"

    # Strategy 2: Infer from paragraph numbering (e.g., "9.1." means Chapter IX)
    para_match = re.search(r"\b(\d{1,2})\.\d{1,3}\.\s", text)
    if para_match:
        ch_num = para_match.group(1)
        num_map = {
            "1": "I", "2": "II", "3": "III", "4": "IV", "5": "V",
            "6": "VI", "7": "VII", "8": "VIII", "9": "IX", "10": "X",
        }
        roman = num_map.get(ch_num)
        if roman and roman in ROMAN_TO_CHAPTER:
            return ROMAN_TO_CHAPTER[roman]

    return "Unknown"


def extract_para_ref(text: str) -> str:
    """Extract OECD paragraph reference from text (e.g., 1.29, 6.48, 10.5)."""
    match = PARA_REF_PATTERN.match(text)
    if match:
        return match.group(1)
    return ""


def chunk_text(pages: list[dict]) -> list[dict]:
    """
    Split page text into paragraph-level chunks.
    Detects OECD paragraph references (e.g., 1.29, 6.48, 10.5) and attaches as metadata.
    """
    chunks = []
    current_chapter = "Preamble"
    chunk_id = 0

    for page_data in pages:
        page_num = page_data["page"]
        text = page_data["text"]

        # Detect chapter from page text
        detected = detect_chapter(text)
        if detected != "Unknown":
            current_chapter = detected

        # Split on paragraph reference boundaries: lines starting with "X.XX." or "XX.XX."
        parts = re.split(r"\n(?=\d{1,2}\.\d{1,3}\.\s)", text)

        for part in parts:
            # Also split on double newlines within parts
            sub_parts = re.split(r"\n{2,}", part)

            for para in sub_parts:
                para = para.strip()
                para = re.sub(r"\s+", " ", para)

                if len(para) < MIN_CHUNK_LENGTH:
                    continue

                para_ref = extract_para_ref(para)
                display_text = para

                def make_chunk(text_block, ref):
                    nonlocal chunk_id
                    chunks.append({
                        "id": f"chunk_{chunk_id}",
                        "text": text_block,
                        "page": page_num,
                        "chapter": current_chapter,
                        "para_ref": ref,
                    })
                    chunk_id += 1

                if len(display_text) > MAX_CHUNK_LENGTH:
                    sentences = re.split(r"(?<=[.!?])\s+", display_text)
                    current_block = ""
                    first_block = True
                    for sentence in sentences:
                        if len(current_block) + len(sentence) > MAX_CHUNK_LENGTH and current_block:
                            make_chunk(current_block.strip(), para_ref if first_block else "")
                            first_block = False
                            current_block = sentence
                        else:
                            current_block += " " + sentence
                    if current_block.strip():
                        make_chunk(current_block.strip(), para_ref if first_block else "")
                else:
                    make_chunk(display_text, para_ref)

    return chunks


def ingest():
    """Main ingestion pipeline."""
    if not os.path.exists(PDF_PATH):
        print(f"ERROR: PDF not found at {os.path.abspath(PDF_PATH)}")
        print("Place the OECD Transfer Pricing Guidelines PDF at: docs/OECD_Guidelines.pdf")
        return

    print(f"Loading PDF from {os.path.abspath(PDF_PATH)}...")
    pages = extract_text_by_page(PDF_PATH)
    print(f"Extracted text from {len(pages)} pages.")

    print("Chunking text into paragraphs...")
    chunks = chunk_text(pages)
    para_count = sum(1 for c in chunks if c["para_ref"])
    print(f"Created {len(chunks)} chunks ({para_count} with paragraph references).")

    # Print chapter breakdown
    chapter_counts = {}
    for c in chunks:
        ch = c["chapter"]
        chapter_counts[ch] = chapter_counts.get(ch, 0) + 1
    print("\nChapter breakdown:")
    for ch in sorted(chapter_counts.keys()):
        print(f"  {ch}: {chapter_counts[ch]} chunks")

    print(f"\nLoading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Generating embeddings...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    print(f"Storing in ChromaDB at {os.path.abspath(CHROMA_DIR)}...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size].tolist()
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=batch_embeddings,
            metadatas=[
                {
                    "page": c["page"],
                    "chapter": c["chapter"],
                    "para_ref": c["para_ref"],
                }
                for c in batch
            ],
        )

    print(f"\nDone. {len(chunks)} chunks ingested into collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    ingest()
