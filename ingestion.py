"""
Ingestion: extract text from uploaded files and split it into retrieval-ready
chunks.

Two upgrades over naive fixed-width character chunking:
1. Sentence-aware chunking — chunks end on sentence boundaries instead of
   cutting words/sentences in half, which measurably improves embedding
   quality (a half-sentence embeds to a worse, noisier vector).
2. Page tracking for PDFs — each chunk remembers which page it came from,
   so the app can cite "source.pdf, page 4" instead of just a filename.
"""
import io
import re

from pypdf import PdfReader
from docx import Document


def extract_text(uploaded_file):
    """
    Extract text from .txt, .pdf, .docx, .md, .json, or .csv files.

    Returns a list of segments: [{"text": str, "page": int | None}, ...]
    """
    file_type = uploaded_file.name.split(".")[-1].lower()
    segments = []

    if file_type in ("txt", "md"):
        content = uploaded_file.read()
        text = content.decode("utf-8", errors="ignore") if isinstance(content, bytes) else str(content)
        segments.append({"text": text, "page": None})

    elif file_type == "pdf":
        reader = PdfReader(uploaded_file)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                segments.append({"text": page_text, "page": i + 1})

    elif file_type == "docx":
        data = uploaded_file.read()
        doc = Document(io.BytesIO(data) if isinstance(data, bytes) else data)
        text = "\n".join(p.text for p in doc.paragraphs)
        segments.append({"text": text, "page": None})

    elif file_type in ("json", "csv"):
        content = uploaded_file.read()
        text = content.decode("utf-8", errors="ignore") if isinstance(content, bytes) else str(content)
        segments.append({"text": text, "page": None})

    else:
        return None

    return segments


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return _SENTENCE_SPLIT.split(text)


def chunk_text(text, chunk_size=None, overlap_sentences=None):
    from config import CHUNK_SIZE, OVERLAP_SENTENCES
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if overlap_sentences is None:
        overlap_sentences = OVERLAP_SENTENCES
    """
    Group whole sentences into chunks of roughly `chunk_size` characters,
    carrying the last `overlap_sentences` sentences into the next chunk so
    context isn't lost at chunk boundaries.
    """
    sentences = split_sentences(text)
    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        current.append(sentence)
        current_len += len(sentence)
        if current_len >= chunk_size:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_len = sum(len(s) for s in current)

    if current:
        chunks.append(" ".join(current))

    return [c.strip() for c in chunks if c.strip()]


def chunk_document(segments, chunk_size=None, overlap_sentences=None):
    """
    Chunk each segment (PDF page, or whole doc for txt/docx) independently,
    so chunks never straddle a page boundary. Returns:
    [{"text": ..., "page": ...}, ...]
    """
    all_chunks = []
    for seg in segments:
        for c in chunk_text(seg["text"], chunk_size, overlap_sentences):
            all_chunks.append({"text": c, "page": seg["page"]})
    return all_chunks