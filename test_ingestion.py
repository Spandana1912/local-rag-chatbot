import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingestion import chunk_text, split_sentences


def test_split_sentences_basic():
    assert split_sentences("Hello world. This is a test!") == ["Hello world.", "This is a test!"]


def test_split_sentences_empty():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_chunk_text_respects_sentence_boundaries():
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    chunks = chunk_text(text, chunk_size=30, overlap_sentences=1)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.strip().endswith((".", "!", "?"))


def test_chunk_text_empty_input():
    assert chunk_text("") == []


def test_chunk_text_overlap_carries_context():
    text = "Alpha sentence. Beta sentence. Gamma sentence. Delta sentence."
    chunks = chunk_text(text, chunk_size=20, overlap_sentences=1)
    # with overlap, the last sentence of one chunk should reappear
    # as the start of the next chunk (when there's more than one chunk)
    if len(chunks) > 1:
        last_sentence_of_first = chunks[0].split(". ")[-1]
        assert last_sentence_of_first.strip(".") in chunks[1]