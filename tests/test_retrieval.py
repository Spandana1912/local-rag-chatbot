import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retrieval import (
    generate_query_variations,
    rewrite_query_with_history,
    _tokenize,
)


def test_tokenize():
    tokens = _tokenize("What is Retrieval Augmented Generation?")
    assert tokens == ["what", "is", "retrieval", "augmented", "generation?"]


def test_query_rewrite_no_history():
    query = "What is the capital of France?"
    rewritten = rewrite_query_with_history("llama3.2", query, [])
    assert rewritten == query


def test_generate_query_variations_fallback():
    # Calling with invalid model should fallback gracefully to returning [query]
    variations = generate_query_variations("non_existent_model_xyz", "What is RAG?")
    assert len(variations) >= 1
    assert variations[0] == "What is RAG?"
