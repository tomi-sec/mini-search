from indexer import build_index
from search import _snippet, search
from textpipeline import tokenize

DOCS_RAW = {
    0: {
        "url": "https://docs.python.org/3/library/functions.html",
        "title": "Built-in Functions",
        "text": "The Python interpreter has a number of built-in functions that are always available for use.",
    },
    1: {
        "url": "https://docs.python.org/3/tutorial/index.html",
        "title": "The Python Tutorial",
        "text": "Python is an easy to learn, powerful programming language used for many kinds of software.",
    },
}


def _build():
    """Builds a tiny index + document store from DOCS_RAW, for search() tests.

    Returns:
        A tuple (index, documents) ready to pass to search.search.
    """
    tokenized = {doc_id: tokenize(doc["text"]) for doc_id, doc in DOCS_RAW.items()}
    return build_index(tokenized), DOCS_RAW


def test_search_finds_the_relevant_document():
    index, documents = _build()
    result = search(index, documents, "built-in functions")
    assert result["results"][0]["url"] == DOCS_RAW[0]["url"]


def test_search_reports_query_took_ms_and_total_candidates():
    index, documents = _build()
    result = search(index, documents, "python")
    assert result["query"] == "python"
    assert result["took_ms"] >= 0
    assert result["total_candidates"] == 2  # "python" appears in both docs


def test_search_respects_the_limit():
    index, documents = _build()
    result = search(index, documents, "python", limit=1)
    assert len(result["results"]) == 1


def test_search_on_a_query_with_no_matches_returns_no_results():
    index, documents = _build()
    result = search(index, documents, "zzz_nonexistent_term")
    assert result["results"] == []
    assert result["total_candidates"] == 0


def test_search_tfidf_and_bm25_both_run_and_return_a_score():
    index, documents = _build()
    for method in ("tfidf", "bm25"):
        result = search(index, documents, "python", method=method)
        assert all(isinstance(r["score"], float) for r in result["results"])


def test_snippet_bolds_the_matched_term_in_its_original_form():
    text = "This function returns the running total of all values."
    snippet = _snippet(text, ["run"])  # "run" is the stem of "running"
    assert "**running**" in snippet


def test_snippet_falls_back_to_the_start_of_the_text_on_no_match():
    text = "No query terms appear anywhere in this sentence at all."
    snippet = _snippet(text, ["zzz_nonexistent"])
    assert snippet.startswith("No query terms")


def test_snippet_truncates_a_long_document_around_the_match():
    text = "padding " * 50 + "target word here" + " more padding" * 50
    snippet = _snippet(text, ["target"])
    assert "**target**" in snippet
    assert len(snippet) < len(text)
