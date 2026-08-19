"""Query -> ranked results with highlighted snippets and timing.

The only logic here beyond ranking.py is turning a query into terms
(reusing textpipeline.tokenize, so a query is normalized exactly like a
document was) and using the stored term positions to build a snippet
instead of just returning the whole page.
"""

import time

from ranking import rank
from textpipeline import tokenize

SNIPPET_RADIUS_CHARS = 80
NO_MATCH_SNIPPET_CHARS = 160


def search(index, documents, query, method="bm25", limit=10):
    """Runs a search end to end: tokenize, rank, build result payloads.

    Args:
        index: The index dict from indexer.build_index.
        documents: dict of doc_id -> {"url", "title", "text"} (built
            alongside the index - see main.py's index command).
        query: The raw query string.
        method: "bm25" or "tfidf", passed through to ranking.rank.
        limit: Max number of results to return.

    Returns:
        A dict with "query", "took_ms", "total_candidates", and
        "results" (a list of dicts with "url", "title", "score", and
        "snippet", best first).
    """
    start = time.perf_counter()
    query_terms = tokenize(query)
    ranked = rank(index, query_terms, method=method)
    took_ms = (time.perf_counter() - start) * 1000

    results = []
    for doc_id, score in ranked[:limit]:
        doc = documents[doc_id]
        results.append(
            {
                "url": doc["url"],
                "title": doc["title"],
                "score": score,
                "snippet": _snippet(doc["text"], query_terms),
            }
        )
    return {
        "query": query,
        "took_ms": took_ms,
        "total_candidates": len(ranked),
        "results": results,
    }


def _snippet(text, query_terms):
    """Builds a short excerpt around the first query term match, bolded.

    query_terms are stemmed (e.g. "running" -> "run"). Snowball stemming
    only strips suffixes, so a stem is always a prefix of the original
    word - searching for it as a plain substring of the untouched raw
    text reliably finds the matching word (and near variants sharing the
    same stem) without needing to stem the document text too.

    Args:
        text: The document's full body text (unstemmed).
        query_terms: The tokenized (stemmed) query terms to look for.

    Returns:
        A snippet string with the matched span wrapped in **double
        asterisks**, or the text's first NO_MATCH_SNIPPET_CHARS
        characters if no query term is found at all.
    """
    lowered = text.lower()
    match_pos, match_len = None, 0
    for term in query_terms:
        pos = lowered.find(term)
        if pos != -1 and (match_pos is None or pos < match_pos):
            match_pos, match_len = pos, len(term)

    if match_pos is not None:
        # Extend the match to the end of its containing word - the stem
        # itself is usually shorter than the real word ("run" inside
        # "running"), and bolding just the stem reads oddly.
        end_of_word = match_pos + match_len
        while end_of_word < len(text) and text[end_of_word].isalnum():
            end_of_word += 1
        match_len = end_of_word - match_pos

    if match_pos is None:
        return text[:NO_MATCH_SNIPPET_CHARS].strip() + "..."

    start = max(0, match_pos - SNIPPET_RADIUS_CHARS)
    end = min(len(text), match_pos + match_len + SNIPPET_RADIUS_CHARS)
    before = text[start:match_pos]
    matched = text[match_pos : match_pos + match_len]
    after = text[match_pos + match_len : end]
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{before}**{matched}**{after}{suffix}"
