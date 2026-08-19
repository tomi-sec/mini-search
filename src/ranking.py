"""Ranking: TF-IDF first as a baseline, then BM25 - both hand-written.

The README's ranking section explains what k1 and b actually control;
tests/test_ranking.py is where "hand-written" gets checked against the
rank_bm25 reference library on a toy corpus so it doesn't quietly mean
"wrong."
"""

import math

K1 = 1.5
B = 0.75
# A floor for BM25's IDF term - see bm25_score for why this is needed on
# a real (non-toy) corpus.
MIN_IDF = 1e-3


def candidate_doc_ids(index, query_terms):
    """Finds every document containing at least one query term.

    Args:
        index: The index dict from indexer.build_index.
        query_terms: A list of (already-tokenized) query terms.

    Returns:
        A set of candidate doc_ids - the search space TF-IDF/BM25 score
        over, instead of scoring the whole corpus per query.
    """
    candidates = set()
    for term in query_terms:
        candidates.update(index["postings"].get(term, {}).keys())
    return candidates


def average_doc_length(index):
    """Computes the corpus's average document length, in tokens.

    Args:
        index: The index dict from indexer.build_index.

    Returns:
        The mean of index["doc_lengths"].values(), or 0.0 for an empty
        corpus.
    """
    lengths = list(index["doc_lengths"].values())
    return sum(lengths) / len(lengths) if lengths else 0.0


def tfidf_score(index, query_terms, doc_id):
    """Scores one document against a query using raw TF-IDF.

    Args:
        index: The index dict from indexer.build_index.
        query_terms: A list of (already-tokenized) query terms.
        doc_id: The document to score.

    Returns:
        The summed TF-IDF score across query_terms present in this doc:
        term frequency times log(N / document frequency) for each.
    """
    score = 0.0
    n_docs = index["doc_count"]
    for term in query_terms:
        postings = index["postings"].get(term)
        if not postings or doc_id not in postings:
            continue
        tf = len(postings[doc_id])
        df = len(postings)
        score += tf * math.log(n_docs / df)
    return score


def bm25_score(index, query_terms, doc_id, avg_doc_length, k1=K1, b=B):
    """Scores one document against a query using Okapi BM25.

    Args:
        index: The index dict from indexer.build_index.
        query_terms: A list of (already-tokenized) query terms.
        doc_id: The document to score.
        avg_doc_length: The corpus's average document length in tokens
            (see average_doc_length) - required for the length-
            normalization term.
        k1: Term-frequency saturation. Higher lets a repeated term keep
            adding score for longer before diminishing returns kick in;
            1.5 is the conventional default.
        b: Length-normalization strength, from 0 (ignore document length
            entirely) to 1 (fully normalize by it); 0.75 is the
            conventional default.

    Returns:
        The summed BM25 score across query_terms present in this doc.
    """
    score = 0.0
    n_docs = index["doc_count"]
    doc_length = index["doc_lengths"][doc_id]
    for term in query_terms:
        postings = index["postings"].get(term)
        if not postings or doc_id not in postings:
            continue
        tf = len(postings[doc_id])
        df = len(postings)
        # The classic Robertson/Sparck-Jones IDF - also what the
        # rank_bm25 reference library uses, which is why
        # test_ranking.py's validation against it expects close
        # agreement on a toy corpus where no term appears in over half
        # the documents. On the real corpus, common words like "module"
        # or "class" *do* cross that line, and a negative idf there
        # dragged an entire multi-term query's score negative even when
        # a distinctive term ("json") was a strong match - a real
        # eval-time failure ("json module" surfaced pprint.html over
        # json.html), not a hypothetical one. Floored at a small
        # positive epsilon instead of rank_bm25's fancier corpus-average-
        # based replacement, which needs the full vocabulary's mean idf
        # computed up front; this is simpler and fixes the same failure.
        idf = max(math.log((n_docs - df + 0.5) / (df + 0.5)), MIN_IDF)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * doc_length / avg_doc_length)
        score += idf * numerator / denominator
    return score


def rank(index, query_terms, method="bm25"):
    """Ranks every candidate document for a query, best first.

    Args:
        index: The index dict from indexer.build_index.
        query_terms: A list of (already-tokenized) query terms.
        method: "bm25" or "tfidf".

    Returns:
        A list of (doc_id, score) tuples, sorted by descending score.

    Raises:
        ValueError: If method isn't "bm25" or "tfidf".
    """
    if method not in ("bm25", "tfidf"):
        raise ValueError(f"unknown ranking method: {method!r}")

    candidates = candidate_doc_ids(index, query_terms)
    avg_len = average_doc_length(index)
    if method == "bm25":
        scored = [(doc_id, bm25_score(index, query_terms, doc_id, avg_len)) for doc_id in candidates]
    else:
        scored = [(doc_id, tfidf_score(index, query_terms, doc_id)) for doc_id in candidates]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored
