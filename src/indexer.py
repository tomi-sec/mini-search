"""The inverted index: hand-built term -> postings, no search library involved.

A postings entry is doc_id -> [positions] - positions are what make the
search UI's highlighted snippets possible without a second pass over the
original text, and term frequency (tf) is just len(positions), so it's
never stored separately.
"""

import pickle
from collections import defaultdict


def build_index(documents):
    """Builds an inverted index over a set of tokenized documents.

    Args:
        documents: A dict of doc_id -> token list, as produced by
            textpipeline.tokenize.

    Returns:
        A dict with "postings" (term -> {doc_id: [positions]}),
        "doc_lengths" (doc_id -> token count, which BM25 needs for
        length normalization), and "doc_count" (total documents) -
        everything ranking.py needs to score a query.
    """
    postings = defaultdict(dict)
    doc_lengths = {}
    for doc_id, tokens in documents.items():
        doc_lengths[doc_id] = len(tokens)
        term_positions = defaultdict(list)
        for position, term in enumerate(tokens):
            term_positions[term].append(position)
        for term, positions in term_positions.items():
            postings[term][doc_id] = positions
    return {"postings": dict(postings), "doc_lengths": doc_lengths, "doc_count": len(documents)}


def save_index(index, path):
    """Pickles an index to disk.

    Pickle over a bespoke binary format: the index is just nested
    dicts/lists of Python primitives, so a hand-rolled format would add
    code without adding correctness. The tradeoff, which real search
    engines don't have: this is Python-only and version-fragile (a
    pickle from a different Python or a changed index shape may not
    load), not something to hand another language or a stable format.

    Args:
        index: The index dict, as returned by build_index.
        path: Path to write the pickle file to.
    """
    with open(path, "wb") as f:
        pickle.dump(index, f)


def load_index(path):
    """Unpickles an index from disk.

    Args:
        path: Path to a pickle file written by save_index.

    Returns:
        The index dict.
    """
    with open(path, "rb") as f:
        return pickle.load(f)
