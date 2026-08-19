import pytest
from rank_bm25 import BM25Okapi

import ranking
from indexer import build_index

TOY_DOCS = {
    0: "the cat sat on the mat".split(),
    1: "the dog sat on the log".split(),
    2: "cats and dogs are common pets".split(),
    3: "the quick brown fox jumps over the lazy dog".split(),
    4: "python is a programming language".split(),
}


def test_candidate_doc_ids_only_returns_docs_containing_a_query_term():
    index = build_index(TOY_DOCS)
    assert ranking.candidate_doc_ids(index, ["python"]) == {4}
    assert ranking.candidate_doc_ids(index, ["dog"]) == {1, 3}
    assert ranking.candidate_doc_ids(index, ["nonexistent"]) == set()


def test_average_doc_length():
    index = build_index({0: ["a", "b"], 1: ["a", "b", "c", "d"]})
    assert ranking.average_doc_length(index) == 3.0


def test_tfidf_prefers_the_document_with_more_term_occurrences():
    # "dog" must NOT appear in every document, or its idf (log(N/df))
    # is log(1) = 0 and every score collapses to 0 regardless of tf.
    index = build_index({0: "dog dog dog".split(), 1: "dog cat".split(), 2: "bird bird".split()})
    scored = dict(ranking.rank(index, ["dog"], method="tfidf"))
    assert scored[0] > scored[1]


def test_bm25_scores_the_correct_document_highest_for_a_distinctive_query():
    index = build_index(TOY_DOCS)
    ranked = ranking.rank(index, ["python", "program"], method="bm25")
    assert ranked[0][0] == 4


def test_rank_raises_on_an_unknown_method():
    index = build_index(TOY_DOCS)
    with pytest.raises(ValueError):
        ranking.rank(index, ["cat"], method="nonsense")


def test_bm25_agrees_with_rank_bm25_reference_implementation():
    """My hand-written BM25 should rank the same as the reference library, on the same corpus."""
    doc_ids = sorted(TOY_DOCS)
    corpus_tokens = [TOY_DOCS[doc_id] for doc_id in doc_ids]
    reference = BM25Okapi(corpus_tokens, k1=ranking.K1, b=ranking.B)

    index = build_index(TOY_DOCS)
    queries = [["dog"], ["cat", "mat"], ["python", "program"], ["quick", "fox"]]

    for query_terms in queries:
        reference_scores = reference.get_scores(query_terms)
        candidates = ranking.candidate_doc_ids(index, query_terms)

        # Only compare docs that actually match a query term - rank_bm25
        # scores every document in the corpus (non-matching docs get 0,
        # or with its epsilon smoothing, a small positive score), while
        # mine only ranks true candidates.
        reference_ranked = [
            doc_id
            for doc_id, _ in sorted(zip(doc_ids, reference_scores), key=lambda p: p[1], reverse=True)
            if doc_id in candidates
        ]
        mine_ranked_scores = ranking.rank(index, query_terms, method="bm25")
        mine_ranked = [doc_id for doc_id, _ in mine_ranked_scores]
        assert mine_ranked == reference_ranked, query_terms

        mine_scores = dict(mine_ranked_scores)
        for doc_id in candidates:
            reference_score = reference_scores[doc_ids.index(doc_id)]
            assert mine_scores[doc_id] == pytest.approx(reference_score, abs=0.05), (query_terms, doc_id)
