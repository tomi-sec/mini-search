import indexer


def test_build_index_tracks_positions_per_document():
    documents = {0: ["cat", "sat", "cat"]}
    index = indexer.build_index(documents)
    assert index["postings"]["cat"] == {0: [0, 2]}
    assert index["postings"]["sat"] == {0: [1]}


def test_build_index_tracks_postings_across_multiple_documents():
    documents = {0: ["cat", "dog"], 1: ["dog", "bird"]}
    index = indexer.build_index(documents)
    assert index["postings"]["dog"] == {0: [1], 1: [0]}
    assert index["postings"]["cat"] == {0: [0]}
    assert index["postings"]["bird"] == {1: [1]}


def test_build_index_doc_lengths_and_doc_count():
    documents = {0: ["a", "b", "c"], 1: ["a"]}
    index = indexer.build_index(documents)
    assert index["doc_lengths"] == {0: 3, 1: 1}
    assert index["doc_count"] == 2


def test_build_index_on_empty_corpus():
    index = indexer.build_index({})
    assert index["postings"] == {}
    assert index["doc_lengths"] == {}
    assert index["doc_count"] == 0


def test_term_frequency_is_derivable_from_postings_length():
    documents = {0: ["x", "x", "x", "y"]}
    index = indexer.build_index(documents)
    assert len(index["postings"]["x"][0]) == 3


def test_save_and_load_index_roundtrip(tmp_path):
    index = indexer.build_index({0: ["alpha", "beta"], 1: ["beta", "gamma"]})
    path = tmp_path / "index.pkl"
    indexer.save_index(index, path)
    loaded = indexer.load_index(path)
    assert loaded == index
