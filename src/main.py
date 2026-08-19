"""CLI entrypoint: crawl a corpus, build its index, or run the eval queries.

    python main.py crawl [--seed URL] [--max-pages N] [--delay SECONDS]
    python main.py index
    python main.py evaluate
"""

import argparse
import json
import pickle
import sqlite3
import time
from pathlib import Path

from crawler import crawl
from indexer import build_index, load_index, save_index
from ranking import average_doc_length, candidate_doc_ids
from search import search
from textpipeline import extract_text, tokenize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "crawl.db"
INDEX_PATH = DATA_DIR / "index.pkl"
DOCUMENTS_PATH = DATA_DIR / "documents.pkl"
EVAL_QUERIES_PATH = PROJECT_ROOT / "eval" / "queries.json"
DEFAULT_SEED = "https://docs.python.org/3/"


def cmd_crawl(args):
    """Runs (or resumes) the crawl and prints a progress summary.

    Args:
        args: Parsed CLI args (seed, max_pages, delay).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fetched = crawl(args.seed, DB_PATH, max_pages=args.max_pages, delay_seconds=args.delay)
    conn = sqlite3.connect(DB_PATH)
    total_done = conn.execute("SELECT count(*) FROM pages WHERE status = 'done'").fetchone()[0]
    conn.close()
    print(f"fetched {fetched} pages this run, {total_done} total done in {DB_PATH}")


def cmd_index(args):
    """Builds the inverted index and document store from crawled pages.

    Args:
        args: Parsed CLI args (unused, kept for a consistent dispatch
            signature across the crawl/index/evaluate subcommands).
    """
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT url, html FROM pages WHERE status = 'done' AND status_code = 200 AND html IS NOT NULL"
    ).fetchall()
    conn.close()

    documents = {}
    tokenized = {}
    start = time.time()
    for doc_id, (url, html) in enumerate(rows):
        title, body_text = extract_text(html)
        documents[doc_id] = {"url": url, "title": title, "text": body_text}
        tokenized[doc_id] = tokenize(body_text)
    build_seconds = time.time() - start

    index = build_index(tokenized)
    save_index(index, INDEX_PATH)
    with open(DOCUMENTS_PATH, "wb") as f:
        pickle.dump(documents, f)

    vocab_size = len(index["postings"])
    print(
        f"indexed {len(documents)} documents, {vocab_size} unique terms, "
        f"avg doc length {average_doc_length(index):.1f} tokens, "
        f"built in {build_seconds:.1f}s"
    )
    print(f"index size on disk: {INDEX_PATH.stat().st_size / 1e6:.1f} MB")


def _unranked_hit(index, documents, query, expected_url, top_n=5):
    """Checks whether the expected URL survives an *unranked* top-N cutoff.

    "Unranked" here means every candidate document (any doc sharing at
    least one query term) in whatever order the index's postings happen
    to produce - no scoring at all - truncated to the same top_n a real
    search result page would show. This is the baseline ranked search is
    actually being compared against, not a strawman: for a query with
    many candidates, a real page only shows the first handful, so an
    unordered candidate set is exactly as useless as it would be in a
    real product with no ranking.

    Args:
        index: The index dict from indexer.build_index.
        documents: dict of doc_id -> {"url", "title", "text"}.
        query: The raw query string.
        expected_url: The URL a human judged as the right answer.
        top_n: How many unranked candidates to keep, matching the ranked
            comparison's limit.

    Returns:
        True if expected_url is among the first top_n candidate doc_ids.
    """
    query_terms = tokenize(query)
    candidates = list(candidate_doc_ids(index, query_terms))[:top_n]
    return expected_url in {documents[doc_id]["url"] for doc_id in candidates}


def cmd_evaluate(args):
    """Runs the hand-curated eval queries: ranked vs unranked, BM25 vs TF-IDF.

    Args:
        args: Parsed CLI args (unused, kept for a consistent dispatch
            signature across the crawl/index/evaluate subcommands).
    """
    index = load_index(INDEX_PATH)
    with open(DOCUMENTS_PATH, "rb") as f:
        documents = pickle.load(f)
    queries = json.loads(EVAL_QUERIES_PATH.read_text(encoding="utf-8"))

    unranked_hits = sum(
        _unranked_hit(index, documents, case["query"], case["expected_url"]) for case in queries
    )
    print(f"unranked (top 5 candidates, no scoring): {unranked_hits}/{len(queries)} queries")

    for method in ("tfidf", "bm25"):
        hits_top5 = 0
        hits_top1 = 0
        for case in queries:
            result = search(index, documents, case["query"], method=method, limit=5)
            top_urls = [r["url"] for r in result["results"]]
            if case["expected_url"] in top_urls:
                hits_top5 += 1
            if top_urls[:1] == [case["expected_url"]]:
                hits_top1 += 1
        print(
            f"{method}: expected URL in top 5 for {hits_top5}/{len(queries)} queries "
            f"({hits_top1}/{len(queries)} ranked #1)"
        )


def main():
    """Parses CLI arguments and dispatches to the crawl/index/evaluate subcommand."""
    parser = argparse.ArgumentParser(description="Mini search engine over docs.python.org")
    subparsers = parser.add_subparsers(dest="command", required=True)

    crawl_parser = subparsers.add_parser("crawl", help="crawl (or resume crawling) the corpus")
    crawl_parser.add_argument("--seed", default=DEFAULT_SEED)
    crawl_parser.add_argument("--max-pages", type=int, default=3000)
    crawl_parser.add_argument("--delay", type=float, default=0.4)
    crawl_parser.set_defaults(func=cmd_crawl)

    index_parser = subparsers.add_parser("index", help="build the inverted index from crawled pages")
    index_parser.set_defaults(func=cmd_index)

    eval_parser = subparsers.add_parser("evaluate", help="run the hand-curated eval queries")
    eval_parser.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
