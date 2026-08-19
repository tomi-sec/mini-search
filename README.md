# Mini Search Engine Over the Python Docs

A crawler, a hand-built inverted index, and BM25 ranking over the real Python 3
documentation tree - no Elasticsearch, no vector database, nothing that would let a
library do the actual work this project exists to demonstrate.

## How it works

`crawler.py` does a polite, resumable BFS over `docs.python.org/3/` (robots.txt
respected, one request every 0.4s, state entirely in SQLite so killing the process
and restarting it just resumes). `textpipeline.py` turns each page's HTML into
stemmed, stopword-filtered tokens. `indexer.py` builds a plain `term -> {doc_id:
[positions]}` dict by hand and pickles it. `ranking.py` implements TF-IDF and BM25
from scratch. `search.py` ties it together and builds highlighted snippets from the
stored positions. `api.py` (FastAPI) and `frontend/` (React + TypeScript) are the
demo surface.

## The corpus

561 pages crawled, 560 indexed (one was a linked `.py` source file, correctly
excluded as non-HTML). That's short of the 2,000+ page target I started with, and
the reason is structural, not a bug: Sphinx docs bundle a lot per physical page -
the entire "Data Structures" tutorial chapter is one page, `functions.html` alone
documents every built-in function. 327 of the 561 pages are individual library
reference pages (one per stdlib module), 76 are C API pages, the rest are tutorial
chapters, HOWTOs, and reference material. It's the real, complete `/3/` tree, not a
truncated crawl - the frontier ran dry, not the page budget.

35,728 unique terms, avg. 2,712 tokens/doc, index built in 1.8s, 7.0 MB on disk.
Query latency: 0.13-1.2ms depending on how many candidates match, averaging under
1ms across a sample of real queries - nowhere close to needing the 50ms budget.

## Ranking: does it actually help, and is BM25 actually right

I validated my BM25 math against the `rank_bm25` library on a toy corpus first
(`tests/test_ranking.py`) - same IDF formula, scores agree to within float
tolerance, ranking order matches exactly.

Then I ran 15 real queries against the real corpus, each with a human-judged
expected page, and compared three retrieval strategies at a top-5 cutoff:

| Method | Expected page in top 5 | Ranked #1 |
|---|---|---|
| Unranked (candidates, unordered) | 0/15 | - |
| TF-IDF | 14/15 | 4/15 |
| BM25 | 15/15 | 12/15 |

Unranked isn't a strawman: a query like "json module" matches 560 of 560 documents
(nearly every doc mentions "module" somewhere), so without scoring, the right page
showing up in the first 5 by chance is essentially never. Ranking is not an
optimization here, it's the entire product. BM25 beating TF-IDF 12/15 vs 4/15 on
rank #1 is the expected result, but it's worth having the number instead of assuming
it.

## Hardest problems (the honest version)

**A crawler bug that only showed up on a redirect.** My first canonicalization pass
stripped trailing slashes ("/3/" -> "/3") for dedup purposes, and I reused that same
canonical form as the base for resolving relative links. `docs.python.org/3` real-
redirects to `docs.python.org/3/`, and resolving `"tutorial/index.html"` against
`.../3` (no trailing slash - urljoin reads "3" as a filename) silently produced
`docs.python.org/tutorial/index.html` - one directory too high, quietly dropping
the whole crawl into nothing. Fixed by resolving links against the *raw* post-
redirect URL and only canonicalizing the result, not the base itself.

**A scoping problem that isn't a bug, just a trap.** A same-domain-only crawl
wandered into `docs.python.org/3.13/`, `/3.12/`, ... - a full parallel doc tree for
every Python version, linked from the homepage nav. I added a path-prefix scope
(same domain *and* under `/3/`) so the page budget goes toward depth in one version
instead of N shallow near-duplicates. This is the site-structure version of the
crawler traps the project brief warns about.

**A ranking bug the eval set actually caught.** The classic BM25 IDF formula can go
negative for a term appearing in over half the corpus. "json module" was scoring
*negative* and losing to `pprint.html` because "module" (extremely common in
documentation about modules) dragged the sum down even though "json" matched
strongly. I'd flagged this as a theoretical gap in a code comment before ever
running the real eval - then the eval actually hit it. Fixed with an IDF floor;
`json.html` now wins, and the toy-corpus validation against `rank_bm25` still
matches exactly since that corpus never crosses the 50% threshold.

**What's still a known gap.** `genindex-*.html` pages (Python's own A-Z symbol
index) are just enormous flat lists of terms, and they win queries like "regular
expressions" and "walrus operator" over the actual `re.html` / `whatsnew/3.8.html`
pages purely on term density - they're in the eval set as honest misses, not
excluded. A real fix would down-weight or exclude auto-generated index pages from
ranking; I didn't do that here, since one page-type heuristic away from "search
engine" starts becoming "search engine plus a growing pile of site-specific rules."

## Running it

```
pip install -r requirements.txt

python src/main.py crawl              # resumable - safe to Ctrl+C and rerun
python src/main.py index
python src/main.py evaluate           # ranked vs unranked, BM25 vs TF-IDF

uvicorn api:app --app-dir src --port 8000

cd frontend && npm install && npm run dev
```
