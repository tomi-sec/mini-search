"""FastAPI backend: one search endpoint over a pre-built index.

The index and document store are loaded once at startup (see lifespan)
rather than per-request - both are read-only after main.py's index
command builds them, so there's nothing to invalidate or lock.
"""

import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from indexer import load_index
from search import search

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_PATH = Path(os.environ.get("INDEX_PATH", DATA_DIR / "index.pkl"))
DOCUMENTS_PATH = Path(os.environ.get("DOCUMENTS_PATH", DATA_DIR / "documents.pkl"))


@asynccontextmanager
async def lifespan(app):
    """Loads the prebuilt index and document store once at startup.

    Args:
        app: The FastAPI application instance.
    """
    app.state.index = load_index(INDEX_PATH)
    with open(DOCUMENTS_PATH, "rb") as f:
        app.state.documents = pickle.load(f)
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])


@app.get("/search")
async def search_endpoint(q, method="bm25", limit=10):
    """Runs a search query against the loaded index.

    Args:
        q: The query string.
        method: "bm25" (default) or "tfidf".
        limit: Max results to return.

    Returns:
        The dict produced by search.search: query, took_ms,
        total_candidates, and results.
    """
    return search(app.state.index, app.state.documents, q, method=method, limit=int(limit))
