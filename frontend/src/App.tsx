import { FormEvent, useState } from "react";

interface SearchResult {
  url: string;
  title: string;
  score: number;
  snippet: string;
}

interface SearchResponse {
  query: string;
  took_ms: number;
  total_candidates: number;
  results: SearchResult[];
}

// Snippets come from the backend with the matched span wrapped in
// **double asterisks** (see search.py's _snippet) - splitting on that
// pattern is simpler than shipping a markdown parser for one tag.
function highlightSnippet(snippet: string) {
  const parts = snippet.split(/\*\*(.+?)\*\*/g);
  return parts.map((part, i) => (i % 2 === 1 ? <mark key={i}>{part}</mark> : <span key={i}>{part}</span>));
}

export default function App() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runSearch(e: FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/search?q=${encodeURIComponent(query)}`);
      if (!res.ok) throw new Error(`search failed: ${res.status}`);
      setResponse((await res.json()) as SearchResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <h1>Python Docs Search</h1>
      <form onSubmit={runSearch} className="search-form">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search the Python docs..."
          autoFocus
        />
        <button type="submit" disabled={loading}>
          {loading ? "Searching..." : "Search"}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {response && (
        <>
          <p className="meta">
            {response.took_ms.toFixed(1)} ms &middot; {response.total_candidates} documents searched
          </p>
          <ul className="results">
            {response.results.map((r) => (
              <li key={r.url}>
                <a href={r.url} target="_blank" rel="noreferrer">
                  {r.title || r.url}
                </a>
                <p className="snippet">{highlightSnippet(r.snippet)}</p>
                <p className="url">{r.url}</p>
              </li>
            ))}
          </ul>
          {response.results.length === 0 && <p>No results.</p>}
        </>
      )}
    </div>
  );
}
