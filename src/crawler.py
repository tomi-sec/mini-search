"""A polite, resumable BFS crawler over one domain.

State lives entirely in SQLite (see PAGES_SCHEMA) - a row's status
column doubles as both the frontier (status="pending") and the visited
set (any other status), so killing the process and restarting it just
means re-running the same "fetch every pending row" loop against
whatever's left. There is deliberately no in-memory queue that would
lose that state on a crash.
"""

import sqlite3
import time
import urllib.robotparser
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from selectolax.parser import HTMLParser

USER_AGENT = (
    "ResumePortfolioSearchCrawler/1.0 "
    "(educational portfolio project; polite, single-threaded, rate-limited)"
)
DEFAULT_DELAY_SECONDS = 0.4
SKIP_EXTENSIONS = frozenset(
    {
        ".zip", ".tar", ".gz", ".bz2", ".tgz", ".epub", ".pdf",
        ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
        ".whl", ".exe", ".msi",
    }
)

PAGES_SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    url TEXT PRIMARY KEY,
    depth INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    status_code INTEGER,
    content_type TEXT,
    html TEXT,
    fetched_at REAL
);
"""


def open_db(path):
    """Opens (creating if needed) the crawl's SQLite database.

    Args:
        path: Path to the SQLite file.

    Returns:
        A sqlite3.Connection with the pages table ready.
    """
    conn = sqlite3.connect(path)
    conn.execute(PAGES_SCHEMA)
    conn.commit()
    return conn


def canonicalize_url(url):
    """Normalizes a URL so equivalent pages map to one canonical form.

    Lowercases scheme and host, drops the fragment (#section-anchor,
    which never changes what page loads), and strips a trailing slash
    from any path longer than "/" - so ".../a" and ".../a/" collapse to
    the same row instead of being crawled as two different pages.

    Args:
        url: The URL to canonicalize.

    Returns:
        The canonical URL string.
    """
    parts = urlsplit(url)
    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def extract_links(html, base_url):
    """Extracts, resolves, and canonicalizes every link on a page.

    Args:
        html: The page's raw HTML.
        base_url: The page's own canonical URL, used to resolve relative
            hrefs.

    Returns:
        A list of canonicalized absolute URLs, one per usable <a href>
        found (duplicates included - the caller dedupes against what's
        already known). Fragment-only links, mailto:/javascript: links,
        non-http(s) schemes, and known binary/media extensions are
        skipped so the crawler doesn't waste a fetch on a non-page or
        wander into a download link.
    """
    tree = HTMLParser(html)
    links = []
    for node in tree.css("a[href]"):
        href = node.attributes.get("href")
        if not href or href.startswith(("#", "mailto:", "javascript:")):
            continue
        absolute = urljoin(base_url, href)
        if urlsplit(absolute).scheme not in ("http", "https"):
            continue
        path_only = urlsplit(absolute).path.lower()
        if any(path_only.endswith(ext) for ext in SKIP_EXTENSIONS):
            continue
        links.append(canonicalize_url(absolute))
    return links


def load_robots(base_url):
    """Fetches and parses the target domain's robots.txt.

    Args:
        base_url: Any URL on the target domain; only its scheme/host is
            used to locate robots.txt.

    Returns:
        A urllib.robotparser.RobotFileParser, already read.
    """
    parts = urlsplit(base_url)
    robots_url = urlunsplit((parts.scheme, parts.netloc, "/robots.txt", "", ""))
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(robots_url)
    parser.read()
    return parser


def _in_scope(url, domain, path_prefix):
    """Checks whether a URL is both in-domain and under the seed's path prefix.

    Restricting to the seed's own path prefix (e.g. "/3/" when seeding
    from docs.python.org/3/) keeps the crawl inside one version's docs
    tree. docs.python.org links to a full parallel doc tree for every
    other Python version from its own homepage nav - a same-domain-only
    check would follow all of them and spend the page budget on N
    near-duplicate trees instead of depth in the one that matters. This
    is exactly the kind of crawler trap the spec warns about, just
    site-structure-shaped rather than infinite-pagination-shaped.

    Args:
        url: A canonicalized URL to check.
        domain: The in-domain netloc.
        path_prefix: The seed's path, with a trailing slash (e.g. "/3/").

    Returns:
        True if url shares domain and its path is either the bare
        prefix (no trailing slash, matching the seed's own canonical
        form) or starts with path_prefix.
    """
    parts = urlsplit(url)
    if parts.netloc != domain:
        return False
    return parts.path == path_prefix.rstrip("/") or parts.path.startswith(path_prefix)


def _fetch_one(client, conn, url, depth, domain, path_prefix, max_pages):
    """Fetches one URL, records the result, and enqueues new in-scope links.

    Args:
        client: The shared httpx.Client.
        conn: The crawl database connection.
        url: The URL to fetch (already known to be robots-allowed).
        depth: This URL's BFS depth - discovered links are enqueued at
            depth + 1.
        domain: The in-domain netloc; only links matching it are
            considered.
        path_prefix: The seed's path prefix (see _in_scope); only links
            under it are enqueued.
        max_pages: Used to stop growing the frontier once it's already
            well past the target - already-pending rows still get
            fetched out, so this only caps runaway link discovery on a
            page-rich site, not the final crawl size.
    """
    try:
        response = client.get(url)
    except httpx.HTTPError as exc:
        conn.execute(
            "UPDATE pages SET status = 'error', fetched_at = ? WHERE url = ?", (time.time(), url)
        )
        conn.commit()
        print(f"crawler: {url} -> error ({exc})")
        return

    content_type = response.headers.get("content-type", "")
    is_html = "text/html" in content_type
    html = response.text if is_html else None
    # A redirect (e.g. "/3" -> "/3/") changes what relative links on the
    # page actually mean - "tutorial/index.html" resolves differently
    # against "/3" (no trailing slash, so "3" reads as a filename to
    # urljoin) than against "/3/". raw_final_url keeps that trailing
    # slash for link resolution; final_url is the canonicalized form
    # (slash stripped) used for row identity/dedup below - canonicalizing
    # the *base* itself would silently reintroduce the same bug.
    raw_final_url = str(response.url)
    final_url = canonicalize_url(raw_final_url)

    conn.execute(
        """UPDATE pages SET status = 'done', status_code = ?, content_type = ?,
           html = ?, fetched_at = ? WHERE url = ?""",
        (response.status_code, content_type, html, time.time(), url),
    )
    if final_url != url:
        # Record the redirect target as already-fetched too, so a later
        # in-page link straight to it doesn't trigger a second fetch.
        conn.execute(
            """INSERT INTO pages (url, depth, status, status_code, content_type, html, fetched_at)
               VALUES (?, ?, 'done', ?, ?, ?, ?)
               ON CONFLICT(url) DO UPDATE SET status='done', status_code=excluded.status_code,
                 content_type=excluded.content_type, html=excluded.html, fetched_at=excluded.fetched_at""",
            (final_url, depth, response.status_code, content_type, html, time.time()),
        )

    if is_html and response.status_code == 200:
        frontier_size = conn.execute("SELECT count(*) FROM pages").fetchone()[0]
        if frontier_size < max_pages * 2:
            for link in extract_links(html, raw_final_url):
                if _in_scope(link, domain, path_prefix):
                    conn.execute(
                        "INSERT OR IGNORE INTO pages (url, depth, status) VALUES (?, ?, 'pending')",
                        (link, depth + 1),
                    )
    conn.commit()
    redirect_note = f" -> {final_url}" if final_url != url else ""
    print(f"crawler: [{response.status_code}] {url}{redirect_note}")


def crawl(seed_url, db_path, max_pages=3000, delay_seconds=DEFAULT_DELAY_SECONDS):
    """Runs (or resumes) a breadth-first crawl of one domain.

    Args:
        seed_url: Where to start. Also defines the crawl's scope: only
            links sharing its netloc *and* its path prefix are followed
            (see _in_scope) - so seeding from docs.python.org/3/ stays
            inside the Python 3 docs tree instead of wandering into
            every other version's parallel tree via the site's own nav.
        db_path: Path to the SQLite crawl database (see open_db).
        max_pages: Stop once this many pages have been processed
            (fetched, errored, or robots-disallowed).
        delay_seconds: Minimum seconds between requests. The delay
            actually used is max(this, the site's own robots.txt
            Crawl-delay for our user agent, if any) - the site's stated
            preference always wins if it's stricter than ours.

    Returns:
        The number of pages fetched in this call (0 on a fully-resumed,
        already-finished crawl - nothing left in the frontier).
    """
    conn = open_db(db_path)
    seed = canonicalize_url(seed_url)
    domain = urlsplit(seed).netloc
    path_prefix = urlsplit(seed_url).path
    if not path_prefix.endswith("/"):
        path_prefix += "/"

    robots = load_robots(seed)
    site_delay = robots.crawl_delay(USER_AGENT) or robots.crawl_delay("*")
    delay = max(delay_seconds, float(site_delay) if site_delay else 0.0)

    conn.execute("INSERT OR IGNORE INTO pages (url, depth, status) VALUES (?, 0, 'pending')", (seed,))
    conn.commit()

    client = httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=10.0)
    fetched_this_run = 0
    try:
        while True:
            done_count = conn.execute(
                "SELECT count(*) FROM pages WHERE status != 'pending'"
            ).fetchone()[0]
            if done_count >= max_pages:
                break

            row = conn.execute(
                "SELECT url, depth FROM pages WHERE status = 'pending' ORDER BY rowid LIMIT 1"
            ).fetchone()
            if row is None:
                break
            url, depth = row

            if not robots.can_fetch(USER_AGENT, url):
                conn.execute("UPDATE pages SET status = 'disallowed' WHERE url = ?", (url,))
                conn.commit()
                continue

            time.sleep(delay)
            _fetch_one(client, conn, url, depth, domain, path_prefix, max_pages)
            fetched_this_run += 1
    finally:
        client.close()
        conn.close()
    return fetched_this_run
