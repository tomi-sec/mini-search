import crawler


def test_canonicalize_url_lowercases_scheme_and_host():
    assert crawler.canonicalize_url("HTTPS://Example.COM/Path") == "https://example.com/Path"


def test_canonicalize_url_drops_the_fragment():
    assert crawler.canonicalize_url("https://example.com/a#section-2") == "https://example.com/a"


def test_canonicalize_url_strips_a_trailing_slash_from_a_multi_char_path():
    assert crawler.canonicalize_url("https://example.com/a/") == "https://example.com/a"


def test_canonicalize_url_keeps_the_bare_root_slash():
    assert crawler.canonicalize_url("https://example.com/") == "https://example.com/"


def test_canonicalize_url_makes_equivalent_urls_identical():
    a = crawler.canonicalize_url("https://example.com/a")
    b = crawler.canonicalize_url("https://example.com/a/#top")
    assert a == b


def test_canonicalize_url_keeps_the_query_string():
    assert crawler.canonicalize_url("https://example.com/a?x=1") == "https://example.com/a?x=1"


def test_extract_links_resolves_relative_hrefs_against_the_base():
    html = '<a href="tutorial/index.html">Tutorial</a>'
    links = crawler.extract_links(html, "https://docs.python.org/3/")
    assert links == ["https://docs.python.org/3/tutorial/index.html"]


def test_extract_links_skips_fragment_only_mailto_and_javascript_links():
    html = """
    <a href="#top">Top</a>
    <a href="mailto:someone@example.com">Mail</a>
    <a href="javascript:void(0)">JS</a>
    <a href="/real-page.html">Real</a>
    """
    links = crawler.extract_links(html, "https://example.com/")
    assert links == ["https://example.com/real-page.html"]


def test_extract_links_skips_known_binary_extensions():
    html = '<a href="/archive.tar.bz2">Download</a><a href="/page.html">Page</a>'
    links = crawler.extract_links(html, "https://example.com/")
    assert links == ["https://example.com/page.html"]


def test_extract_links_skips_non_http_schemes():
    html = '<a href="ftp://example.com/file">FTP</a><a href="/ok.html">OK</a>'
    links = crawler.extract_links(html, "https://example.com/")
    assert links == ["https://example.com/ok.html"]


def test_extract_links_includes_duplicates_for_the_caller_to_dedupe():
    html = '<a href="/a.html">A</a><a href="/a.html">A again</a>'
    links = crawler.extract_links(html, "https://example.com/")
    assert links == ["https://example.com/a.html", "https://example.com/a.html"]


def test_in_scope_requires_matching_domain():
    assert crawler._in_scope("https://other.com/3/x.html", "docs.python.org", "/3/") is False


def test_in_scope_accepts_the_bare_prefix_and_anything_under_it():
    domain = "docs.python.org"
    assert crawler._in_scope("https://docs.python.org/3", domain, "/3/") is True
    assert crawler._in_scope("https://docs.python.org/3/tutorial/index.html", domain, "/3/") is True


def test_in_scope_rejects_a_sibling_version_tree():
    """A path like /3.12/ must not match the /3/ prefix by string-prefix accident."""
    domain = "docs.python.org"
    assert crawler._in_scope("https://docs.python.org/3.12/index.html", domain, "/3/") is False
