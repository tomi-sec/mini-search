import textpipeline


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert textpipeline.tokenize("Hello, World!") == ["hello", "world"]


def test_tokenize_removes_stopwords():
    tokens = textpipeline.tokenize("the cat and the dog")
    assert "the" not in tokens
    assert "and" not in tokens


def test_tokenize_applies_stemming():
    assert textpipeline.tokenize("running runner runs") == ["run", "runner", "run"]


def test_tokenize_keeps_multi_digit_numbers():
    # Single-character tokens are dropped (see test_tokenize_drops_single_characters),
    # so "3" from "3.12" doesn't survive, but the multi-digit "12" does.
    assert textpipeline.tokenize("Python 3.12 released") == ["python", "12", "releas"]


def test_tokenize_drops_single_characters():
    assert textpipeline.tokenize("a b cat") == ["cat"]


def test_tokenize_on_empty_string_is_empty():
    assert textpipeline.tokenize("") == []


def test_tokenize_on_mixed_language_keeps_ascii_words_only():
    """Non-ASCII characters aren't in the [a-z0-9] token pattern, so they act as separators."""
    tokens = textpipeline.tokenize("hello café world")
    assert tokens == ["hello", "caf", "world"]


def test_tokenize_is_deterministic_and_order_preserving():
    text = "alpha beta gamma"
    assert textpipeline.tokenize(text) == textpipeline.tokenize(text)


def test_extract_text_pulls_the_title():
    html = "<html><head><title>My Page</title></head><body><p>Hello</p></body></html>"
    title, body = textpipeline.extract_text(html)
    assert title == "My Page"
    assert "Hello" in body


def test_extract_text_strips_script_and_nav_content():
    html = """
    <html><body>
      <nav>Site Nav Links</nav>
      <script>var x = 1;</script>
      <p>Real content here.</p>
    </body></html>
    """
    _, body = textpipeline.extract_text(html)
    assert "Real content here." in body
    assert "Site Nav Links" not in body
    assert "var x" not in body


def test_extract_text_on_a_page_with_no_title_returns_empty_string():
    title, _ = textpipeline.extract_text("<html><body><p>No title here</p></body></html>")
    assert title == ""


def test_extract_text_collapses_whitespace():
    html = "<html><body><p>Line one</p>\n\n<p>Line   two</p></body></html>"
    _, body = textpipeline.extract_text(html)
    assert "  " not in body
