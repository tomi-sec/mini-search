"""HTML -> clean text -> tokens: a pure, testable pipeline.

Every function here takes and returns plain strings/lists - no I/O, no
database, no network - so the ugly cases (punctuation, numbers, mixed
case, empty documents) are unit-testable in isolation from the crawler
and the index.
"""

import re

import Stemmer
from selectolax.parser import HTMLParser

STEMMER = Stemmer.Stemmer("english")

# A hand-rolled list rather than a corpus-derived one - small and
# readable enough to audit at a glance, which matters more here than
# completeness.
STOPWORDS = frozenset(
    """
    a an the and or but if then else for while with without of to in on
    at by from as is are was were be been being this that these those
    it its not no nor so than too very can will just should now do does
    did doing have has had having i you he she we they them his her our
    your their what which who whom
    """.split()
)

TOKEN_RE = re.compile(r"[a-z0-9]+")
STRIP_TAGS = "script, style, nav, header, footer"


def extract_text(html):
    """Pulls a title and readable body text out of a raw HTML page.

    Args:
        html: The page's raw HTML.

    Returns:
        A tuple `(title, body_text)`. `title` is the <title> tag's text
        (or "" if absent). `body_text` is every text node under <body>
        with script/style/nav/header/footer content removed first and
        whitespace collapsed to single spaces.
    """
    tree = HTMLParser(html)
    title_node = tree.css_first("title")
    title = title_node.text(strip=True) if title_node else ""

    for tag in tree.css(STRIP_TAGS):
        tag.decompose()

    body = tree.body
    body_text = body.text(separator=" ", strip=True) if body else ""
    body_text = re.sub(r"\s+", " ", body_text).strip()
    return title, body_text


def tokenize(text):
    """Splits text into lowercase, stopword-filtered, stemmed tokens.

    Args:
        text: Raw text, e.g. extract_text's body_text or a search query.

    Returns:
        A list of stemmed tokens in their original order - the position
        of a token in this list is what indexer.py stores as its term
        position, which is why tokenize must be deterministic and
        order-preserving.
    """
    words = TOKEN_RE.findall(text.lower())
    return [STEMMER.stemWord(w) for w in words if w not in STOPWORDS and len(w) > 1]
