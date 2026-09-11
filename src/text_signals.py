"""Text-based red flags from 10-K narrative sections (Item 1A Risk Factors,
Item 7 MD&A): year-over-year rewrite magnitude and shifts in cautious/
negative/litigious language.

The word lists below are a compact, hand-curated subset inspired by the
categories in the Loughran-McDonald financial sentiment dictionary (the
standard academic word list for 10-K text analysis) -- not the full
official list, which would need to be downloaded separately.
"""
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NEGATIVE_WORDS = {
    "adverse", "adversely", "loss", "losses", "decline", "declined", "declining",
    "deficit", "impairment", "impairments", "write-off", "writedown", "restructuring",
    "weakness", "weaknesses", "unfavorable", "unfavourable", "failure", "failed",
    "default", "defaults", "downturn", "deteriorate", "deteriorated", "deterioration",
    "shortfall", "penalty", "penalties", "misstatement", "restatement", "restated",
}

UNCERTAINTY_WORDS = {
    "may", "might", "could", "uncertain", "uncertainty", "uncertainties",
    "approximately", "possibly", "unpredictable", "fluctuate", "fluctuations",
    "contingent", "contingency", "unknown", "variability", "volatile", "volatility",
}

LITIGIOUS_WORDS = {
    "litigation", "lawsuit", "lawsuits", "plaintiff", "plaintiffs", "defendant",
    "settlement", "subpoena", "regulatory", "investigation", "sec", "allegation",
    "allegations", "indemnify", "indemnification", "breach", "claims", "liable",
    "liability", "enforcement",
}

ITEM_1A_RE = re.compile(r"item\s*1a\.?\s*risk\s*factors(.*?)item\s*1b", re.I | re.S)
ITEM_7_RE = re.compile(
    r"item\s*7\.?\s*management.?s\s*discussion(.*?)item\s*7a", re.I | re.S
)


MIN_SECTION_CHARS = 200


def extract_section(full_text: str, pattern: re.Pattern) -> str:
    """A 10-K's table of contents often matches 'Item 1A ... Item 1B' too
    (just a page number in between), so the first regex match is usually
    the ToC line, not the real section. The real section is always much
    longer, so take the longest match instead of the first.
    """
    matches = [m.group(1) for m in pattern.finditer(full_text)]
    candidates = [m for m in matches if len(m) >= MIN_SECTION_CHARS]
    if candidates:
        return max(candidates, key=len)
    return max(matches, key=len) if matches else ""


def word_frequency_score(text: str, word_set: set[str]) -> float:
    """Occurrences of words in `word_set` per 1,000 words of text."""
    words = re.findall(r"[a-zA-Z\-]+", text.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in word_set)
    return 1000 * hits / len(words)


def tone_scores(text: str) -> dict:
    return {
        "negative_per_1k": word_frequency_score(text, NEGATIVE_WORDS),
        "uncertainty_per_1k": word_frequency_score(text, UNCERTAINTY_WORDS),
        "litigious_per_1k": word_frequency_score(text, LITIGIOUS_WORDS),
    }


def similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity (0-1) between two filings' TF-IDF vectors.
    A large drop year-over-year means the section was substantially
    rewritten -- sometimes benign, sometimes a sign something changed
    that management had to explain differently.
    """
    if not text_a.strip() or not text_b.strip():
        return float("nan")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    try:
        tfidf = vectorizer.fit_transform([text_a, text_b])
    except ValueError:
        return float("nan")
    return float(cosine_similarity(tfidf[0], tfidf[1])[0][0])
