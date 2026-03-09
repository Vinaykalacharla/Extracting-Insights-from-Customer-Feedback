"""
Lightweight Aspect-Based Sentiment Analysis utilities.

Design goals:
- Work out-of-the-box with minimal dependencies.
- Use NLTK VADER when available for better sentiment scoring.
- Fall back to a small lexicon-based scorer if dependencies are missing.
- Rule-based aspect extraction using keyword maps; optional spaCy integration can be added later.

Functions:
- analyze_reviews(reviews: List[str]) -> (aggregated_list, details)
  aggregated_list: list of dicts with aspect, count, avg_score, pos_count, neg_count, neutral_count
  details: list per-review with detected aspects and their sentence-level scores
"""
from typing import List, Tuple, Dict
import re
import collections

try:
    import pandas as pd
except Exception:
    pd = None

try:
    # Prefer NLTK's VADER if available
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk import download as nltk_download
    try:
        # Ensure lexicon is present
        nltk_download('vader_lexicon', quiet=True)
    except Exception:
        pass
    _VADER_AVAILABLE = True
except Exception:
    SentimentIntensityAnalyzer = None
    _VADER_AVAILABLE = False

# Minimal fallback lexicons
_POS_WORDS = set(["good", "great", "excellent", "amazing", "love", "nice", "best", "happy", "satisfied"])
_NEG_WORDS = set(["bad", "poor", "terrible", "awful", "hate", "worst", "disappointed", "slow", "problem"])

# Simple aspect keywords mapping (expandable)
ASPECT_KEYWORDS = {
    'battery': ['battery', 'charge', 'charging', 'battery life', 'power'],
    'screen': ['screen', 'display', 'resolution', 'touchscreen', 'brightness'],
    'service': ['service', 'support', 'customer service', 'staff', 'agent', 'help'],
    'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'affordable','pricing'],
    'performance': ['performance', 'speed', 'slow', 'fast', 'lag', 'smooth'],
    'camera': ['camera', 'photo', 'picture', 'selfie', 'video'],
    'design': ['design', 'look', 'appearance', 'build', 'quality'],
    'software': ['software', 'app', 'update', 'bug', 'feature'],
}

_ASPECT_PATTERN = {
    a: re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keys) + r")\b", flags=re.I)
    for a, keys in ASPECT_KEYWORDS.items()
}


def _simple_sentiment_score(text: str) -> float:
    """Fallback sentiment: (+1 for pos word, -1 for neg) / sqrt(len words) to normalize a bit."""
    words = re.findall(r"\w+", text.lower())
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in _POS_WORDS)
    neg = sum(1 for w in words if w in _NEG_WORDS)
    score = pos - neg
    # normalize
    import math
    return float(score) / math.sqrt(len(words))


def _vader_score(text: str) -> float:
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(text)['compound']


def _score_text(text: str) -> float:
    if _VADER_AVAILABLE and SentimentIntensityAnalyzer is not None:
        try:
            return _vader_score(text)
        except Exception:
            return _simple_sentiment_score(text)
    else:
        return _simple_sentiment_score(text)


def _sentences(text: str):
    # very small sentence splitter
    parts = re.split(r'[\.\!?]\s+|\n+', text)
    return [p.strip() for p in parts if p.strip()]


def extract_aspects(text: str) -> List[Tuple[str, str]]:
    """Return list of (aspect, sentence) pairs found in the text.

    We find sentences that match any aspect keyword and associate the aspect.
    """
    results = []
    for sent in _sentences(text):
        for aspect, pattern in _ASPECT_PATTERN.items():
            if pattern.search(sent):
                results.append((aspect, sent))
    return results


def analyze_reviews(reviews: List[str]) -> Tuple[List[Dict], List[Dict]]:
    """
    Analyze a list of reviews and aggregate sentiment per aspect.

    Returns (aggregated_list, details)

    aggregated_list: list of dicts with keys: aspect, count, avg_score, pos_count, neg_count, neutral_count
    details: per-review detail list: {index, text, aspects: [{aspect, sentence, score}]}
    """
    per_aspect_scores = collections.defaultdict(list)
    details = []

    for i, review in enumerate(reviews):
        found = extract_aspects(review)
        entry = {'index': i, 'text': review, 'aspects': []}
        for aspect, sent in found:
            score = _score_text(sent)
            per_aspect_scores[aspect].append(score)
            entry['aspects'].append({'aspect': aspect, 'sentence': sent, 'score': float(score)})
        details.append(entry)

    rows = []
    for aspect, scores in per_aspect_scores.items():
        cnt = len(scores)
        avg = float(sum(scores) / cnt) if cnt else 0.0
        pos = sum(1 for s in scores if s > 0.05)
        neg = sum(1 for s in scores if s < -0.05)
        neu = cnt - pos - neg
        rows.append({'aspect': aspect, 'count': cnt, 'avg_score': avg, 'pos_count': pos, 'neg_count': neg, 'neutral_count': neu})

    # If pandas present, create a DataFrame and sort; otherwise just return list sorted by count
    if pd is not None:
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(by='count', ascending=False)
        aggregated = df
    else:
        aggregated = sorted(rows, key=lambda r: r['count'], reverse=True)

    return aggregated, details


def highlight_aspects(text: str) -> str:
    """Wrap aspect keywords with a span that is styled as bold by CSS."""
    highlighted = text
    for aspect, pattern in _ASPECT_PATTERN.items():
        highlighted = pattern.sub(lambda m: f"<span class=\"aspect-highlight\">{m.group()}</span>", highlighted)

    return highlighted
