import nltk
import spacy
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from nltk.corpus import stopwords, opinion_lexicon
from collections import defaultdict, Counter
import json

# Initialize NLP tools
nltk.download("stopwords", quiet=True)
nltk.download("opinion_lexicon", quiet=True)
nltk.download("vader_lexicon", quiet=True)

stop_words = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")

# Load sentiment lexicon
positive_words = set(opinion_lexicon.positive())
negative_words = set(opinion_lexicon.negative())

# Load sentiment models
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
MODEL_IRONY = "cardiffnlp/twitter-roberta-base-irony"

# Initialize models (lazy loading to avoid issues)
_sentiment_analyzer = None
_irony_analyzer = None
_tokenizer = None
_model = None

def get_sentiment_analyzer():
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = pipeline("sentiment-analysis",
                                    model=MODEL_NAME,
                                    tokenizer=MODEL_NAME)
    return _sentiment_analyzer

def get_irony_analyzer():
    global _irony_analyzer
    if _irony_analyzer is None:
        _irony_analyzer = pipeline("text-classification", model=MODEL_IRONY)
    return _irony_analyzer

# Sentiment mapping helper with irony adjustment
def map_sentiment(label, irony_label=None, irony_score=0.0):
    """Map sentiment labels and adjust for irony/sarcasm."""
    base_sentiment = "Neutral"
    if str(label).lower() in ["label_2", "positive"]:
        base_sentiment = "Positive"
    elif str(label).lower() in ["label_1", "neutral"]:
        base_sentiment = "Neutral"
    elif str(label).lower() in ["label_0", "negative"]:
        base_sentiment = "Negative"

    # Adjust for irony/sarcasm
    if irony_label == "LABEL_1" and irony_score > 0.5:
        if base_sentiment == "Positive":
            return "Negative"
        elif base_sentiment == "Negative":
            return "Positive"
    return base_sentiment

# Text preprocessing
def preprocess_text(text: str) -> str:
    """Clean raw review text before sentiment analysis."""
    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs and HTML tags
    text = re.sub(r"http\S+|www\S+|<.*?>", " ", text)

    # Remove special characters / digits (keep words)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenize with spaCy
    doc = nlp(text)

    # Remove stopwords + lemmatize
    clean_tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and token.text not in stop_words
    ]

    return " ".join(clean_tokens).strip()

# Aspect extraction using spaCy NER and dependency parsing
def extract_aspects(text: str):
    """Extract key aspects from review text using NLP."""
    if not text:
        return []

    doc = nlp(text)

    aspects = []
    aspect_candidates = []

    # Extract noun phrases and named entities
    for chunk in doc.noun_chunks:
        aspect_text = chunk.text.strip()
        if len(aspect_text.split()) <= 3 and len(aspect_text) > 2:
            aspect_candidates.append(aspect_text)

    # Add named entities that might be aspects
    for ent in doc.ents:
        if ent.label_ in ['PRODUCT', 'ORG'] and len(ent.text.split()) <= 3:
            aspect_candidates.append(ent.text.strip())

    # Common aspect keywords for product reviews
    aspect_keywords = {
        'battery', 'camera', 'screen', 'display', 'performance', 'speed',
        'quality', 'price', 'value', 'service', 'support', 'delivery',
        'design', 'build', 'sound', 'speaker', 'microphone', 'charging',
        'storage', 'memory', 'processor', 'cpu', 'gpu', 'graphics',
        'connectivity', 'wifi', 'bluetooth', 'ports', 'durability',
        'reliability', 'usability', 'interface', 'ui', 'ux', 'app',
        'software', 'update', 'compatibility', 'warranty', 'packaging'
    }

    # Filter and prioritize aspects
    for candidate in aspect_candidates:
        candidate_lower = candidate.lower()
        # Direct keyword match
        if any(keyword in candidate_lower for keyword in aspect_keywords):
            aspects.append(candidate)
        # Length and position based filtering
        elif len(candidate.split()) == 1 and len(candidate) > 3:
            aspects.append(candidate)
        elif len(candidate.split()) == 2 and not any(word in stop_words for word in candidate_lower.split()):
            aspects.append(candidate)

    # Remove duplicates and return top aspects
    unique_aspects = list(set(aspects))
    return unique_aspects[:10]  # Limit to top 10 aspects

# Analyze sentiment for specific aspects (optimized)
def analyze_aspect_sentiment(text: str, aspects: list, max_aspects: int = 5):
    """Analyze sentiment for each extracted aspect with performance optimizations."""
    if not text or not aspects:
        return {}

    # Limit number of aspects to prevent excessive processing
    aspects = aspects[:max_aspects]

    doc = nlp(text)
    aspect_sentiments = {}

    # Get analyzers once to avoid repeated initialization
    sentiment_analyzer = get_sentiment_analyzer()
    irony_analyzer = get_irony_analyzer()

    for aspect in aspects:
        # Find sentences containing the aspect
        aspect_sentences = []
        for sent in doc.sents:
            if aspect.lower() in sent.text.lower():
                aspect_sentences.append(sent.text.strip())

        if aspect_sentences:
            # Analyze sentiment of aspect-related sentences
            combined_text = " ".join(aspect_sentences)
            if len(combined_text) > 10:  # Only analyze if substantial text
                try:
                    # Use shorter text for faster processing
                    analysis_text = combined_text[:256]  # Reduced from 512

                    sent_result = sentiment_analyzer(analysis_text)[0]
                    sent_label, sent_score = sent_result["label"], float(sent_result["score"])

                    irony_result = irony_analyzer(analysis_text)[0]
                    irony_label, irony_score = irony_result["label"], float(irony_result["score"])

                    final_sentiment = map_sentiment(sent_label, irony_label, irony_score)
                    aspect_sentiments[aspect] = {
                        'sentiment': final_sentiment,
                        'confidence': sent_score,
                        'irony_score': irony_score,
                        'sentences': aspect_sentences[:2]  # Limit sentences to prevent memory issues
                    }
                except Exception as e:
                    # Fallback to lexicon-based analysis
                    aspect_sentiments[aspect] = {
                        'sentiment': 'Neutral',
                        'confidence': 0.5,
                        'irony_score': 0.0,
                        'sentences': aspect_sentences[:2]
                    }
        else:
            aspect_sentiments[aspect] = {
                'sentiment': 'Neutral',
                'confidence': 0.3,
                'irony_score': 0.0,
                'sentences': []
            }

    return aspect_sentiments

# Generate comprehensive analysis summary
def generate_analysis_summary(aspect_sentiments: dict, overall_sentiment: str = None, overall_confidence: float = 0.0):
    """Generate summary statistics for the analysis."""
    if not aspect_sentiments:
        return {
            'total_aspects': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'overall_sentiment': overall_sentiment or 'Neutral',
            'overall_confidence': overall_confidence,
            'aspect_distribution': {}
        }

    sentiment_counts = Counter(aspect['sentiment'] for aspect in aspect_sentiments.values())

    return {
        'total_aspects': len(aspect_sentiments),
        'positive_count': sentiment_counts.get('Positive', 0),
        'negative_count': sentiment_counts.get('Negative', 0),
        'neutral_count': sentiment_counts.get('Neutral', 0),
        'overall_sentiment': overall_sentiment or 'Neutral',
        'overall_confidence': overall_confidence,
        'aspect_distribution': dict(sentiment_counts)
    }

# Enhanced keyword highlighting for aspects
def highlight_aspects(text: str, aspect_sentiments: dict):
    """Highlight aspect terms in the original text."""
    if not text or not aspect_sentiments:
        return text

    # Create a copy of the text for highlighting
    highlighted_text = text

    # Sort aspects by length (longest first) to avoid partial replacements
    sorted_aspects = sorted(aspect_sentiments.keys(), key=len, reverse=True)

    for aspect in sorted_aspects:
        sentiment_info = aspect_sentiments[aspect]
        color = {
            'Positive': 'lightgreen',
            'Negative': 'lightcoral',
            'Neutral': 'lightyellow'
        }.get(sentiment_info['sentiment'], 'lightgray')

        # Use word boundaries to avoid partial replacements
        pattern = r'\b' + re.escape(aspect) + r'\b'
        replacement = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px;">{aspect}</span>'
        highlighted_text = re.sub(pattern, replacement, highlighted_text, flags=re.IGNORECASE)

    return highlighted_text

# Keyword highlighting helper - Highlight sentiment-bearing words from lexicon
def highlight_keywords(text, sentiment):
    """Highlight sentiment-bearing words from lexicon in the text."""
    if not text:
        return text

    # Process text with spaCy for tokenization
    doc = nlp(text)

    highlighted_words = []
    for token in doc:
        word = token.text
        lemma = token.lemma_.lower()

        # Skip punctuation and spaces
        if token.is_punct or token.is_space:
            highlighted_words.append(word)
            continue

        should_highlight = False
        highlight_color = ""

        if sentiment == "Positive" and lemma in positive_words:
            should_highlight = True
            highlight_color = "lightgreen"
        elif sentiment == "Negative" and lemma in negative_words:
            should_highlight = True
            highlight_color = "lightcoral"
        # For neutral, no highlighting or highlight neutral words if needed
        # elif sentiment == "Neutral":
        #     should_highlight = True
        #     highlight_color = "lightyellow"

        if should_highlight:
            highlighted_words.append(f'<span style="background-color: {highlight_color};">{word}</span>')
        else:
            highlighted_words.append(word)

    return " ".join(highlighted_words)

# Simple cache for analysis results to prevent repeated processing
_analysis_cache = {}

def clear_analysis_cache():
    """Clear the analysis cache to free memory."""
    global _analysis_cache
    _analysis_cache.clear()

def analyze_review_detailed(review_text: str, overall_sentiment: str = None, overall_confidence: float = 0.0):
    """Perform comprehensive analysis of a review with caching."""
    # Create cache key
    cache_key = hash(review_text.strip().lower())

    # Check cache first
    if cache_key in _analysis_cache:
        cached_result = _analysis_cache[cache_key].copy()
        # Update with new overall sentiment if provided
        if overall_sentiment:
            cached_result['summary']['overall_sentiment'] = overall_sentiment
            cached_result['summary']['overall_confidence'] = overall_confidence
        return cached_result

    # Preprocess text
    clean_text = preprocess_text(review_text)

    # Extract aspects
    aspects = extract_aspects(review_text)

    # Analyze sentiment for each aspect (with performance limits)
    aspect_sentiments = analyze_aspect_sentiment(review_text, aspects, max_aspects=5)

    # Generate summary
    summary = generate_analysis_summary(aspect_sentiments, overall_sentiment, overall_confidence)

    # Create highlighted text
    highlighted_text = highlight_aspects(review_text, aspect_sentiments)

    result = {
        'original_text': review_text,
        'clean_text': clean_text,
        'highlighted_text': highlighted_text,
        'aspects': aspects,
        'aspect_sentiments': aspect_sentiments,
        'summary': summary
    }

    # Cache the result (limit cache size to prevent memory issues)
    if len(_analysis_cache) < 100:  # Limit cache to 100 entries
        _analysis_cache[cache_key] = result.copy()

    return result
