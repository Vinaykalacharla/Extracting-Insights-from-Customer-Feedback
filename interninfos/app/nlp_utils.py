import html
import json
import logging
import os
import re
from copy import deepcopy
from collections import Counter

import nltk
import numpy as np
import spacy
from nltk.corpus import opinion_lexicon, stopwords
from transformers import pipeline

try:
    from psycopg2.extras import RealDictCursor
except ModuleNotFoundError:
    RealDictCursor = None

try:
    from langdetect import detect_langs
except ModuleNotFoundError:
    detect_langs = None

try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
except ModuleNotFoundError:
    KMeans = None
    TfidfVectorizer = None

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    SentenceTransformer = None

_nltk_ready = False
_spacy_load_attempted = False
stop_words = None
positive_words = None
negative_words = None
nlp = None

# Negation words and phrases
NEGATION_WORDS = {
    'not', 'no', 'never', 'none', 'nothing', 'nowhere', 'neither', 'nor',
    'dont', "don't", 'doesnt', "doesn't", 'didnt', "didn't", 'cant', "can't",
    'cannot', 'wont', "won't", 'wouldnt', "wouldn't", 'shouldnt', "shouldn't",
    'couldnt', "couldn't", 'isnt', "isn't", 'arent', "aren't", 'wasnt', "wasn't",
    'werent', "weren't", 'hasnt', "hasn't", 'havent', "haven't", 'hadnt', "hadn't",
    'aint', "ain't", 'hardly', 'barely', 'scarcely', 'rarely', 'seldom'
}

# Intensifiers and diminishers
INTENSIFIERS = {'very', 'really', 'extremely', 'incredibly', 'absolutely', 'totally', 'completely', 'utterly', 'highly', 'so', 'quite', 'pretty', 'fairly', 'rather'}
DIMINISHERS = {'slightly', 'somewhat', 'kind of', 'sort of', 'a bit', 'a little', 'barely', 'hardly', 'scarcely'}

# Sarcasm indicators
SARCASM_INDICATORS = {
    'oh', 'sure', 'right', 'yeah', 'totally', 'absolutely', 'definitely', 'obviously',
    'clearly', 'apparently', 'evidently', 'obviously', 'naturally', 'of course',
    'as if', 'whatever', 'like', 'duh', 'wow', 'gee', 'gosh', 'oh boy', 'oh dear'
}

EMOTION_LEXICON = {
    'frustration': {'broken', 'frustrated', 'annoying', 'slow', 'stuck', 'worse', 'failed', 'terrible', 'awful'},
    'delight': {'love', 'amazing', 'great', 'excellent', 'smooth', 'fast', 'perfect', 'awesome', 'happy'},
    'trust': {'reliable', 'stable', 'consistent', 'secure', 'dependable', 'trusted', 'solid'},
    'disappointment': {'expected', 'disappointed', 'lacking', 'missing', 'issue', 'problem', 'bad'}
}

INTENT_PATTERNS = {
    'feature_request': [r'\bplease add\b', r'\bwould love\b', r'\bshould have\b', r'\bneed(s)?\b', r'\bfeature request\b'],
    'bug_report': [r'\bbug\b', r'\bcrash(es|ed)?\b', r'\berror\b', r'\bnot working\b', r'\bfails?\b', r'\bbroken\b'],
    'complaint': [r'\bworst\b', r'\bterrible\b', r'\bawful\b', r'\bdisappointed\b', r'\bpoor\b', r'\bunhappy\b'],
    'praise': [r'\blove\b', r'\bamazing\b', r'\bgreat\b', r'\bexcellent\b', r'\bimpressed\b', r'\bawesome\b'],
    'question': [r'\?'],
}

URGENT_TERMS = {
    'urgent', 'immediately', 'asap', 'critical', 'blocked', 'blocking', 'outage',
    'cannot', "can't", 'unable', 'refund', 'refunds', 'cancel', 'canceled', 'fraud'
}

# Load sentiment models
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_BERT = "nlptown/bert-base-multilingual-uncased-sentiment"
MODEL_IRONY = "cardiffnlp/twitter-roberta-base-irony"

# Initialize models (lazy loading to avoid issues)
_sentiment_analyzer = None
_bert_analyzer = None
_irony_analyzer = None
_tokenizer = None
_model = None
_embedding_model = None
ALLOW_MODEL_DOWNLOADS = os.getenv("ALLOW_MODEL_DOWNLOADS", "false").lower() == "true"


def _ensure_nltk_resources():
    global _nltk_ready, stop_words, positive_words, negative_words
    if _nltk_ready:
        return

    resources = {
        "corpora/stopwords": "stopwords",
        "corpora/opinion_lexicon": "opinion_lexicon",
        "sentiment/vader_lexicon.zip": "vader_lexicon",
    }
    for lookup_path, resource_name in resources.items():
        try:
            nltk.data.find(lookup_path)
        except LookupError:
            try:
                nltk.download(resource_name, quiet=True)
            except Exception as exc:
                logging.warning(f"Failed to download NLTK resource '{resource_name}': {exc}")

    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        logging.warning("NLTK stopwords corpus is unavailable; preprocessing fallback will be limited.")
        stop_words = set()

    try:
        positive_words = set(opinion_lexicon.positive())
        negative_words = set(opinion_lexicon.negative())
    except LookupError:
        logging.warning("NLTK opinion lexicon is unavailable; lexicon-based sentiment fallback will be limited.")
        positive_words = set()
        negative_words = set()

    positive_words.update(['must', 'stood'])
    _nltk_ready = True


def get_stop_words():
    _ensure_nltk_resources()
    return stop_words or set()


def get_positive_words():
    _ensure_nltk_resources()
    return positive_words or set()


def get_negative_words():
    _ensure_nltk_resources()
    return negative_words or set()


def get_spacy_model():
    global nlp, _spacy_load_attempted
    if nlp is None and not _spacy_load_attempted:
        _spacy_load_attempted = True
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            logging.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            nlp = None
    return nlp


def _safe_doc(text: str):
    model = get_spacy_model()
    if model is None or not text:
        return None
    return model(text)


def detect_language(text: str):
    text = (text or '').strip()
    if not text:
        return {'language': 'unknown', 'confidence': 0.0}

    if detect_langs is not None:
        try:
            languages = detect_langs(text[:1000])
            if languages:
                best = languages[0]
                return {'language': best.lang, 'confidence': round(float(best.prob), 3)}
        except Exception:
            pass

    ascii_ratio = sum(1 for ch in text if ord(ch) < 128) / max(len(text), 1)
    return {
        'language': 'en' if ascii_ratio > 0.9 else 'unknown',
        'confidence': round(ascii_ratio, 3)
    }


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None and SentenceTransformer is not None:
        try:
            kwargs = {}
            if not ALLOW_MODEL_DOWNLOADS:
                kwargs["local_files_only"] = True
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2', **kwargs)
        except TypeError:
            if ALLOW_MODEL_DOWNLOADS:
                _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            else:
                logging.warning("Embedding model downloads are disabled and local_files_only is unsupported.")
                _embedding_model = None
        except Exception as exc:
            logging.warning(f"Failed to load embedding model: {exc}")
            _embedding_model = None
    return _embedding_model

def detect_negation_scope(text: str) -> list:
    """Detect negation words and their scope in the text."""
    doc = _safe_doc(text)
    if doc is None:
        return []
    negations = []

    for i, token in enumerate(doc):
        if token.lemma_.lower() in NEGATION_WORDS:
            # Find the scope of negation (typically 3-5 words after negation)
            start_idx = i
            end_idx = min(i + 6, len(doc))  # Extended scope

            # Skip punctuation and conjunctions to find actual scope
            scope_tokens = []
            for j in range(start_idx + 1, end_idx):
                if not doc[j].is_punct and doc[j].text.lower() not in ['and', 'or', 'but', 'so', 'then']:
                    scope_tokens.append(doc[j])
                    if len(scope_tokens) >= 4:  # Extended scope to 4 words
                        break

            negations.append({
                'negation_word': token.text,
                'position': i,
                'scope_start': start_idx + 1,
                'scope_end': start_idx + 1 + len(scope_tokens),
                'scope_tokens': scope_tokens
            })

    return negations

def detect_sarcasm_indicators(text: str) -> dict:
    indicators_found = []
    doc = _safe_doc(text)
    if doc is None:
        return {
            'indicators': [],
            'contradiction_score': 0.0,
            'has_sarcasm_potential': False
        }
    positive_lexicon = get_positive_words()
    negative_lexicon = get_negative_words()

    # Check for sarcasm indicators
    for token in doc:
        if token.lemma_.lower() in SARCASM_INDICATORS:
            indicators_found.append(token.text)

    # Check for contradictory sentiment patterns
    positive_count = 0
    negative_count = 0
    contradiction_score = 0.0

    for token in doc:
        lemma = token.lemma_.lower()
        if lemma in positive_lexicon:
            positive_count += 1
        elif lemma in negative_lexicon:
            negative_count += 1

    # High contradiction score if both positive and negative words are present
    if positive_count > 0 and negative_count > 0:
        contradiction_score = min(positive_count, negative_count) / max(positive_count, negative_count)

    # Additional sarcasm patterns
    text_lower = text.lower()

    # Check for exaggerated politeness or obvious statements
    exaggerated_patterns = [
        r'oh.*so.*good', r'how.*wonderful', r'such.*amazing', r'too.*perfect',
        r'annoyingly.*good', r'frustratingly.*nice', r'irritatingly.*perfect',
        r'just.*fantastic', r'love.*it', r'great.*job', r'amazing.*work'
    ]

    for pattern in exaggerated_patterns:
        if re.search(pattern, text_lower):
            contradiction_score += 0.3

    # Check for mixed sentiments in the same sentence
    for sent in doc.sents:
        sent_text = sent.text.lower()
        sent_positive = sum(1 for token in sent if token.lemma_.lower() in positive_lexicon)
        sent_negative = sum(1 for token in sent if token.lemma_.lower() in negative_lexicon)
        if sent_positive > 0 and sent_negative > 0:
            contradiction_score += 0.2

    # Check for obvious positive words with negative context
    obvious_sarcasm_patterns = [
        r'fantastic.*not', r'wonderful.*but', r'amazing.*however',
        r'perfect.*except', r'excellent.*although', r'great.*unfortunately'
    ]

    for pattern in obvious_sarcasm_patterns:
        if re.search(pattern, text_lower):
            contradiction_score += 0.4
            indicators_found.append("obvious_sarcasm")

    return {
        'indicators': indicators_found,
        'contradiction_score': min(contradiction_score, 1.0),
        'has_sarcasm_potential': len(indicators_found) > 0 or contradiction_score > 0.3
    }

def lexicon_sentiment_with_negation(text: str) -> dict:
    """Perform lexicon-based sentiment analysis with negation handling."""
    positive_lexicon = get_positive_words()
    negative_lexicon = get_negative_words()
    doc = _safe_doc(text)
    if doc is None:
        # Fallback: simple word-based analysis
        words = text.lower().split()
        positive_count = sum(1 for w in words if w in positive_lexicon)
        negative_count = sum(1 for w in words if w in negative_lexicon)
        if positive_count > negative_count:
            return {'sentiment': 'Positive', 'score': 0.5, 'confidence': 0.5, 'positive_words': [], 'negative_words': [], 'negated_words': [], 'negations_detected': 0}
        elif negative_count > positive_count:
            return {'sentiment': 'Negative', 'score': -0.5, 'confidence': 0.5, 'positive_words': [], 'negative_words': [], 'negated_words': [], 'negations_detected': 0}
        else:
            return {'sentiment': 'Neutral', 'score': 0.0, 'confidence': 0.5, 'positive_words': [], 'negative_words': [], 'negated_words': [], 'negations_detected': 0}
    negations = detect_negation_scope(text)

    sentiment_score = 0.0
    positive_words_found = []
    negative_words_found = []
    negated_words = []

    for i, token in enumerate(doc):
        if token.is_punct or token.is_space:
            continue

        lemma = token.lemma_.lower()
        word_sentiment = 0.0

        # Check if word is in sentiment lexicons
        if lemma in positive_lexicon:
            word_sentiment = 1.0
            positive_words_found.append(token.text)
        elif lemma in negative_lexicon:
            word_sentiment = -1.0
            negative_words_found.append(token.text)

        # Apply intensifiers/diminishers
        if word_sentiment != 0.0:
            # Check previous words for intensifiers/diminishers
            for j in range(max(0, i-2), i):
                prev_lemma = doc[j].lemma_.lower()
                if prev_lemma in INTENSIFIERS:
                    word_sentiment *= 1.5
                    break
                elif prev_lemma in DIMINISHERS:
                    word_sentiment *= 0.5
                    break

        # Check if word is within negation scope
        is_negated = False
        for negation in negations:
            if negation['scope_start'] <= i < negation['scope_end']:
                is_negated = True
                negated_words.append(token.text)
                break

        if is_negated:
            word_sentiment *= -1  # Flip sentiment

        sentiment_score += word_sentiment

    # Normalize score
    total_sentiment_words = len(positive_words_found) + len(negative_words_found)
    if total_sentiment_words > 0:
        sentiment_score = sentiment_score / total_sentiment_words

    # Determine final sentiment
    if sentiment_score > 0.1:
        final_sentiment = "Positive"
    elif sentiment_score < -0.1:
        final_sentiment = "Negative"
    else:
        final_sentiment = "Neutral"

    return {
        'sentiment': final_sentiment,
        'score': sentiment_score,
        'confidence': min(abs(sentiment_score), 1.0),
        'positive_words': positive_words_found,
        'negative_words': negative_words_found,
        'negated_words': negated_words,
        'negations_detected': len(negations)
    }

def enhanced_sentiment_analysis(text: str) -> dict:
    """Enhanced sentiment analysis combining multiple approaches with negation and sarcasm handling."""
    if not text or len(text.strip()) < 3:
        return {
            'sentiment': 'Neutral',
            'confidence': 0.5,
            'methods_used': [],
            'negation_info': {},
            'sarcasm_info': {}
        }

    results = {}

    # Detect negation and sarcasm
    negation_info = detect_negation_scope(text)
    sarcasm_info = detect_sarcasm_indicators(text)

    # Lexicon-based analysis with negation
    lexicon_result = lexicon_sentiment_with_negation(text)
    results['lexicon'] = lexicon_result

    # Transformer-based analysis (existing)
    try:
        sentiment_analyzer = get_sentiment_analyzer()
        bert_analyzer = get_bert_analyzer()
        irony_analyzer = get_irony_analyzer()

        if sentiment_analyzer is None or bert_analyzer is None or irony_analyzer is None:
            raise RuntimeError("Transformer analyzers are unavailable")

        sent_roberta = sentiment_analyzer(text[:256])[0]
        sent_bert = bert_analyzer(text[:256])[0]
        irony_result = irony_analyzer(text[:256])[0]

        transformer_result = ensemble_sentiment(
            {'label': sent_roberta['label'], 'score': float(sent_roberta['score'])},
            {'label': sent_bert['label'], 'score': float(sent_bert['score'])},
            irony_result["label"], float(irony_result["score"])
        )
        results['transformer'] = {
            'sentiment': transformer_result[0],
            'confidence': transformer_result[1]
        }
    except Exception as e:
        logging.warning(f"Transformer analysis failed: {e}")
        results['transformer'] = {'sentiment': 'Neutral', 'confidence': 0.3}

    # Weighted ensemble with negation/sarcasm adjustments
    weights = {
        'lexicon': 0.4,  # Higher weight for lexicon with negation handling
        'transformer': 0.6
    }

    # Convert sentiments to scores
    sentiment_to_score = {'Positive': 1.0, 'Neutral': 0.0, 'Negative': -1.0}

    lexicon_score = sentiment_to_score.get(results['lexicon']['sentiment'], 0.0)
    transformer_score = sentiment_to_score.get(results['transformer']['sentiment'], 0.0)

    # Check for mixed sentiments (both positive and negative words present)
    has_mixed_sentiments = (
        len(results['lexicon']['positive_words']) > 0 and
        len(results['lexicon']['negative_words']) > 0
    )

    # If mixed sentiments detected, adjust weights to favor lexicon analysis
    if has_mixed_sentiments:
        weights['lexicon'] = 0.6
        weights['transformer'] = 0.4

    # Apply sarcasm adjustment
    if sarcasm_info['has_sarcasm_potential']:
        # Reduce confidence when sarcasm is detected
        weights['lexicon'] *= 0.8
        weights['transformer'] *= 0.8

        # If high contradiction, significantly flip the sentiment
        if sarcasm_info['contradiction_score'] > 0.6:
            lexicon_score *= -0.7  # Stronger flip for high contradiction
            transformer_score *= -0.7

    # Calculate weighted score
    final_score = (
        lexicon_score * weights['lexicon'] +
        transformer_score * weights['transformer']
    ) / sum(weights.values())

    # Determine final sentiment with adjusted thresholds for mixed sentiments
    if has_mixed_sentiments:
        # More conservative thresholds for mixed sentiments
        if final_score > 0.3:
            final_sentiment = "Positive"
        elif final_score < -0.3:
            final_sentiment = "Negative"
        else:
            final_sentiment = "Neutral"
    else:
        # Standard thresholds
        if final_score > 0.2:
            final_sentiment = "Positive"
        elif final_score < -0.2:
            final_sentiment = "Negative"
        else:
            final_sentiment = "Neutral"

    # Calculate confidence based on agreement and negation/sarcasm factors
    agreement_bonus = 1.0 if results['lexicon']['sentiment'] == results['transformer']['sentiment'] else 0.7
    negation_penalty = 0.9 if negation_info else 1.0
    sarcasm_penalty = 0.8 if sarcasm_info['has_sarcasm_potential'] else 1.0

    final_confidence = min(
        (abs(final_score) + results['lexicon']['confidence'] + results['transformer']['confidence']) / 3.0 *
        agreement_bonus * negation_penalty * sarcasm_penalty,
        1.0
    )

    return {
        'sentiment': final_sentiment,
        'confidence': final_confidence,
        'score': final_score,
        'methods_used': list(results.keys()),
        'negation_info': {
            'negations_detected': len(negation_info),
            'negated_words': results['lexicon'].get('negated_words', [])
        },
        'sarcasm_info': {
            'indicators_found': sarcasm_info['indicators'],
            'contradiction_score': sarcasm_info['contradiction_score'],
            'has_sarcasm_potential': sarcasm_info['has_sarcasm_potential']
        },
        'component_results': results
    }

def get_sentiment_analyzer():
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        try:
            pipeline_kwargs = {"device": -1}
            if not ALLOW_MODEL_DOWNLOADS:
                pipeline_kwargs["local_files_only"] = True
            _sentiment_analyzer = pipeline("sentiment-analysis",
                                        model=MODEL_NAME,
                                        tokenizer=MODEL_NAME,
                                        **pipeline_kwargs)
        except (OSError, ValueError, TypeError) as e:
            logging.warning(f"Failed to load sentiment analyzer: {e}")
            _sentiment_analyzer = None
    return _sentiment_analyzer

def get_bert_analyzer():
    global _bert_analyzer
    if _bert_analyzer is None:
        try:
            pipeline_kwargs = {"device": -1}
            if not ALLOW_MODEL_DOWNLOADS:
                pipeline_kwargs["local_files_only"] = True
            _bert_analyzer = pipeline("sentiment-analysis", model=MODEL_BERT, **pipeline_kwargs)
        except (OSError, ValueError, TypeError) as e:
            logging.warning(f"Failed to load BERT analyzer: {e}")
            _bert_analyzer = None
    return _bert_analyzer

def get_irony_analyzer():
    global _irony_analyzer
    if _irony_analyzer is None:
        try:
            pipeline_kwargs = {"device": -1}
            if not ALLOW_MODEL_DOWNLOADS:
                pipeline_kwargs["local_files_only"] = True
            _irony_analyzer = pipeline("text-classification", model=MODEL_IRONY, **pipeline_kwargs)
        except (OSError, ValueError, TypeError) as e:
            logging.warning(f"Failed to load irony analyzer: {e}")
            _irony_analyzer = None
    return _irony_analyzer

# Sentiment mapping helper with irony adjustment
def map_sentiment(label, irony_label=None, irony_score=0.0):
    """Map sentiment labels and adjust for irony/sarcasm."""
    base_sentiment = "Neutral"
    if str(label).lower() in ["label_2", "positive", "5 star"]:
        base_sentiment = "Positive"
    elif str(label).lower() in ["label_1", "neutral", "4 star", "3 star"]:
        base_sentiment = "Neutral"
    elif str(label).lower() in ["label_0", "negative", "1 star", "2 star"]:
        base_sentiment = "Negative"

    # Adjust for irony/sarcasm
    if irony_label == "LABEL_1" and irony_score > 0.5:
        if base_sentiment == "Positive":
            return "Negative"
        elif base_sentiment == "Negative":
            return "Positive"
    return base_sentiment


def ensemble_sentiment(sent_roberta, sent_bert, irony_label=None, irony_score=0.0):
    """Combine sentiments from RoBERTa and BERT models."""
    # Map to scores: Positive=1, Neutral=0, Negative=-1
    def label_to_score(label):
        if label == "Positive":
            return 1.0
        elif label == "Neutral":
            return 0.0
        elif label == "Negative":
            return -1.0
        return 0.0

    roberta_score = label_to_score(map_sentiment(sent_roberta['label']))
    bert_score = label_to_score(map_sentiment(sent_bert['label']))

    # Weighted average (equal weights for simplicity)
    combined_score = (roberta_score + bert_score) / 2

    # Map back to sentiment
    if combined_score > 0.3:
        final_sentiment = "Positive"
    elif combined_score < -0.3:
        final_sentiment = "Negative"
    else:
        final_sentiment = "Neutral"

    # Adjust for irony
    if irony_label == "LABEL_1" and irony_score > 0.5:
        if final_sentiment == "Positive":
            final_sentiment = "Negative"
        elif final_sentiment == "Negative":
            final_sentiment = "Positive"

    # Combined confidence as average
    combined_confidence = (sent_roberta['score'] + sent_bert['score']) / 2

    return final_sentiment, combined_confidence

# Text preprocessing
def preprocess_text(text: str) -> str:
    """Clean raw review text before sentiment analysis."""
    if not text:
        return ""
    stopword_lexicon = get_stop_words()

    # Lowercase
    text = text.lower()

    # Remove URLs and HTML tags
    text = re.sub(r"http\S+|www\S+|<.*?>", " ", text)

    # Remove special characters / digits (keep words)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenize with spaCy
    doc = _safe_doc(text)
    if doc is None:
        # Fallback: simple tokenization
        tokens = text.split()
        clean_tokens = [t for t in tokens if t not in stopword_lexicon]
        return " ".join(clean_tokens).strip()

    # Remove stopwords + lemmatize
    clean_tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and token.text not in stopword_lexicon
    ]

    return " ".join(clean_tokens).strip()

# Aspect extraction using spaCy NER and dependency parsing
def extract_aspects(text: str):
    """Extract key aspects from review text using NLP."""
    if not text:
        return []
    stopword_lexicon = get_stop_words()

    doc = _safe_doc(text)
    if doc is None:
        fallback_aspects = []
        aspect_keywords = {
            'battery', 'camera', 'screen', 'display', 'performance', 'speed',
            'quality', 'price', 'value', 'service', 'support', 'delivery',
            'design', 'build', 'sound', 'speaker', 'microphone', 'charging',
            'storage', 'memory', 'processor', 'cpu', 'gpu', 'graphics',
            'connectivity', 'wifi', 'bluetooth', 'ports', 'durability',
            'reliability', 'usability', 'interface', 'ui', 'ux', 'app',
            'software', 'update', 'compatibility', 'warranty', 'packaging'
        }
        words = re.findall(r"[a-zA-Z]+", text.lower())
        for keyword in aspect_keywords:
            if keyword in words:
                fallback_aspects.append(keyword)
        return sorted(set(fallback_aspects))[:10]

    aspects = []
    aspect_candidates = []

    def clean_aspect(aspect):
        """Remove leading determiners from aspects."""
        words = aspect.split()
        if words and words[0].lower() in ['the', 'a', 'an', 'this', 'that', 'these', 'those']:
            return ' '.join(words[1:])
        return aspect

    # Extract noun phrases and named entities
    for chunk in doc.noun_chunks:
        aspect_text = clean_aspect(chunk.text.strip())
        if len(aspect_text.split()) <= 3 and len(aspect_text) > 2:
            aspect_candidates.append(aspect_text)

    # Add named entities that might be aspects
    for ent in doc.ents:
        if ent.label_ in ['PRODUCT', 'ORG'] and len(ent.text.split()) <= 3:
            aspect_text = clean_aspect(ent.text.strip())
            aspect_candidates.append(aspect_text)

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
        elif len(candidate.split()) == 2 and not any(word in stopword_lexicon for word in candidate_lower.split()):
            aspects.append(candidate)

    # Remove duplicates and return top aspects
    unique_aspects = list(set(aspects))
    return unique_aspects[:10]  # Limit to top 10 aspects

# Analyze sentiment for specific aspects (optimized)
def analyze_aspect_sentiment(text: str, aspects: list, max_aspects: int = 5):
    """Analyze sentiment for each extracted aspect with performance optimizations."""
    logger = logging.getLogger(__name__)
    if not text or not aspects:
        return {}

    # Limit number of aspects to prevent excessive processing
    aspects = aspects[:max_aspects]

    doc = _safe_doc(text)
    if doc is None:
        fallback_result = lexicon_sentiment_with_negation(text)
        return {
            aspect: {
                'sentiment': fallback_result['sentiment'],
                'confidence': fallback_result['confidence'],
                'irony_score': 0.0,
                'sentences': [text[:280]]
            }
            for aspect in aspects
        }

    aspect_sentiments = {}

    # Get analyzers once to avoid repeated initialization
    sentiment_analyzer = get_sentiment_analyzer()
    bert_analyzer = get_bert_analyzer()
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
                    # Check if analyzers are available
                    if sentiment_analyzer is not None and bert_analyzer is not None and irony_analyzer is not None:
                        # Use shorter text for faster processing
                        analysis_text = combined_text[:256]  # Reduced from 512

                        sent_roberta = sentiment_analyzer(analysis_text)[0]
                        sent_bert = bert_analyzer(analysis_text)[0]
                        irony_result = irony_analyzer(analysis_text)[0]
                        irony_label, irony_score = irony_result["label"], float(irony_result["score"])

                        final_sentiment, combined_confidence = ensemble_sentiment(
                            {'label': sent_roberta['label'], 'score': float(sent_roberta['score'])},
                            {'label': sent_bert['label'], 'score': float(sent_bert['score'])},
                            irony_label, irony_score
                        )
                        aspect_sentiments[aspect] = {
                            'sentiment': final_sentiment,
                            'confidence': combined_confidence,
                            'irony_score': irony_score,
                            'sentences': aspect_sentences[:2]  # Limit sentences to prevent memory issues
                        }
                    else:
                        # Fallback to lexicon-based analysis if models failed to load
                        lexicon_result = lexicon_sentiment_with_negation(combined_text)
                        aspect_sentiments[aspect] = {
                            'sentiment': lexicon_result['sentiment'],
                            'confidence': lexicon_result['confidence'],
                            'irony_score': 0.0,
                            'sentences': aspect_sentences[:2]
                        }
                except Exception as e:
                    logger.error(f"Error analyzing sentiment for aspect '{aspect}': {e}")
                    # Fallback to lexicon-based analysis
                    lexicon_result = lexicon_sentiment_with_negation(combined_text)
                    aspect_sentiments[aspect] = {
                        'sentiment': lexicon_result['sentiment'],
                        'confidence': lexicon_result['confidence'],
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


def detect_intent(text: str, overall_sentiment: str | None = None):
    text_lower = (text or '').lower()
    scores = {}
    for intent, patterns in INTENT_PATTERNS.items():
        scores[intent] = sum(1 for pattern in patterns if re.search(pattern, text_lower))

    top_intent = max(scores, key=scores.get) if scores else 'general_feedback'
    if scores.get(top_intent, 0) == 0:
        if (overall_sentiment or '').lower() == 'negative':
            top_intent = 'complaint'
        elif (overall_sentiment or '').lower() == 'positive':
            top_intent = 'praise'
        else:
            top_intent = 'general_feedback'

    return {
        'label': top_intent,
        'scores': scores
    }


def detect_emotions(text: str):
    tokens = re.findall(r"[a-zA-Z']+", (text or '').lower())
    if not tokens:
        return {'primary_emotion': 'neutral', 'scores': {}}

    emotion_scores = {}
    for emotion, words in EMOTION_LEXICON.items():
        matches = sum(1 for token in tokens if token in words)
        emotion_scores[emotion] = round(matches / max(len(tokens), 1), 3)

    if any(emotion_scores.values()):
        primary_emotion = max(emotion_scores, key=emotion_scores.get)
    else:
        primary_emotion = 'neutral'

    return {
        'primary_emotion': primary_emotion,
        'scores': emotion_scores
    }


def assess_urgency(text: str, overall_sentiment: str | None = None):
    text = text or ''
    text_lower = text.lower()
    urgency_hits = sum(1 for term in URGENT_TERMS if term in text_lower)
    exclamations = text.count('!')
    uppercase_ratio = 0.0
    alpha_chars = [c for c in text if c.isalpha()]
    if alpha_chars:
        uppercase_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)

    raw_score = min(1.0, urgency_hits * 0.22 + exclamations * 0.06 + uppercase_ratio * 0.6)
    if (overall_sentiment or '').lower() == 'negative':
        raw_score = min(1.0, raw_score + 0.12)

    if raw_score >= 0.7:
        level = 'high'
    elif raw_score >= 0.35:
        level = 'medium'
    else:
        level = 'low'

    return {
        'level': level,
        'score': round(raw_score, 3)
    }


def build_advanced_insights(review_text: str, aspect_sentiments: dict, overall_sentiment: str, overall_confidence: float):
    language = detect_language(review_text)
    intent = detect_intent(review_text, overall_sentiment)
    emotions = detect_emotions(review_text)
    urgency = assess_urgency(review_text, overall_sentiment)

    positive_aspects = [aspect for aspect, meta in aspect_sentiments.items() if meta.get('sentiment') == 'Positive']
    negative_aspects = [aspect for aspect, meta in aspect_sentiments.items() if meta.get('sentiment') == 'Negative']
    neutral_aspects = [aspect for aspect, meta in aspect_sentiments.items() if meta.get('sentiment') == 'Neutral']

    risk_flags = []
    if urgency['level'] == 'high':
        risk_flags.append('High urgency customer signal')
    if negative_aspects:
        risk_flags.append('Negative aspect cluster detected')
    if overall_confidence < 0.45:
        risk_flags.append('Low-confidence prediction; review manually')
    if intent['label'] == 'bug_report':
        risk_flags.append('Potential product defect report')
    if intent['label'] == 'feature_request':
        risk_flags.append('Feature demand signal')

    experience_score = 50
    if (overall_sentiment or '').lower() == 'positive':
        experience_score += 25
    elif (overall_sentiment or '').lower() == 'negative':
        experience_score -= 25
    experience_score += min(len(positive_aspects) * 6, 18)
    experience_score -= min(len(negative_aspects) * 8, 24)
    experience_score -= 10 if urgency['level'] == 'high' else 0
    experience_score = max(0, min(100, experience_score))

    if negative_aspects or urgency['level'] == 'high':
        priority = 'investigate_now'
    elif intent['label'] == 'feature_request':
        priority = 'roadmap_candidate'
    elif positive_aspects:
        priority = 'retain_strength'
    else:
        priority = 'monitor'

    recommended_actions = []
    if negative_aspects:
        recommended_actions.append(f"Investigate {', '.join(negative_aspects[:3])}")
    if intent['label'] == 'feature_request':
        recommended_actions.append('Route to product planning for feature triage')
    if intent['label'] == 'bug_report':
        recommended_actions.append('Create engineering bug ticket with reproduction context')
    if positive_aspects:
        recommended_actions.append(f"Preserve strengths in {', '.join(positive_aspects[:2])}")
    if not recommended_actions:
        recommended_actions.append('Monitor similar reviews for clearer pattern formation')

    impact_score = 35.0
    if (overall_sentiment or '').lower() == 'negative':
        impact_score += 28.0
    elif (overall_sentiment or '').lower() == 'positive':
        impact_score -= 10.0
    impact_score += urgency['score'] * 25.0
    impact_score += min(len(negative_aspects) * 7.0, 21.0)
    impact_score += 8.0 if intent['label'] in {'bug_report', 'complaint'} else 0.0
    impact_score -= min(len(positive_aspects) * 4.0, 12.0)
    impact_score = round(max(0.0, min(100.0, impact_score)), 2)

    return {
        'language': language,
        'intent': intent,
        'emotion_profile': emotions,
        'urgency': urgency,
        'experience_score': experience_score,
        'impact_score': impact_score,
        'priority': priority,
        'risk_flags': risk_flags,
        'positive_aspects': positive_aspects,
        'negative_aspects': negative_aspects,
        'neutral_aspects': neutral_aspects,
        'recommended_actions': recommended_actions[:3]
    }


def summarize_cluster(reviews: list[dict]):
    if not reviews:
        return {
            'summary': 'No recurring issue summary available.',
            'top_terms': [],
            'dominant_sentiment': 'Neutral'
        }

    token_counter = Counter()
    sentiment_counter = Counter()
    for review in reviews:
        text = preprocess_text(review.get('review_text', ''))
        token_counter.update(token for token in text.split() if len(token) > 2)
        sentiment_counter.update([(review.get('overall_sentiment') or 'Neutral').title()])

    top_terms = [term for term, _ in token_counter.most_common(5)]
    dominant_sentiment = sentiment_counter.most_common(1)[0][0] if sentiment_counter else 'Neutral'
    phrase = ", ".join(top_terms[:3]) if top_terms else "mixed issues"
    summary = f"Recurring {dominant_sentiment.lower()} feedback centered on {phrase}."
    return {
        'summary': summary,
        'top_terms': top_terms,
        'dominant_sentiment': dominant_sentiment
    }


def cluster_reviews_by_similarity(reviews: list[dict], max_clusters: int = 5):
    if not reviews:
        return []

    texts = [review.get('review_text', '') for review in reviews if review.get('review_text')]
    if len(texts) < 2 or KMeans is None:
        return [{
            'cluster_id': 0,
            'size': len(reviews),
            'summary': summarize_cluster(reviews)['summary'],
            'top_terms': summarize_cluster(reviews)['top_terms'],
            'dominant_sentiment': summarize_cluster(reviews)['dominant_sentiment'],
            'review_ids': [review.get('review_id') for review in reviews if review.get('review_id') is not None],
            'method': 'fallback'
        }]

    embedding_model = get_embedding_model()
    method = 'embeddings' if embedding_model is not None else 'tfidf'
    if embedding_model is not None:
        matrix = np.array(embedding_model.encode(texts, show_progress_bar=False))
    else:
        if TfidfVectorizer is None:
            return [{
                'cluster_id': 0,
                'size': len(reviews),
                'summary': summarize_cluster(reviews)['summary'],
                'top_terms': summarize_cluster(reviews)['top_terms'],
                'dominant_sentiment': summarize_cluster(reviews)['dominant_sentiment'],
                'review_ids': [review.get('review_id') for review in reviews if review.get('review_id') is not None],
                'method': 'fallback'
            }]
        vectorizer = TfidfVectorizer(stop_words='english', max_features=600, ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(texts)
    num_clusters = max(2, min(max_clusters, len(texts)))
    model = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    labels = model.fit_predict(matrix)

    grouped: dict[int, list[dict]] = {}
    valid_reviews = [review for review in reviews if review.get('review_text')]
    for label, review in zip(labels, valid_reviews):
        grouped.setdefault(int(label), []).append(review)

    clusters = []
    for cluster_id, cluster_reviews in grouped.items():
        summary = summarize_cluster(cluster_reviews)
        clusters.append({
            'cluster_id': cluster_id,
            'size': len(cluster_reviews),
            'summary': summary['summary'],
            'top_terms': summary['top_terms'],
            'dominant_sentiment': summary['dominant_sentiment'],
            'review_ids': [review.get('review_id') for review in cluster_reviews if review.get('review_id') is not None],
            'method': method
        })

    clusters.sort(key=lambda item: item['size'], reverse=True)
    return clusters


def compute_aspect_trends(review_analyses: list[dict]):
    if not review_analyses:
        return []

    aspect_counts = {}
    split_index = max(1, len(review_analyses) // 2)
    recent = review_analyses[:split_index]
    older = review_analyses[split_index:]

    def accumulate(target, items):
        for item in items:
            for aspect, meta in (item.get('aspect_sentiments') or {}).items():
                row = target.setdefault(aspect, {'recent': 0, 'previous': 0, 'negative_recent': 0, 'positive_recent': 0})
                if meta.get('sentiment') == 'Negative':
                    row['negative_recent'] += 1
                if meta.get('sentiment') == 'Positive':
                    row['positive_recent'] += 1

    for item in recent:
        for aspect in (item.get('aspect_sentiments') or {}).keys():
            aspect_counts.setdefault(aspect, {'recent': 0, 'previous': 0, 'negative_recent': 0, 'positive_recent': 0})
            aspect_counts[aspect]['recent'] += 1
    for item in older:
        for aspect in (item.get('aspect_sentiments') or {}).keys():
            aspect_counts.setdefault(aspect, {'recent': 0, 'previous': 0, 'negative_recent': 0, 'positive_recent': 0})
            aspect_counts[aspect]['previous'] += 1

    accumulate(aspect_counts, recent)

    trends = []
    for aspect, counts in aspect_counts.items():
        delta = counts['recent'] - counts['previous']
        direction = 'rising' if delta > 0 else 'falling' if delta < 0 else 'stable'
        trends.append({
            'aspect': aspect,
            'recent_mentions': counts['recent'],
            'previous_mentions': counts['previous'],
            'delta': delta,
            'direction': direction,
            'negative_recent': counts['negative_recent'],
            'positive_recent': counts['positive_recent']
        })

    trends.sort(key=lambda item: (abs(item['delta']), item['negative_recent'], item['recent_mentions']), reverse=True)
    return trends[:12]


def get_model_health():
    return {
        'spacy_ready': get_spacy_model() is not None,
        'sentiment_model_ready': _sentiment_analyzer is not None,
        'bert_model_ready': _bert_analyzer is not None,
        'irony_model_ready': _irony_analyzer is not None,
        'language_detection_ready': detect_langs is not None,
        'clustering_ready': TfidfVectorizer is not None and KMeans is not None,
        'embedding_clustering_ready': _embedding_model is not None
    }


def generate_alert_candidates(review_analyses: list[dict]):
    if not review_analyses:
        return []

    alerts = []
    urgent_reviews = [
        analysis for analysis in review_analyses
        if (analysis.get('advanced_insights', {}).get('urgency', {}).get('level') == 'high')
    ]
    if len(urgent_reviews) >= 3:
        alerts.append({
            'alert_type': 'urgency_spike',
            'severity': 'high',
            'title': 'High urgency review spike detected',
            'message': f'{len(urgent_reviews)} recent reviews were marked high urgency.',
            'payload': {
                'review_ids': [item.get('review_id') for item in urgent_reviews[:10]],
                'count': len(urgent_reviews)
            }
        })

    high_impact_reviews = [
        analysis for analysis in review_analyses
        if float(analysis.get('advanced_insights', {}).get('impact_score') or 0) >= 75
    ]
    if len(high_impact_reviews) >= 3:
        alerts.append({
            'alert_type': 'impact_spike',
            'severity': 'high',
            'title': 'High-impact negative feedback cluster',
            'message': f'{len(high_impact_reviews)} reviews crossed the high-impact threshold.',
            'payload': {
                'review_ids': [item.get('review_id') for item in high_impact_reviews[:10]],
                'count': len(high_impact_reviews)
            }
        })

    clusters = cluster_reviews_by_similarity(review_analyses, max_clusters=5)
    if clusters:
        top_cluster = clusters[0]
        if top_cluster.get('size', 0) >= 4 and top_cluster.get('dominant_sentiment', '').lower() == 'negative':
            alerts.append({
                'alert_type': 'recurring_issue_cluster',
                'severity': 'medium',
                'title': 'Recurring negative issue cluster detected',
                'message': top_cluster.get('summary') or 'A recurring negative cluster needs review.',
                'payload': top_cluster
            })

    trends = compute_aspect_trends(review_analyses)
    rising_negative = [
        trend for trend in trends
        if trend.get('direction') == 'rising' and trend.get('negative_recent', 0) >= 2
    ]
    if rising_negative:
        top_trend = rising_negative[0]
        alerts.append({
            'alert_type': 'aspect_negative_trend',
            'severity': 'medium',
            'title': f"Negative trend rising for {top_trend['aspect']}",
            'message': f"Aspect '{top_trend['aspect']}' is rising with delta {top_trend['delta']} and {top_trend['negative_recent']} recent negative mentions.",
            'payload': top_trend
        })

    return alerts

# Enhanced keyword highlighting for aspects
def _apply_highlight_spans(text: str, spans: list[tuple[int, int, str]]):
    if not text:
        return ""
    if not spans:
        return html.escape(text)

    parts = []
    last_end = 0
    for start, end, replacement in sorted(spans, key=lambda item: item[0]):
        if start < last_end:
            continue
        parts.append(html.escape(text[last_end:start]))
        parts.append(replacement)
        last_end = end
    parts.append(html.escape(text[last_end:]))
    return "".join(parts)


def highlight_aspects(text: str, aspect_sentiments: dict):
    """Highlight aspect terms in the original text."""
    if not text or not aspect_sentiments:
        return html.escape(text or "")

    # Sort aspects by length (longest first) to avoid partial replacements
    sorted_aspects = sorted(aspect_sentiments.keys(), key=len, reverse=True)
    replacements = []

    for aspect in sorted_aspects:
        if not aspect:
            continue
        sentiment_info = aspect_sentiments[aspect]
        color = {
            'Positive': 'lightgreen',
            'Negative': 'lightcoral',
            'Neutral': 'lightyellow'
        }.get(sentiment_info['sentiment'], 'lightgray')

        # Use word boundaries to avoid partial replacements
        pattern = r'\b' + re.escape(aspect) + r'\b'
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            start, end = match.span()
            if any(start < existing_end and end > existing_start for existing_start, existing_end, _ in replacements):
                continue
            matched_text = match.group(0)
            replacement = (
                f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px;">'
                f'{html.escape(matched_text)}</span>'
            )
            replacements.append((start, end, replacement))

    return _apply_highlight_spans(text, replacements)

# Keyword highlighting helper - Highlight sentiment-bearing words from opinion lexicon
def highlight_keywords(text, sentiment=None):
    """Highlight sentiment-bearing words from opinion lexicon in the text with appropriate colors."""
    if not text:
        return ""

    try:
        # Process text with spaCy for tokenization
        doc = _safe_doc(text)
        if doc is None:
            return html.escape(text)
        positive_lexicon = get_positive_words()
        negative_lexicon = get_negative_words()

        # Create a list of (start, end, replacement) tuples for replacements
        replacements = []
        for token in doc:
            if token.is_punct or token.is_space:
                continue

            lemma = token.lemma_.lower()
            # Only highlight if lemma is a whole word in lexicon (avoid partial matches)
            # Fix: Also check if token.text.lower() matches lemma to avoid missing highlights
            if lemma in positive_lexicon and token.text.lower() == lemma:
                color = "lightgreen"
                replacement = (
                    f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px;">'
                    f'{html.escape(token.text)}</span>'
                )
                replacements.append((token.idx, token.idx + len(token.text), replacement))
            elif lemma in negative_lexicon and token.text.lower() == lemma:
                color = "lightcoral"
                replacement = (
                    f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px;">'
                    f'{html.escape(token.text)}</span>'
                )
                replacements.append((token.idx, token.idx + len(token.text), replacement))

        return _apply_highlight_spans(text, replacements)
    except Exception as e:
        logging.warning(f"Failed to highlight keywords: {e}")
        return html.escape(text)

# Simple cache for analysis results to prevent repeated processing
_analysis_cache = {}

def clear_analysis_cache():
    """Clear the analysis cache to free memory."""
    global _analysis_cache
    _analysis_cache.clear()

def analyze_review_detailed(
    review_text: str,
    overall_sentiment: str = None,
    overall_confidence: float = 0.0,
    mysql=None,
    review_id: int | None = None
):
    """Perform comprehensive analysis of a review with caching in DB if mysql connection provided."""
    if mysql is None:
        # Fallback to existing in-memory cache if no mysql connection provided
        cache_key = review_id or hash(review_text.strip().lower())
        if cache_key in _analysis_cache:
            cached_result = deepcopy(_analysis_cache[cache_key])
            cached_result['original_text'] = review_text
            cached_result['clean_text'] = preprocess_text(review_text)
            cached_result['highlighted_text'] = highlight_aspects(
                review_text,
                cached_result.get('aspect_sentiments') or {}
            )
            if overall_sentiment:
                cached_result['summary']['overall_sentiment'] = overall_sentiment
                cached_result['summary']['overall_confidence'] = overall_confidence
            return cached_result

        clean_text = preprocess_text(review_text)
        aspects = extract_aspects(review_text)
        aspect_sentiments = analyze_aspect_sentiment(review_text, aspects, max_aspects=10)
        summary = generate_analysis_summary(aspect_sentiments, overall_sentiment, overall_confidence)
        highlighted_text = highlight_aspects(review_text, aspect_sentiments)
        advanced_insights = build_advanced_insights(review_text, aspect_sentiments, overall_sentiment or 'Neutral', overall_confidence)

        result = {
            'original_text': review_text,
            'clean_text': clean_text,
            'highlighted_text': highlighted_text,
            'aspects': aspects,
            'aspect_sentiments': aspect_sentiments,
            'summary': summary,
            'advanced_insights': advanced_insights
        }

        if len(_analysis_cache) < 100:
            _analysis_cache[cache_key] = deepcopy(result)

        return result

    # Use persistent cache in DB
    cursor = mysql.connection.cursor(cursor_factory=RealDictCursor)
    try:
        # Check if cached result exists
        if review_id is None:
            cursor.execute("SELECT review_id FROM reviews WHERE review_text = %s ORDER BY review_id DESC LIMIT 1", (review_text,))
            review_row = cursor.fetchone()
            review_id = review_row["review_id"] if review_row else None

        if review_id is None:
            raise LookupError("Review id is required for persistent cache lookup")

        cursor.execute(
            """
            SELECT aspect_sentiments, analysis_payload, language_code, language_confidence,
                   intent_label, urgency_level, experience_score, impact_score
            FROM review_aspect_sentiments
            WHERE review_id = %s
            """,
            (review_id,)
        )
        row = cursor.fetchone()
        if row:
            payload = row.get('analysis_payload')
            if payload:
                result = json.loads(payload) if isinstance(payload, str) else payload
                result['original_text'] = review_text
                result['clean_text'] = preprocess_text(review_text)
                result['highlighted_text'] = highlight_aspects(
                    review_text,
                    result.get('aspect_sentiments') or {}
                )
                if overall_sentiment:
                    result['summary']['overall_sentiment'] = overall_sentiment
                    result['summary']['overall_confidence'] = overall_confidence
                return result

            raw_aspects = row['aspect_sentiments']
            aspect_sentiments = json.loads(raw_aspects) if isinstance(raw_aspects, str) else raw_aspects
            summary = generate_analysis_summary(aspect_sentiments, overall_sentiment, overall_confidence)
            highlighted_text = highlight_aspects(review_text, aspect_sentiments)
            advanced_insights = build_advanced_insights(review_text, aspect_sentiments, overall_sentiment or 'Neutral', overall_confidence)
            return {
                'original_text': review_text,
                'clean_text': preprocess_text(review_text),
                'highlighted_text': highlighted_text,
                'aspects': list(aspect_sentiments.keys()),
                'aspect_sentiments': aspect_sentiments,
                'summary': summary,
                'advanced_insights': advanced_insights
            }
    except Exception as e:
        logging.error(f"Error fetching cached aspect sentiments: {e}")

    # If no cache, perform analysis
    clean_text = preprocess_text(review_text)
    aspects = extract_aspects(review_text)
    aspect_sentiments = analyze_aspect_sentiment(review_text, aspects, max_aspects=10)
    summary = generate_analysis_summary(aspect_sentiments, overall_sentiment, overall_confidence)
    highlighted_text = highlight_aspects(review_text, aspect_sentiments)
    advanced_insights = build_advanced_insights(review_text, aspect_sentiments, overall_sentiment or 'Neutral', overall_confidence)

    result = {
        'original_text': review_text,
        'clean_text': clean_text,
        'highlighted_text': highlighted_text,
        'aspects': aspects,
        'aspect_sentiments': aspect_sentiments,
        'summary': summary,
        'advanced_insights': advanced_insights
    }

    # Save to DB cache
    try:
        if review_id is not None:
            aspect_sentiments_json = json.dumps(aspect_sentiments)
            analysis_payload_json = json.dumps(result)
            language = advanced_insights.get('language', {})
            intent = advanced_insights.get('intent', {})
            urgency = advanced_insights.get('urgency', {})
            cursor.execute("""
                INSERT INTO review_aspect_sentiments (
                    review_id, aspect_sentiments, analysis_payload, language_code,
                    language_confidence, intent_label, urgency_level, experience_score, impact_score
                )
                VALUES (%s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (review_id)
                DO UPDATE SET
                    aspect_sentiments = EXCLUDED.aspect_sentiments,
                    analysis_payload = EXCLUDED.analysis_payload,
                    language_code = EXCLUDED.language_code,
                    language_confidence = EXCLUDED.language_confidence,
                    intent_label = EXCLUDED.intent_label,
                    urgency_level = EXCLUDED.urgency_level,
                    experience_score = EXCLUDED.experience_score,
                    impact_score = EXCLUDED.impact_score,
                    cached_at = CURRENT_TIMESTAMP
            """, (
                review_id,
                aspect_sentiments_json,
                analysis_payload_json,
                language.get('language'),
                language.get('confidence'),
                intent.get('label'),
                urgency.get('level'),
                advanced_insights.get('experience_score'),
                advanced_insights.get('impact_score')
            ))
            mysql.connection.commit()
    except Exception as e:
        logging.error(f"Error saving aspect sentiments cache: {e}")
    finally:
        cursor.close()

    return result
