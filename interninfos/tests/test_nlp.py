import pytest
import sys
import os
import warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Suppress deprecation warnings from dependencies
warnings.filterwarnings("ignore", message=".*parser.split_arg_string.*", category=DeprecationWarning)

from app.nlp_utils import (
    preprocess_text, extract_aspects, analyze_aspect_sentiment,
    generate_analysis_summary, highlight_aspects, highlight_keywords,
    map_sentiment, ensemble_sentiment, analyze_review_detailed,
    get_sentiment_analyzer, get_bert_analyzer, get_irony_analyzer,
    clear_analysis_cache
)

class TestNLPUtils:
    """Test suite for NLP utilities."""

    @pytest.fixture(scope="class")
    def sample_texts(self):
        """Sample texts for testing."""
        return {
            'positive': "This product is amazing! Great quality and fast delivery.",
            'negative': "Terrible experience. Poor quality and slow service.",
            'neutral': "The product arrived on time. It's okay.",
            'sarcastic': "Oh wow, another fantastic product that works perfectly. So impressed.",
            'empty': "",
            'long': "This is a very long review text that goes on and on with many words to test the preprocessing and analysis capabilities of our sentiment analysis system. It includes multiple sentences and various aspects like quality, performance, design, and customer service. The product has excellent battery life, amazing camera quality, and superb performance. However, the design could be better and customer service was lacking. Overall, it's a mixed bag with some great features and some disappointments."[:1000]  # Truncate for testing
        }

    def test_preprocess_text(self, sample_texts):
        """Test text preprocessing."""
        # Positive text
        processed = preprocess_text(sample_texts['positive'])
        assert isinstance(processed, str)
        assert len(processed) > 0
        assert not any(char.isdigit() for char in processed)  # No digits

        # Empty text
        assert preprocess_text(sample_texts['empty']) == ""

        # Long text
        processed_long = preprocess_text(sample_texts['long'])
        assert len(processed_long) < len(sample_texts['long'])  # Should be shorter after cleaning

    def test_extract_aspects(self, sample_texts):
        """Test aspect extraction."""
        aspects = extract_aspects(sample_texts['positive'])
        assert isinstance(aspects, list)
        assert len(aspects) <= 10  # Limited to 10

        # Should extract some aspects from long text
        aspects_long = extract_aspects(sample_texts['long'])
        assert len(aspects_long) > 0
        assert all(isinstance(aspect, str) for aspect in aspects_long)

    def test_map_sentiment(self):
        """Test sentiment mapping."""
        # Test various labels
        assert map_sentiment("LABEL_2") == "Positive"
        assert map_sentiment("positive") == "Positive"
        assert map_sentiment("LABEL_0") == "Negative"
        assert map_sentiment("negative") == "Negative"
        assert map_sentiment("LABEL_1") == "Neutral"
        assert map_sentiment("neutral") == "Neutral"

        # Test irony adjustment
        assert map_sentiment("LABEL_2", "LABEL_1", 0.6) == "Negative"  # Positive becomes negative
        assert map_sentiment("LABEL_0", "LABEL_1", 0.6) == "Positive"  # Negative becomes positive

    def test_ensemble_sentiment(self):
        """Test ensemble sentiment combination."""
        sent_roberta = {'label': 'LABEL_2', 'score': 0.8}
        sent_bert = {'label': 'LABEL_2', 'score': 0.7}

        sentiment, confidence = ensemble_sentiment(sent_roberta, sent_bert)
        assert sentiment in ["Positive", "Negative", "Neutral"]
        assert 0.0 <= confidence <= 1.0

        # Test with irony
        sentiment_irony, _ = ensemble_sentiment(sent_roberta, sent_bert, "LABEL_1", 0.6)
        assert sentiment_irony == "Negative"  # Should flip

    def test_analyze_aspect_sentiment(self, sample_texts):
        """Test aspect sentiment analysis."""
        aspects = extract_aspects(sample_texts['positive'])
        if aspects:
            sentiments = analyze_aspect_sentiment(sample_texts['positive'], aspects, max_aspects=3)
            assert isinstance(sentiments, dict)
            for aspect, info in sentiments.items():
                assert 'sentiment' in info
                assert 'confidence' in info
                assert 'irony_score' in info
                assert 'sentences' in info
                assert info['sentiment'] in ["Positive", "Negative", "Neutral"]

    def test_generate_analysis_summary(self):
        """Test analysis summary generation."""
        aspect_sentiments = {
            'quality': {'sentiment': 'Positive', 'confidence': 0.8, 'irony_score': 0.1, 'sentences': []},
            'service': {'sentiment': 'Negative', 'confidence': 0.7, 'irony_score': 0.2, 'sentences': []},
            'price': {'sentiment': 'Neutral', 'confidence': 0.5, 'irony_score': 0.0, 'sentences': []}
        }

        summary = generate_analysis_summary(aspect_sentiments, "Positive", 0.75)
        assert summary['total_aspects'] == 3
        assert summary['positive_count'] == 1
        assert summary['negative_count'] == 1
        assert summary['neutral_count'] == 1
        assert summary['overall_sentiment'] == "Positive"
        assert summary['overall_confidence'] == 0.75

    def test_highlight_aspects(self, sample_texts):
        """Test aspect highlighting."""
        aspect_sentiments = {
            'quality': {'sentiment': 'Positive', 'confidence': 0.8, 'irony_score': 0.1, 'sentences': []}
        }

        highlighted = highlight_aspects(sample_texts['positive'], aspect_sentiments)
        assert isinstance(highlighted, str)
        assert 'background-color' in highlighted  # Should contain highlighting

    def test_highlight_keywords(self, sample_texts):
        """Test keyword highlighting."""
        highlighted = highlight_keywords(sample_texts['positive'], "Positive")
        assert isinstance(highlighted, str)
        # Should highlight positive words if present

    def test_highlighting_escapes_html(self):
        """Highlighted output should not render raw user HTML."""
        text = '<script>alert("x")</script> quality is terrible'
        aspect_sentiments = {
            'quality': {'sentiment': 'Negative', 'confidence': 0.9, 'irony_score': 0.0, 'sentences': []}
        }

        highlighted_aspects = highlight_aspects(text, aspect_sentiments)
        highlighted_keywords = highlight_keywords(text, "Negative")

        assert '<script>' not in highlighted_aspects
        assert '&lt;script&gt;' in highlighted_aspects
        assert '<script>' not in highlighted_keywords
        assert '&lt;script&gt;' in highlighted_keywords

    def test_analyze_review_detailed(self, sample_texts):
        """Test detailed review analysis."""
        clear_analysis_cache()  # Start fresh

        result = analyze_review_detailed(sample_texts['positive'], "Positive", 0.8)
        assert 'original_text' in result
        assert 'clean_text' in result
        assert 'highlighted_text' in result
        assert 'aspects' in result
        assert 'aspect_sentiments' in result
        assert 'summary' in result

        # Test caching
        result2 = analyze_review_detailed(sample_texts['positive'], "Positive", 0.8)
        assert result['summary']['overall_sentiment'] == result2['summary']['overall_sentiment']

        # Test empty text
        result_empty = analyze_review_detailed(sample_texts['empty'])
        assert result_empty['aspects'] == []

    def test_model_loading_is_safe(self):
        """Model getters may return None, but they should not raise."""
        try:
            analyzer = get_sentiment_analyzer()
            bert_analyzer = get_bert_analyzer()
            irony_analyzer = get_irony_analyzer()
        except Exception as e:
            pytest.fail(f"Model getter raised unexpectedly: {e}")

        assert analyzer is None or callable(analyzer)
        assert bert_analyzer is None or callable(bert_analyzer)
        assert irony_analyzer is None or callable(irony_analyzer)

    def test_error_handling(self, sample_texts):
        """Test error handling and fallbacks."""
        # Test with invalid text
        result = analyze_review_detailed("   ", "Neutral", 0.5)
        assert result['aspects'] == []

        # Test aspect analysis with empty aspects
        sentiments = analyze_aspect_sentiment(sample_texts['positive'], [], max_aspects=5)
        assert sentiments == {}

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
