"""
VADER Sentiment Analyzer (Fallback)
Rule-based sentiment analysis for social media text
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VADERAnalyzer:
    """VADER-based sentiment analyzer as fallback/complement"""

    def __init__(self):
        """Initialize VADER analyzer"""
        try:
            self.analyzer = SentimentIntensityAnalyzer()
            logger.info("VADER analyzer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize VADER: {e}")
            raise

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER

        Returns:
            Dict with compound score and individual scores
        """
        if not text or len(text.strip()) == 0:
            return {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0
            }

        try:
            scores = self.analyzer.polarity_scores(text)
            return {
                'compound': scores['compound'],  # Range [-1, 1]
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu']
            }
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze multiple texts

        Returns:
            List of sentiment dictionaries
        """
        return [self.analyze_text(text) for text in texts]

    def compute_aggregate_sentiment(self, sentiments: List[Dict[str, float]], 
                                   weights: List[float] = None) -> Dict[str, float]:
        """
        Compute weighted aggregate sentiment

        Args:
            sentiments: List of VADER sentiment dicts
            weights: Optional weights for each sentiment

        Returns:
            Aggregated sentiment scores
        """
        if not sentiments:
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}

        if weights is None:
            weights = [1.0] * len(sentiments)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0] * len(sentiments)
            total_weight = len(sentiments)

        weights = [w / total_weight for w in weights]

        # Weighted average
        aggregate = {
            'compound': sum(s['compound'] * w for s, w in zip(sentiments, weights)),
            'positive': sum(s['positive'] * w for s, w in zip(sentiments, weights)),
            'negative': sum(s['negative'] * w for s, w in zip(sentiments, weights)),
            'neutral': sum(s['neutral'] * w for s, w in zip(sentiments, weights))
        }

        return aggregate

    def classify_sentiment(self, compound_score: float) -> str:
        """
        Classify compound score into sentiment category

        VADER compound score interpretation:
            >= 0.05: positive
            <= -0.05: negative
            else: neutral
        """
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'


if __name__ == "__main__":
    analyzer = VADERAnalyzer()

    # Test
    test_texts = [
        "Bitcoin is showing strong bullish momentum! 🚀",
        "This is a disaster, selling all my holdings immediately",
        "The market opened today"
    ]

    for text in test_texts:
        sentiment = analyzer.analyze_text(text)
        classification = analyzer.classify_sentiment(sentiment['compound'])
        print(f"Text: {text}")
        print(f"Compound: {sentiment['compound']:.3f} ({classification})")
        print(f"Scores: {sentiment}\n")
