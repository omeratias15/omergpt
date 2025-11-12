"""
src/sentiment_analysis/finbert_analyzer.py

GPU-accelerated FinBERT sentiment classifier for financial text analysis.
Uses HuggingFace transformers with PyTorch CUDA support for batch inference
on cryptocurrency-related Reddit posts and financial news.
"""

import asyncio
import logging
import time
from typing import List, Dict, Optional

import numpy as np

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None

logger = logging.getLogger("omerGPT.sentiment.finbert")

MODEL_NAME = "ProsusAI/finbert"


class FinBERTAnalyzer:
    """
    GPU-accelerated FinBERT sentiment classifier.
    
    Features:
    - Batch inference with configurable batch size
    - Automatic GPU/CPU device selection
    - Three-class sentiment: Positive, Neutral, Negative
    - Confidence scores for each prediction
    """
    
    def __init__(
        self,
        max_batch_size: int = 32,
        use_gpu: bool = True,
        max_length: int = 128,
    ):
        """
        Initialize FinBERT analyzer.
        
        Args:
            max_batch_size: Maximum batch size for inference
            use_gpu: Whether to use GPU acceleration if available
            max_length: Maximum token length for input text
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required. Install: pip install transformers"
            )
        
        self.max_batch_size = max_batch_size
        self.max_length = max_length
        
        # Device selection
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        
        # Load model and tokenizer
        logger.info(f"Loading FinBERT model: {MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME
        ).to(self.device)
        self.model.eval()
        
        # Sentiment labels
        self.labels = ["negative", "neutral", "positive"]
        
        logger.info(f"FinBERT initialized on device: {self.device}")
    
    async def analyze_posts(self, posts: List[Dict]) -> List[Dict]:
        """
        Classify sentiment for a list of posts asynchronously.
        
        Args:
            posts: List of dicts with 'title' or 'text' fields
        
        Returns:
            List of dicts with keys: text, sentiment, score, confidence
        """
        if not posts:
            logger.warning("No posts to analyze")
            return []
        
        # Extract text from posts
        texts = [
            p.get("title") or p.get("text") or "" for p in posts
        ]
        
        # Filter empty texts
        texts = [t for t in texts if t.strip()]
        
        if not texts:
            logger.warning("No valid text found in posts")
            return []
        
        # Run inference in executor to avoid blocking
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, self._analyze_batch, texts)
        
        return results
    
    def _analyze_batch(self, texts: List[str]) -> List[Dict]:
        """
        Perform batch sentiment inference on GPU/CPU.
        
        Args:
            texts: List of text strings to analyze
        
        Returns:
            List of result dictionaries
        """
        start_time = time.time()
        results = []
        
        # Process in batches
        num_batches = (len(texts) + self.max_batch_size - 1) // self.max_batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.max_batch_size
            end_idx = min((batch_idx + 1) * self.max_batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            try:
                # Tokenize batch
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                
                # Inference
                with torch.no_grad():
                    outputs = self.model(**encoded)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                
                # Convert to numpy
                probs_np = probs.cpu().numpy()
                
                # Extract results
                for text, prob in zip(batch_texts, probs_np):
                    pred_idx = int(np.argmax(prob))
                    sentiment = self.labels[pred_idx].capitalize()
                    confidence = float(prob[pred_idx])
                    
                    results.append({
                        "text": text[:100],  # Truncate for logging
                        "sentiment": sentiment,
                        "score": self._sentiment_to_score(sentiment),
                        "confidence": confidence,
                        "scores": {
                            "negative": float(prob[0]),
                            "neutral": float(prob[1]),
                            "positive": float(prob[2]),
                        }
                    })
            
            except Exception as e:
                logger.error(f"Batch {batch_idx} inference failed: {e}")
                # Add placeholder results for failed batch
                for text in batch_texts:
                    results.append({
                        "text": text[:100],
                        "sentiment": "Neutral",
                        "score": 0.0,
                        "confidence": 0.0,
                        "scores": {"negative": 0.0, "neutral": 1.0, "positive": 0.0}
                    })
        
        # Calculate statistics
        elapsed = time.time() - start_time
        sentiment_counts = {
            "Positive": sum(1 for r in results if r["sentiment"] == "Positive"),
            "Neutral": sum(1 for r in results if r["sentiment"] == "Neutral"),
            "Negative": sum(1 for r in results if r["sentiment"] == "Negative"),
        }
        
        logger.info(
            f"Analyzed {len(texts)} texts in {elapsed:.2f}s | "
            f"Sentiment distribution: {sentiment_counts}"
        )
        
        return results
    
    def _sentiment_to_score(self, sentiment: str) -> float:
        """
        Convert sentiment label to numeric score.
        
        Args:
            sentiment: Sentiment label (Positive/Neutral/Negative)
        
        Returns:
            Numeric score: Positive=1.0, Neutral=0.0, Negative=-1.0
        """
        mapping = {
            "Positive": 1.0,
            "Neutral": 0.0,
            "Negative": -1.0,
        }
        return mapping.get(sentiment, 0.0)


# Test block
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    async def test_finbert():
        """Test FinBERT analyzer with sample texts."""
        print("Testing FinBERTAnalyzer...")
        
        if not TRANSFORMERS_AVAILABLE:
            print("transformers not installed - skipping test")
            return
        
        analyzer = FinBERTAnalyzer(max_batch_size=8, use_gpu=True)
        
        # Sample crypto-related texts
        samples = [
            {"title": "Bitcoin surges 10% as institutional adoption accelerates"},
            {"title": "Ethereum faces regulatory uncertainty amid SEC crackdown"},
            {"title": "Crypto markets show resilience after major correction"},
            {"title": "FUD dominates r/CryptoMarkets, sentiment plummets"},
            {"title": "Bullish breakout expected for Solana, technical indicators align"},
            {"title": "Market remains neutral as traders await Fed decision"},
        ]
        
        print("\nAnalyzing sample texts...")
        results = await analyzer.analyze_posts(samples)
        
        print("\nResults:")
        print(f"{'Sentiment':<12} {'Conf':<6} {'Score':<6} Text")
        print("-" * 80)
        
        for res in results:
            print(
                f"{res['sentiment']:<12} "
                f"{res['confidence']:<6.2f} "
                f"{res['score']:<6.1f} "
                f"{res['text'][:50]}"
            )
        
        print("\nTest completed successfully!")
    
    asyncio.run(test_finbert())
