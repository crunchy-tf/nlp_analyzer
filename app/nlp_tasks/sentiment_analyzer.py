# nlp_analyzer_service/app/nlp_tasks/sentiment_analyzer.py
from transformers import pipeline
from loguru import logger
from typing import List, Dict, Optional, Any

from app.config import settings # CORRECTED IMPORT

class SentimentAnalyzer:
    def __init__(self, model_name: str = settings.SENTIMENT_MODEL_NAME, labels: List[str] = settings.HEALTHCARE_SENTIMENT_LABELS): # CORRECTED DEFAULTS
        self.model_name = model_name
        self.labels = labels
        self.classifier: Optional[pipeline] = None
        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"Loading zero-shot sentiment model: {self.model_name}")
            self.classifier = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                use_fast=False 
            )
            logger.info("Zero-shot sentiment model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load zero-shot sentiment model: {e}", exc_info=True)
            self.classifier = None

    def analyze(self, text: str, custom_labels: Optional[List[str]] = None) -> Optional[List[Dict[str, Any]]]:
        if not self.classifier:
            logger.error("Sentiment classifier not loaded. Cannot analyze.")
            return None
        if not text or not text.strip():
            logger.warning("Input text for sentiment analysis is empty.")
            return [] 
        candidate_labels = custom_labels if custom_labels else self.labels
        try:
            result = self.classifier(text, candidate_labels=candidate_labels)
            formatted_results = []
            if result and 'labels' in result and 'scores' in result:
                for label, score in zip(result['labels'], result['scores']):
                    formatted_results.append({"label": label, "score": score})
                return formatted_results
            else:
                logger.warning(f"Unexpected sentiment analysis result format for text: '{text[:100]}...'")
                return None
        except Exception as e:
            logger.error(f"Error during sentiment analysis for text '{text[:100]}...': {e}", exc_info=True)
            return None

sentiment_pipeline = SentimentAnalyzer()
def get_sentiment_analyzer_instance():
    return sentiment_pipeline