# Topic Modeler Logic
import joblib
from bertopic import BERTopic
from loguru import logger
from typing import List, Tuple, Optional, Dict, Any
from app.config import BERTOPIC_MODEL_PATH

class TopicModeler:
    def __init__(self, model_path: str = BERTOPIC_MODEL_PATH):
        self.model_path = model_path
        self.topic_model: Optional[BERTopic] = None
        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"Loading BERTopic model from: {self.model_path}")
            self.topic_model = BERTopic.load(self.model_path) 
            logger.info("BERTopic model loaded successfully.")
        except FileNotFoundError:
            logger.error(f"BERTopic model file not found at {self.model_path}. Ensure it's placed correctly.")
            self.topic_model = None
        except Exception as e:
            logger.error(f"Failed to load BERTopic model: {e}", exc_info=True)
            self.topic_model = None

    def get_topics_for_doc(self, document_text: str) -> Optional[Tuple[List[int], Optional[List[float]]]]:
        if not self.topic_model:
            logger.error("BERTopic model not loaded. Cannot get topics.")
            return None
        if not document_text or not document_text.strip():
            logger.warning("Input document for topic modeling is empty.")
            return ([-1], [1.0])

        try:
            topics, probabilities = self.topic_model.transform([document_text])
            return topics, probabilities
        except Exception as e:
            logger.error(f"Error during BERTopic transform for doc '{document_text[:100]}...': {e}", exc_info=True)
            return None

    def get_topic_details(self, topic_id: int) -> Optional[List[Tuple[str, float]]]:
        if not self.topic_model:
            logger.error("BERTopic model not loaded. Cannot get topic details.")
            return None
        try:
            return self.topic_model.get_topic(topic_id)
        except Exception as e:
            logger.error(f"Error getting details for topic ID {topic_id}: {e}", exc_info=True)
            return None
            
    def get_topic_name(self, topic_id: int) -> str:
        if not self.topic_model:
            return f"Topic {topic_id}"
        try:
            topic_info_df = self.topic_model.get_topic_info(topic_id)
            if not topic_info_df.empty and 'Name' in topic_info_df.columns:
                return topic_info_df['Name'].iloc[0]
            else: 
                details = self.get_topic_details(topic_id)
                if details:
                    return "_".join([word for word, score in details[:3]])
                return f"Topic {topic_id}"

        except Exception as e:
            logger.warning(f"Could not get custom name for topic {topic_id}: {e}")
            return f"Topic {topic_id}"

topic_model_instance = TopicModeler()

def get_topic_modeler_instance():
    return topic_model_instance