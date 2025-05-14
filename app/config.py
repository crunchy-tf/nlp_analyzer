# Configuration (model paths, labels)
import os
from typing import List

# --- General Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BERTOPIC_MODEL_FILENAME = "bertopic_model_final_guided_multilang_gensim.joblib"
BERTOPIC_MODEL_PATH = os.path.join(BASE_DIR, BERTOPIC_MODEL_FILENAME)

# --- Sentiment Analysis ---
SENTIMENT_MODEL_NAME = "joeddav/xlm-roberta-large-xnli"
HEALTHCARE_SENTIMENT_LABELS: List[str] = [
    "Satisfied", "Grateful", "Concerned", "Anxious", "Confused", "Angry", "Neutral"
]

# --- Keyword Extraction ---
TOP_N_KEYWORDS = 10

# --- Logging ---
LOG_LEVEL = "INFO"

# --- APScheduler ---
SCHEDULER_JOB_INTERVAL_SECONDS = 3600