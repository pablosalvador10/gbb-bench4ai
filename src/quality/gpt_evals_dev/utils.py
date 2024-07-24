import os 
from my_utils.ml_logging import get_logger
logger = get_logger()

def remove_file(path):
    try:
        if os.name == 'nt':
            # Convert to Windows path format if on Windows
            path = path.replace("/", "\\")
        # Attempt to remove the file
        os.remove(path)
        logger.info(f"Successfully removed file: {path}")
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
    except PermissionError:
        logger.error(f"Permission denied: {path}")
    except Exception as e:
        logger.error(f"Error removing file: {path}. Error: {e}")

simplified_names_mapping = {
    "RelevanceEvaluator": "relevance",
    "F1ScoreEvaluator": "f1_score",
    "GroundednessEvaluator": "groundedness",
    "ViolenceEvaluator": "violence",
    "SexualEvaluator": "sexual_content",
    "SelfHarmEvaluator": "self_harm",
    "HateUnfairnessEvaluator": "hate_unfairness",
    "CoherenceEvaluator": "coherence",
    "FluencyEvaluator": "fluency",
    "SimilarityEvaluator": "similarity",
    "QAEvaluator": "qa",
    "ChatEvaluator": "chat",
    "ContentSafetyEvaluator": "content_safety",
    "ContentSafetyChatEvaluator": "content_safety_chat"
}