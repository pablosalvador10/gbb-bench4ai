import tiktoken
from typing import Optional, Tuple
from utils.ml_logging import get_logger
from fuzzywuzzy import process

logger = get_logger()

MODEL_PREFIX_TO_ENCODING = {
    "gpt-4-": "cl100k_base",
    "gpt-3.5-turbo-": "cl100k_base",
    "gpt-35-turbo-": "cl100k_base",
    "ft:gpt-4": "cl100k_base",
    "ft:gpt-3.5-turbo": "cl100k_base",
    "ft:davinci-002": "cl100k_base",
    "ft:babbage-002": "cl100k_base",
}

MODEL_TO_ENCODING = {
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-35-turbo": "cl100k_base",
    "davinci-002": "cl100k_base",
    "babbage-002": "cl100k_base",
    "text-embedding-ada-002": "cl100k_base",
    "text-davinci-003": "p50k_base",
    "text-davinci-002": "p50k_base",
    "text-davinci-001": "r50k_base",
    "text-curie-001": "r50k_base",
    "text-babbage-001": "r50k_base",
    "text-ada-001": "r50k_base",
    "davinci": "r50k_base",
    "curie": "r50k_base",
    "babbage": "r50k_base",
    "ada": "r50k_base",
    "code-davinci-002": "p50k_base",
    "code-davinci-001": "p50k_base",
    "code-cushman-002": "p50k_base",
    "code-cushman-001": "p50k_base",
    "davinci-codex": "p50k_base",
    "cushman-codex": "p50k_base",
    "text-davinci-edit-001": "p50k_edit",
    "code-davinci-edit-001": "p50k_edit",
    "text-similarity-davinci-001": "r50k_base",
    "text-similarity-curie-001": "r50k_base",
    "text-similarity-babbage-001": "r50k_base",
    "text-similarity-ada-001": "r50k_base",
    "text-search-davinci-doc-001": "r50k_base",
    "text-search-curie-doc-001": "r50k_base",
    "text-search-babbage-doc-001": "r50k_base",
    "text-search-ada-doc-001": "r50k_base",
    "code-search-babbage-code-001": "r50k_base",
    "code-search-ada-code-001": "r50k_base",
    "gpt2": "gpt2",
}

from fuzzywuzzy import process  # Ensure you have fuzzywuzzy installed

def detect_model_encoding(model_name: str) -> Tuple[str, str]:
    """
    Detects the encoding for a given model name based on predefined mappings, and returns the best matching model name.
    
    Args:
    - model_name (str): The name of the model.
    
    Returns:
    - tuple: A tuple containing the best matching model name and the encoding type for the model. Defaults to ('gpt-4', 'cl100k_base') if no match is found.
    """
    # Check for exact matches first
    if model_name in MODEL_TO_ENCODING:
        return model_name, MODEL_TO_ENCODING[model_name]

    # If no exact match, check for prefix matches
    for prefix, encoding in MODEL_PREFIX_TO_ENCODING.items():
        if model_name.startswith(prefix):
            return prefix, encoding

    # Use fuzzy matching for close matches
    all_model_names = list(MODEL_TO_ENCODING.keys()) + list(MODEL_PREFIX_TO_ENCODING.keys())
    best_match, score = process.extractOne(model_name, all_model_names)
    if score >= 60:  # Threshold for a good match
        logger.info(f"Fuzzy match found: '{model_name}' matched to '{best_match}' with a score of {score}.")
        if best_match in MODEL_TO_ENCODING:
            return best_match, MODEL_TO_ENCODING[best_match]
        for prefix, encoding in MODEL_PREFIX_TO_ENCODING.items():
            if best_match.startswith(prefix):
                return best_match, encoding

    # Default encoding if no match is found
    logger.warning(f"No good match found for model name '{model_name}'. Defaulting to 'gpt-4'.")
    return "gpt-4", "cl100k_base"

def get_encoding_from_model_name(model_name: str):
    """
    Returns the encoding object for the specified model's encoding.
    
    Args:
    - model_name (str): The name of the model to use for encoding.
    
    Returns:
    - Object: The encoding object for the model.
    
    Raises:
    - ValueError: If the model name is invalid or unspecified.
    """
    if model_name is None:
        logger.warning("Model name is None. Defaulting to 'gpt-4'.")
        model_name = "gpt-4"
    else:
        model_name, _ = detect_model_encoding(model_name)
    encoding = tiktoken.encoding_for_model(model_name)
    return encoding

def count_tokens(encoding, string: str) -> int:
    """
    Returns the number of tokens in a text string using the provided encoding object.
    
    Args:
    - encoding (Object): The encoding object to use for tokenization.
    - string (str): The text string to tokenize.
    
    Returns:
    - int: The number of tokens in the string.
    """
    try:
        num_tokens = len(encoding.encode(string))
        return num_tokens
    except Exception as e:
        logger.exception("Failed to calculate the number of tokens due to an error: %s", e)
        raise

def terminal_http_code(e):
    """
    Determines if an HTTP status code indicates a client error.
    
    Args:
    - e: An exception object that contains an HTTP status code.
    
    Returns:
    - bool: True if the status code is in the range 400-499, indicating a client error. False otherwise.
    """
    return 400 <= e.status < 500
