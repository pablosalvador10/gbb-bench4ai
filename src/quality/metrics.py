from typing import Any

from sentence_transformers import SentenceTransformer


def accuracy(y_pred: Any, y_true: Any) -> int:
    if y_pred == y_true:
        return 1
    else:
        return 0


def sentence_transformer_similarity(corpusA: str, corpusB: str) -> float:
    model = SentenceTransformer(
        "all-mpnet-base-v2"
    )  # Docs: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
    # get embeddings for each sentence
    embeddingsA = model.encode(corpusA.split("."))
    embeddingsB = model.encode(corpusB.split("."))
    # Get similarity matrix all sentance comparisons
    sim_tensor = model.similarity(embeddingsA, embeddingsB)
    # Get the maximum similarity for each sentence across the comparison
    max_sims = sim_tensor.max(axis=1).values

    # Return average of max sims
    return max_sims.mean()
