import faiss
import numpy as np


def create_faiss_index(embeddings):
    """
    Create a FAISS index using cosine similarity.
    """

    if embeddings is None or len(embeddings) == 0:
        return None

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    # Inner product on normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index