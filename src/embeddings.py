from functools import lru_cache


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Load Sentence Transformer only when it is needed.
    """

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


def generate_embeddings(chunks):
    """
    Convert text chunks into numerical embeddings.
    """

    if not chunks:
        return None

    model = get_embedding_model()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    return embeddings