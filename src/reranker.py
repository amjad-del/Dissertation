from functools import lru_cache


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


def rerank_chunks(question, chunks, top_k=3):

    if not chunks:
        return []

    model = get_reranker()

    pairs = [
        [question, chunk["text"]]
        for chunk in chunks
    ]

    scores = model.predict(pairs)

    reranked = []

    for chunk, score in zip(chunks, scores):

        result = chunk.copy()

        result["reranker_score"] = float(score)

        reranked.append(result)

    reranked.sort(
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    return reranked[:top_k]