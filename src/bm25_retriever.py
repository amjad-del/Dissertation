from rank_bm25 import BM25Okapi
import numpy as np


def build_bm25_index(chunks):
    """
    Build a BM25 keyword-search index from paper chunks.
    """

    if not chunks:
        return None

    tokenized_documents = [
        chunk["text"].lower().split()
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_documents)


def retrieve_bm25(
    question,
    chunks,
    bm25_index,
    top_k=8
):
    """
    Retrieve the most relevant chunks using BM25 keyword matching.
    """

    if not question.strip() or not chunks or bm25_index is None:
        return []

    query_tokens = question.lower().split()

    scores = bm25_index.get_scores(query_tokens)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:

        chunk = chunks[index].copy()

        chunk["bm25_score"] = float(scores[index])

        results.append(chunk)

    return results