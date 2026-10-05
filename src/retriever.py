import faiss
import numpy as np

from src.embeddings import get_embedding_model
from src.reranker import rerank_chunks

from src.bm25_retriever import build_bm25_index, retrieve_bm25


def get_section_bonus(question, section):
    """
    Give extra score when the question matches the paper section.
    """

    question = question.lower()
    section = section.lower()

    bonus = 0.0

    if any(word in question for word in ["methodology", "method", "methods"]):
        if "method" in section:
            bonus += 0.25

    if any(word in question for word in ["dataset", "data"]):
        if "dataset" in section or section == "data":
            bonus += 0.25

    if "framework" in question:
        if "framework" in section:
            bonus += 0.25

    if "result" in question:
        if "result" in section:
            bonus += 0.25

    if "future work" in question or "future" in question:
        if "future" in section:
            bonus += 0.25

    if "limitation" in question:
        if "limitation" in section:
            bonus += 0.25

    if "conclusion" in question:
        if "conclusion" in section:
            bonus += 0.25

    return bonus

def retrieve_relevant_chunks(question, index, chunks, top_k=3):
    """
    Hybrid retrieval:
    1. FAISS semantic search
    2. BM25 keyword search
    3. Merge results
    4. Cross-encoder reranking
    """

    if not question.strip() or not chunks:
        return []

    # --------------------------------------------------
    # 1. FAISS semantic retrieval
    # --------------------------------------------------

    model = get_embedding_model()

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        show_progress_bar=False
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    faiss.normalize_L2(question_embedding)

    candidate_count = min(8, len(chunks))

    scores, indices = index.search(
        question_embedding,
        candidate_count
    )

    faiss_results = []

    for rank, chunk_index in enumerate(indices[0]):

        if chunk_index < 0:
            continue

        chunk = chunks[chunk_index].copy()

        chunk["similarity"] = float(
            scores[0][rank]
        )

        chunk["section_bonus"] = get_section_bonus(
            question,
            chunk.get("section", "")
        )

        chunk["faiss_score"] = (
            chunk["similarity"]
            + chunk["section_bonus"]
        )

        faiss_results.append(chunk)

    # --------------------------------------------------
    # 2. BM25 keyword retrieval
    # --------------------------------------------------

    bm25_index = build_bm25_index(chunks)

    bm25_results = retrieve_bm25(
        question,
        chunks,
        bm25_index,
        top_k=8
    )

    # --------------------------------------------------
    # 3. Merge FAISS + BM25 results
    # --------------------------------------------------

    merged_results = {}

    for chunk in faiss_results:

        chunk_id = chunk["chunk_id"]

        merged_results[chunk_id] = chunk

        merged_results[chunk_id]["bm25_score"] = 0.0

    for chunk in bm25_results:

        chunk_id = chunk["chunk_id"]

        if chunk_id in merged_results:

            merged_results[chunk_id]["bm25_score"] = (
                chunk["bm25_score"]
            )

        else:

            chunk["similarity"] = 0.0
            chunk["section_bonus"] = get_section_bonus(
                question,
                chunk.get("section", "")
            )

            chunk["faiss_score"] = 0.0

            merged_results[chunk_id] = chunk

    candidate_results = list(
        merged_results.values()
    )

    # --------------------------------------------------
    # 4. Sort candidates
    # --------------------------------------------------

    candidate_results.sort(
        key=lambda x: (
            x.get("faiss_score", 0.0)
            + x.get("bm25_score", 0.0)
        ),
        reverse=True
    )

    candidate_results = candidate_results[:10]

    # --------------------------------------------------
    # 5. Cross-encoder reranking
    # --------------------------------------------------

    reranked_results = rerank_chunks(
        question,
        candidate_results,
        top_k=top_k
    )

    return reranked_results
