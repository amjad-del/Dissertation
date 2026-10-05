import json
import re

import torch

from src.rag_engine import tokenizer, model
from src.retriever import retrieve_relevant_chunks


# ==================================================
# QUESTIONS USED FOR TARGETED RETRIEVAL
# ==================================================

ANALYSIS_QUESTIONS = {
    "objective": (
        "What is the main research objective, aim, "
        "purpose, or problem addressed by this paper?"
    ),

    "methodology": (
        "What methodology, methods, models, algorithms, "
        "framework, experimental procedure, and training "
        "process were used in this paper?"
    ),

    "dataset": (
        "What dataset or data was used in this paper? "
        "Include data source, sample size, classes, "
        "data collection, and preprocessing."
    ),

    "results": (
        "What are the main experimental results, findings, "
        "performance measurements, evaluation metrics, "
        "accuracy values, and statistical results?"
    ),

    "limitations": (
        "What limitations, weaknesses, constraints, "
        "shortcomings, challenges, or unresolved problems "
        "are explicitly mentioned by the authors?"
    ),

    "future_work": (
        "What future work, future research directions, "
        "recommendations, improvements, or further studies "
        "are explicitly mentioned by the authors?"
    )
}


# ==================================================
# RETRIEVE EVIDENCE FOR EACH FIELD
# ==================================================

def retrieve_analysis_evidence(
    chunks,
    faiss_index,
    top_k=3
):

    evidence = {}

    for field_name, question in ANALYSIS_QUESTIONS.items():

        retrieved_chunks = retrieve_relevant_chunks(
            question,
            faiss_index,
            chunks,
            top_k=top_k
        )

        evidence[field_name] = retrieved_chunks

    return evidence


# ==================================================
# REMOVE DUPLICATE CHUNKS
# ==================================================

def get_unique_chunks(evidence):

    unique_chunks = {}

    for field_chunks in evidence.values():

        for chunk in field_chunks:

            chunk_id = chunk.get("chunk_id")

            if chunk_id not in unique_chunks:
                unique_chunks[chunk_id] = chunk

    return list(unique_chunks.values())


# ==================================================
# BUILD COMBINED CONTEXT
# ==================================================

def build_analysis_context(evidence):

    context_parts = []

    for field_name, chunks in evidence.items():

        context_parts.append(
            f"\n===== EVIDENCE FOR {field_name.upper()} =====\n"
        )

        if not chunks:

            context_parts.append(
                "No relevant evidence retrieved.\n"
            )

            continue

        for i, chunk in enumerate(
            chunks,
            start=1
        ):

            context_parts.append(
                f"""
SOURCE {i}

Paper: {chunk.get('paper_name', 'Unknown')}
Page: {chunk.get('page_number', 'Unknown')}
Section: {chunk.get('section', 'Unknown')}
Chunk ID: {chunk.get('chunk_id', 'Unknown')}

Evidence:
{chunk.get('text', '')}
"""
            )

    return "\n".join(context_parts)


# ==================================================
# CLEAN MODEL JSON RESPONSE
# ==================================================

def clean_json_response(text):

    if not text:
        return ""

    text = text.strip()

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^```\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    return text.strip()


# ==================================================
# EXTRACT JSON OBJECT
# ==================================================

def extract_json_object(text):

    text = clean_json_response(text)

    start = text.find("{")
    end = text.rfind("}")

    if (
        start != -1
        and end != -1
        and end > start
    ):
        return text[start:end + 1]

    return text


# ==================================================
# DEFAULT FIELD
# ==================================================

def default_field():

    return {
        "answer": (
            "Not clearly stated in the provided evidence."
        ),
        "evidence_pages": []
    }


# ==================================================
# NORMALIZE MODEL OUTPUT
# ==================================================

def normalize_field(result, field_name):

    field_data = result.get(field_name)

    if not isinstance(field_data, dict):
        return default_field()

    answer = field_data.get(
        "answer",
        "Not clearly stated in the provided evidence."
    )

    pages = field_data.get(
        "evidence_pages",
        []
    )

    if not isinstance(answer, str):
        answer = str(answer)

    if not isinstance(pages, list):
        pages = []

    cleaned_pages = []

    for page in pages:

        try:
            page_number = int(page)

            if page_number not in cleaned_pages:
                cleaned_pages.append(page_number)

        except (ValueError, TypeError):
            continue

    return {
        "answer": answer.strip(),
        "evidence_pages": cleaned_pages
    }


# ==================================================
# GENERATE ONE STRUCTURED ANALYSIS
# ==================================================

def generate_combined_analysis(evidence):

    context = build_analysis_context(
        evidence
    )

    prompt = f"""
You are an academic research assistant.

Analyze the supplied research-paper evidence and
extract SIX specific categories:

1. Research objective
2. Methodology
3. Dataset
4. Results
5. Limitations
6. Future work

You MUST use ONLY the supplied research-paper evidence.

IMPORTANT RULES:

1. Do not use outside knowledge.

2. Do not invent or assume information.

3. Do not confuse literature-review information
   about other studies with information about the
   current paper.

4. For methodology, describe only methods actually
   used by the current paper.

5. For dataset, report only the dataset actually
   used by the current paper.

6. For results, report only results produced by the
   current paper.

7. A limitation must be explicitly supported by the
   supplied evidence. Do not invent a limitation just
   because you think the study has a weakness.

8. Future work must be explicitly supported by the
   supplied evidence. Do not invent future research
   directions.

9. If a field cannot be determined from the supplied
   evidence, use exactly:

   "Not clearly stated in the provided evidence."

10. Include page references inside each answer using
    the format:

    (Page X)

11. Also return the page numbers separately in
    evidence_pages.

12. Keep the answers concise but sufficiently detailed.

13. Return ONLY valid JSON.

Return exactly this JSON structure:

{{
    "objective": {{
        "answer": "...",
        "evidence_pages": []
    }},
    "methodology": {{
        "answer": "...",
        "evidence_pages": []
    }},
    "dataset": {{
        "answer": "...",
        "evidence_pages": []
    }},
    "results": {{
        "answer": "...",
        "evidence_pages": []
    }},
    "limitations": {{
        "answer": "...",
        "evidence_pages": []
    }},
    "future_work": {{
        "answer": "...",
        "evidence_pages": []
    }}
}}

RESEARCH PAPER EVIDENCE:

{context}

RETURN ONLY VALID JSON.
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=900,
            do_sample=False
        )

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    response = extract_json_object(
        response
    )

    try:

        result = json.loads(
            response
        )

    except json.JSONDecodeError:

        return {
            "status": "json_error",
            "raw_response": response
        }

    return {
        "status": "success",
        "result": result
    }


# ==================================================
# MAIN FUNCTION CALLED FROM APP.PY
# ==================================================

def generate_structured_analysis(
    chunks,
    faiss_index
):

    if not chunks:

        return {
            "status": "No chunks available"
        }

    if faiss_index is None:

        return {
            "status": "FAISS index unavailable"
        }


    # ----------------------------------------------
    # STEP 1:
    # Retrieve targeted evidence for all six fields
    # ----------------------------------------------

    evidence = retrieve_analysis_evidence(
        chunks,
        faiss_index,
        top_k=3
    )


    # ----------------------------------------------
    # STEP 2:
    # One Qwen call for all six fields
    # ----------------------------------------------

    generated = generate_combined_analysis(
        evidence
    )


    if generated.get("status") != "success":

        return {
            "status": generated.get(
                "status",
                "generation_error"
            ),
            "raw_response": generated.get(
                "raw_response",
                ""
            )
        }


    result = generated["result"]


    # ----------------------------------------------
    # STEP 3:
    # Normalize every field
    # ----------------------------------------------

    objective = normalize_field(
        result,
        "objective"
    )

    methodology = normalize_field(
        result,
        "methodology"
    )

    dataset = normalize_field(
        result,
        "dataset"
    )

    results = normalize_field(
        result,
        "results"
    )

    limitations = normalize_field(
        result,
        "limitations"
    )

    future_work = normalize_field(
        result,
        "future_work"
    )


    # ----------------------------------------------
    # STEP 4:
    # Return answers + evidence
    # ----------------------------------------------

    analysis = {

        "status": "success",

        # Objective
        "objective": objective["answer"],
        "objective_pages": (
            objective["evidence_pages"]
        ),
        "objective_evidence": evidence.get(
            "objective",
            []
        ),

        # Methodology
        "methodology": methodology["answer"],
        "methodology_pages": (
            methodology["evidence_pages"]
        ),
        "methodology_evidence": evidence.get(
            "methodology",
            []
        ),

        # Dataset
        "dataset": dataset["answer"],
        "dataset_pages": (
            dataset["evidence_pages"]
        ),
        "dataset_evidence": evidence.get(
            "dataset",
            []
        ),

        # Results
        "results": results["answer"],
        "results_pages": (
            results["evidence_pages"]
        ),
        "results_evidence": evidence.get(
            "results",
            []
        ),

        # Limitations
        "limitations": limitations["answer"],
        "limitations_pages": (
            limitations["evidence_pages"]
        ),
        "limitations_evidence": evidence.get(
            "limitations",
            []
        ),

        # Future work
        "future_work": future_work["answer"],
        "future_work_pages": (
            future_work["evidence_pages"]
        ),
        "future_work_evidence": evidence.get(
            "future_work",
            []
        )
    }

    return analysis