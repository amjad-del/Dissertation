from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
import torch


# ==================================================
# LOCAL LLM
# ==================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    low_cpu_mem_usage=True
)


# ==================================================
# BUILD RAG CONTEXT
# ==================================================

def build_context(retrieved_chunks):

    context_parts = []

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        paper_name = chunk.get(
            "paper_name",
            "Unknown"
        )

        page_number = chunk.get(
            "page_number",
            "Unknown"
        )

        section = chunk.get(
            "section",
            "Unknown"
        )

        text = chunk.get(
            "text",
            ""
        )

        context_parts.append(
            f"""
SOURCE {i}
Paper: {paper_name}
Page: {page_number}
Section: {section}

Evidence:
{text}
"""
        )

    return "\n".join(context_parts)


# ==================================================
# GENERATE RAG ANSWER
# ==================================================

def generate_rag_answer(
    question,
    retrieved_chunks
):

    if not retrieved_chunks:

        return (
            "The requested information could not be "
            "found in the uploaded research paper."
        )


    context = build_context(
        retrieved_chunks
    )


    prompt = f"""
You are an academic research assistant working with
an uploaded research paper.

Answer the user's question using ONLY the retrieved
research-paper evidence provided below.

STRICT RULES:

1. Use only information explicitly supported by the
   retrieved evidence.

2. Do not use outside knowledge.

3. Do not invent any information.

4. Answer ONLY the exact question asked.

5. Do not summarize unrelated information from the
   retrieved passages.

6. Keep the answer concise and academic.

7. Normally answer in 2 to 5 sentences.

8. Include important numerical values when they are
   relevant to the question.

9. Cite supporting page numbers using:
   (Page X)

10. Never cite a page that is not included in the
    retrieved evidence.

11. Do not confuse information about previous studies
    in the literature review with the current study.

12. For methodology questions, describe only the
    methodology used in THIS paper.

13. For dataset questions, describe only the dataset
    used in THIS paper.

14. For results questions, report only results from
    THIS paper.

15. For objective questions, state the main research
    aim or problem. Do not add detailed results,
    ablation studies, or conclusions unless the
    question specifically asks for them.

16. For limitations and future-work questions, report
    them only when explicitly supported by the
    retrieved evidence.

17. If the retrieved evidence does not clearly answer
    the question, respond exactly:

    "The requested information is not clearly stated
    in the retrieved evidence."

USER QUESTION:

{question}

RETRIEVED RESEARCH-PAPER EVIDENCE:

{context}

ANSWER:
"""


    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]


    formatted_prompt = (
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    )


    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    )


    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=350,
            do_sample=False,
            repetition_penalty=1.05
        )


    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]


    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )


    return answer.strip()