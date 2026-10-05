import re


SECTION_MAP = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "literature review": "Literature Review",
    "related work": "Literature Review",
    "background": "Literature Review",

    "methodology": "Methodology",
    "method": "Methodology",
    "methods": "Methodology",
    "materials and methods": "Methodology",
    "proposed research methodology": "Methodology",

    "dataset": "Dataset",
    "data collection": "Dataset",

    "proposed framework": "Proposed Framework",
    "proposed method": "Proposed Framework",
    "framework": "Proposed Framework",

    "results": "Results",
    "results and discussion": "Results",
    "experimental results": "Results",

    "discussion": "Discussion",

    "limitations": "Limitations",
    "limitation": "Limitations",

    "future work": "Future Work",
    "future directions": "Future Work",

    "conclusion": "Conclusion",
    "conclusions": "Conclusion",

    "references": "References",
    "bibliography": "References"
}


def normalize_heading(line):
    """
    Normalize a possible section heading.
    """

    if not line:
        return ""

    text = line.strip().lower()

    # Remove numbering such as:
    # 1 Introduction
    # 2.1 Methodology
    # III. Results
    text = re.sub(
        r"^\s*(?:\d+(?:\.\d+)*|[ivxlcdm]+)[.)]?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def detect_section(line):
    """
    Detect section only when the ENTIRE line represents
    a heading.

    This avoids incorrectly detecting words such as
    'results' when they appear inside normal sentences.
    """

    normalized = normalize_heading(line)

    if not normalized:
        return None

    if normalized in SECTION_MAP:
        return SECTION_MAP[normalized]

    return None


def split_page_into_sections(text, current_section="Unknown"):
    """
    Split a page into sections using only exact heading lines.

    Returns:
        list of dictionaries containing:
        section
        text
    """

    if not text or not text.strip():
        return [], current_section

    lines = text.splitlines()

    sections = []

    active_section = current_section
    section_buffer = []

    for line in lines:

        stripped_line = line.strip()

        if not stripped_line:
            continue

        detected_section = detect_section(stripped_line)

        if detected_section:

            # Save previous section
            if section_buffer:

                sections.append(
                    {
                        "section": active_section,
                        "text": " ".join(section_buffer)
                    }
                )

            active_section = detected_section
            section_buffer = []

        else:

            section_buffer.append(stripped_line)

    # Save final section
    if section_buffer:

        sections.append(
            {
                "section": active_section,
                "text": " ".join(section_buffer)
            }
        )

    return sections, active_section


def create_chunks(
    pages,
    chunk_size=500,
    overlap=100
):
    """
    Create text chunks from extracted PDF pages.

    Each chunk contains:
    - paper ID
    - paper name
    - page number
    - section
    - chunk ID
    - text
    """

    chunks = []

    current_section = "Unknown"

    for page in pages:

        text = page.get("text", "")

        if not text.strip():
            continue

        sections, current_section = split_page_into_sections(
            text,
            current_section
        )

        for section_data in sections:

            section = section_data["section"]
            section_text = section_data["text"]

            # Do not index references
            if section == "References":
                continue

            words = section_text.split()

            if not words:
                continue

            start = 0
            chunk_number = 1

            while start < len(words):

                end = start + chunk_size

                chunk_words = words[start:end]

                chunk_text = " ".join(chunk_words)

                if not chunk_text.strip():
                    break

                safe_section = re.sub(
                    r"[^A-Za-z0-9]+",
                    "_",
                    section
                ).strip("_")

                if not safe_section:
                    safe_section = "Unknown"

                chunk_id = (
                    f"{page['paper_id']}_"
                    f"p{page['page_number']}_"
                    f"{safe_section}_"
                    f"c{chunk_number}"
                )

                chunks.append(
                    {
                        "paper_id": page["paper_id"],
                        "paper_name": page["paper_name"],
                        "page_number": page["page_number"],
                        "section": section,
                        "chunk_id": chunk_id,
                        "text": chunk_text
                    }
                )

                if end >= len(words):
                    break

                start = end - overlap
                chunk_number += 1

    return chunks