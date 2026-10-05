import re


def clean_text(text):
    """
    Clean extracted PDF text while preserving useful academic content.
    """

    if not text:
        return ""

    # Replace repeated whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Fix excessive line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # Strip leading/trailing spaces
    text = text.strip()

    return text