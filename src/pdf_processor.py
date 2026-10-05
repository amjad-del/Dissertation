import fitz
import os
import uuid

from src.cleaner import clean_text


UPLOAD_DIR = "data/uploaded_papers"
IMAGE_DIR = "data/processed/images"


def save_uploaded_pdf(pdf_file):
    """
    Save the original uploaded PDF locally.
    """

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    file_path = os.path.join(
        UPLOAD_DIR,
        pdf_file.name
    )

    pdf_file.seek(0)

    with open(file_path, "wb") as f:
        f.write(pdf_file.read())

    pdf_file.seek(0)

    return file_path


def extract_text_and_images_from_pdf(pdf_file):
    """
    Extract cleaned text, images, and metadata page by page.
    """

    os.makedirs(IMAGE_DIR, exist_ok=True)

    paper_name = pdf_file.name

    paper_id = os.path.splitext(
        paper_name
    )[0]

    pdf_file.seek(0)
    pdf_bytes = pdf_file.read()

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page_number in range(len(document)):

        page = document.load_page(
            page_number
        )

        raw_text = page.get_text()

        cleaned_text = clean_text(
            raw_text
        )

        extracted_images = []

        image_list = page.get_images(
            full=True
        )

        for image in image_list:

            xref = image[0]

            base_image = document.extract_image(
                xref
            )

            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_name = (
                f"{paper_id}_"
                f"page_{page_number + 1}_"
                f"{uuid.uuid4().hex[:8]}."
                f"{image_ext}"
            )

            image_path = os.path.join(
                IMAGE_DIR,
                image_name
            )

            with open(
                image_path,
                "wb"
            ) as image_file:

                image_file.write(
                    image_bytes
                )

            extracted_images.append(
                image_path
            )

        pages.append(
            {
                "paper_id": paper_id,
                "paper_name": paper_name,
                "page_number": page_number + 1,
                "text": cleaned_text,
                "images": extracted_images
            }
        )

    document.close()

    return pages