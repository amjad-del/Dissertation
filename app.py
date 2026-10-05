import streamlit as st

from src.pdf_processor import (
    save_uploaded_pdf,
    extract_text_and_images_from_pdf
)

from src.chunker import create_chunks
from src.embeddings import generate_embeddings
from src.vector_store import create_faiss_index
from src.retriever import retrieve_relevant_chunks
from src.rag_engine import generate_rag_answer


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Automated Literature Review",
    page_icon="📚",
    layout="wide"
)


# ==================================================
# TITLE
# ==================================================

st.title(
    "📚 Automated Literature Review using LLM and RAG"
)

st.write(
    "Upload a research paper and ask questions "
    "using Retrieval-Augmented Generation (RAG)."
)


# ==================================================
# PDF UPLOAD
# ==================================================

uploaded_files = st.file_uploader(
    "Upload Research Paper",
    type=["pdf"],
    accept_multiple_files=True
)


# ==================================================
# PROCESS FILES
# ==================================================

if uploaded_files:

    st.success(
        f"{len(uploaded_files)} research paper(s) "
        f"uploaded successfully."
    )


    for uploaded_file in uploaded_files:

        st.divider()

        st.header(
            f"📄 {uploaded_file.name}"
        )


        # ==========================================
        # SAVE PDF
        # ==========================================

        save_uploaded_pdf(
            uploaded_file
        )


        # ==========================================
        # PDF TEXT EXTRACTION
        # ==========================================

        with st.spinner(
            "Extracting research paper content..."
        ):

            pages = (
                extract_text_and_images_from_pdf(
                    uploaded_file
                )
            )


        # ==========================================
        # CHUNKING
        # ==========================================

        chunks = create_chunks(
            pages,
            chunk_size=500,
            overlap=100
        )


        # ==========================================
        # EMBEDDINGS
        # ==========================================

        with st.spinner(
            "Creating vector embeddings..."
        ):

            embeddings = generate_embeddings(
                chunks
            )


        # ==========================================
        # FAISS VECTOR INDEX
        # ==========================================

        faiss_index = create_faiss_index(
            embeddings
        )


        # ==========================================
        # PROCESSING SUMMARY
        # ==========================================

        st.subheader(
            "⚙️ Document Processing"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "PDF Pages",
                len(pages)
            )


        with col2:

            st.metric(
                "Text Chunks",
                len(chunks)
            )


        with col3:

            if faiss_index is not None:

                st.metric(
                    "Vector Embeddings",
                    faiss_index.ntotal
                )

            else:

                st.metric(
                    "Vector Embeddings",
                    0
                )


        if faiss_index is not None:

            st.success(
                "✓ Research paper processed and "
                "indexed successfully."
            )

        else:

            st.error(
                "The vector index could not be created."
            )


        # ==========================================
        # OPTIONAL DOCUMENT DETAILS
        # ==========================================

        with st.expander(
            "📄 View Extracted Document Information"
        ):

            for page in pages:

                st.markdown(
                    f"### Page "
                    f"{page['page_number']}"
                )

                st.write(
                    page["text"]
                )

                st.divider()


        # ==========================================
        # RAG QUESTION ANSWERING
        # ==========================================

        st.divider()

        st.subheader(
            "🔎 Ask Questions About the Research Paper"
        )

        st.write(
            "Ask a question and the system will retrieve "
            "relevant evidence from the uploaded paper "
            "before generating the answer."
        )


        question = st.text_input(
            "Research Question",
            key=f"question_{uploaded_file.name}",
            placeholder=(
                "Example: What methodology was "
                "used in this research?"
            )
        )


        ask_button = st.button(
            "🔍 Generate RAG Answer",
            key=f"ask_{uploaded_file.name}"
        )


        # ==========================================
        # GENERATE ANSWER
        # ==========================================

        if ask_button:

            if not question.strip():

                st.warning(
                    "Please enter a question first."
                )


            elif faiss_index is None:

                st.error(
                    "The research paper has not been "
                    "indexed successfully."
                )


            else:

                # ==================================
                # RETRIEVAL
                # ==================================

                with st.spinner(
                    "Searching the research paper..."
                ):

                    results = (
                        retrieve_relevant_chunks(
                            question,
                            faiss_index,
                            chunks,
                            top_k=3
                        )
                    )


                # ==================================
                # RAG GENERATION
                # ==================================

                with st.spinner(
                    "Generating evidence-based answer..."
                ):

                    answer = (
                        generate_rag_answer(
                            question,
                            results
                        )
                    )


                # ==================================
                # DISPLAY QUESTION
                # ==================================

                st.divider()

                st.subheader(
                    "❓ Question"
                )

                st.info(
                    question
                )


                # ==================================
                # DISPLAY ANSWER
                # ==================================

                st.subheader(
                    "🤖 RAG Answer"
                )

                st.write(
                    answer
                )


                # ==================================
                # RETRIEVED SOURCES
                # ==================================

                st.subheader(
                    "📌 Supporting Evidence"
                )

                st.write(
                    "The following research-paper "
                    "passages were retrieved before "
                    "the answer was generated."
                )


                if not results:

                    st.warning(
                        "No relevant evidence was found."
                    )


                else:

                    for i, result in enumerate(
                        results,
                        start=1
                    ):

                        page_number = result.get(
                            "page_number",
                            "Unknown"
                        )

                        section = result.get(
                            "section",
                            "Unknown"
                        )

                        with st.expander(
                            f"Evidence {i} — "
                            f"Page {page_number} — "
                            f"{section}"
                        ):

                            st.write(
                                f"**Paper:** "
                                f"{result.get('paper_name', 'Unknown')}"
                            )

                            st.write(
                                f"**Page:** "
                                f"{page_number}"
                            )

                            st.write(
                                f"**Section:** "
                                f"{section}"
                            )

                            st.write(
                                "**Retrieved Passage:**"
                            )

                            st.write(
                                result.get(
                                    "text",
                                    ""
                                )
                            )


        # ==========================================
        # REVIEW 2 INFORMATION
        # ==========================================

        st.divider()

        with st.expander(
            "ℹ️ RAG Pipeline"
        ):

            st.write(
                """
                **Current RAG Pipeline**

                PDF Upload
                → Text Extraction
                → Text Cleaning
                → Section-aware Chunking
                → Sentence Transformer Embeddings
                → FAISS Vector Search
                → BM25 Keyword Retrieval
                → Cross-Encoder Reranking
                → Qwen Local LLM
                → Evidence-based Answer
                """
            )