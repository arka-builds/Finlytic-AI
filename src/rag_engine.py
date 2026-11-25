from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
from functools import lru_cache
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # Semantic model is optional; lexical retrieval remains available.
    SentenceTransformer = None


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "data" / "knowledge_base"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_TOP_K = 3
# Cosine-similarity cut-offs differ by retrieval mode: MiniLM embeddings give
# higher scores for genuinely related text than sparse TF-IDF vectors do, so
# one shared threshold would either admit noise or starve the offline fallback.
MIN_RELEVANCE_SCORES = {
    "semantic": 0.40,
    "tfidf_fallback": 0.15,
}
NO_CONTEXT_MESSAGE = "No relevant knowledge-base context was found."
MAX_WORDS = 180
OVERLAP_WORDS = 30


# ============================================================
# Data structure
# ============================================================

@dataclass
class DocumentChunk:
    source: str
    section: str
    chunk_id: int
    text: str


# ============================================================
# Markdown Knowledge Base Loader
# ============================================================

def load_markdown_documents() -> List[DocumentChunk]:
    """
    Load Markdown files from the knowledge base and split them
    into section-aware chunks.
    """

    chunks: List[DocumentChunk] = []

    if not KNOWLEDGE_BASE_DIR.exists():
        return chunks

    chunk_counter = 0

    for file_path in sorted(KNOWLEDGE_BASE_DIR.glob("*.md")):

        try:
            content = file_path.read_text(
                encoding="utf-8"
            )
        except Exception:
            continue

        lines = content.splitlines()

        current_section = "General"
        current_lines: List[str] = []

        sections = []

        for line in lines:

            # Supports:
            # ## Emergency Fund
            # \## Emergency Fund
            heading = re.match(
                r"^\s*\\?#{1,3}\s+(.+?)\s*$",
                line
            )

            if heading:

                if current_lines:
                    sections.append(
                        (
                            current_section,
                            "\n".join(current_lines).strip()
                        )
                    )

                current_section = heading.group(1).strip()
                current_lines = []

            else:
                current_lines.append(line)

        # Add final section
        if current_lines:
            sections.append(
                (
                    current_section,
                    "\n".join(current_lines).strip()
                )
            )

        # Create chunks
        for section_name, section_text in sections:

            if not section_text:
                continue

            words = section_text.split()

            if not words:
                continue

            start = 0

            while start < len(words):

                end = min(
                    start + MAX_WORDS,
                    len(words)
                )

                chunk_words = words[start:end]

                chunk_text = " ".join(chunk_words).strip()

                if chunk_text:

                    chunks.append(
                        DocumentChunk(
                            source=file_path.name,
                            section=section_name,
                            chunk_id=chunk_counter,
                            text=chunk_text
                        )
                    )

                    chunk_counter += 1

                if end >= len(words):
                    break

                start = max(
                    end - OVERLAP_WORDS,
                    start + 1
                )

    return chunks


# ============================================================
# Sentence Transformer Model
# ============================================================

@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Load the embedding model only once.

    Streamlit reruns the script frequently, so caching prevents
    the model from being loaded repeatedly.
    """

    if SentenceTransformer is None:
        raise RuntimeError(
            "Semantic RAG requires sentence-transformers. "
            "Install project dependencies with: pip install -r requirements.txt"
        )
    return SentenceTransformer(MODEL_NAME)


# ============================================================
# Semantic RAG Retriever
# ============================================================

class LocalRAGRetriever:

    def __init__(self):

        self.chunks = load_markdown_documents()

        self.model = None
        self.embeddings = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.retrieval_mode = "unavailable"

        if not self.chunks:
            return

        documents = [chunk.text for chunk in self.chunks]
        try:
            self.model = get_embedding_model()
            self.embeddings = self.model.encode(
                documents,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self.retrieval_mode = "semantic"
        except Exception:
            # Do not make Finance Chat unusable when model downloads are
            # blocked, sentence-transformers is missing, or a model is offline.
            self.vectorizer = TfidfVectorizer(
                stop_words="english", ngram_range=(1, 2),
                max_features=12000,
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(documents)
            self.retrieval_mode = "tfidf_fallback"

    # --------------------------------------------------------
    # Retrieve relevant chunks
    # --------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K
    ):

        if not self.chunks:
            return []

        if not query or not query.strip():
            return []

        if self.embeddings is None and self.tfidf_matrix is None:
            return []

        top_k = max(1, min(int(top_k), len(self.chunks)))

        if self.retrieval_mode == "semantic":
            query_embedding = self.model.encode(
                query, normalize_embeddings=True, show_progress_bar=False
            )
            scores = np.dot(
                np.asarray(self.embeddings), np.asarray(query_embedding)
            )
        else:
            query_vector = self.vectorizer.transform([query])
            scores = cosine_similarity(self.tfidf_matrix, query_vector).ravel()

        ranked_indices = np.argsort(
            scores
        )[::-1]

        results = []

        for index in ranked_indices[:top_k]:

            score = float(scores[index])

            # Ignore weak/unrelated matches
            if score < MIN_RELEVANCE_SCORES[self.retrieval_mode]:
                continue

            chunk = self.chunks[index]

            results.append(
                {
                    "source": chunk.source,
                    "section": chunk.section,
                    "chunk_id": chunk.chunk_id,
                    "score": score,
                    "text": chunk.text
                }
            )

        return results


# ============================================================
# Cached Retriever
# ============================================================

@lru_cache(maxsize=1)
def get_retriever():
    """
    Create the RAG retriever once and reuse it.
    """

    return LocalRAGRetriever()


# ============================================================
# Public retrieve_context() function
# ============================================================

def retrieve_chunks(query: str, top_k: int = DEFAULT_TOP_K):
    """Return relevance-filtered chunks as structured dictionaries."""
    return get_retriever().retrieve(query, top_k=top_k)


def format_context(results) -> str:
    """Render retrieved chunks as Markdown for display or LLM prompts."""
    if not results:
        return NO_CONTEXT_MESSAGE

    return "\n\n---\n\n".join(
        f"### {r['section']}\n"
        f"**Source:** {r['source']}  \n"
        f"**Relevance:** {r['score']:.3f}\n\n"
        f"{r['text']}"
        for r in results
    )


def retrieve_context(query: str, top_k: int = DEFAULT_TOP_K) -> str:
    """Retrieve and format knowledge-base context for ``query``."""
    return format_context(retrieve_chunks(query, top_k=top_k))


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Finlytic AI Semantic RAG")
    print("=" * 60)

    chunks = load_markdown_documents()

    print(f"Loaded chunks: {len(chunks)}")

    if chunks:

        print("\nSections:")

        sections = []

        for chunk in chunks:

            if chunk.section not in sections:
                sections.append(chunk.section)

        for section in sections:
            print(f" - {section}")

    print("\nLoading embedding model...")

    retriever = get_retriever()

    print(
        f"Embedding dimension: "
        f"{retriever.embeddings.shape[1]}"
        if retriever.embeddings is not None
        else "No embeddings created."
    )

    print("\nTest Query:")
    query = "How much money should I keep for emergencies?"

    print(query)

    print("\nRetrieved Context:")
    print("-" * 60)

    print(
        retrieve_context(
            query,
            top_k=3
        )
    )

    print("\n" + "=" * 60)
    print("Semantic RAG test completed.")
    print("=" * 60)