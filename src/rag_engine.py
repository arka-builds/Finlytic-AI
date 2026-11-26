from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
from functools import lru_cache
import math
import re
from collections import Counter

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

# BM25 scores are unbounded, so they are normalised by the best score a
# perfect query match could reach (sum of the query terms' IDF) before gating.
MIN_BM25_NORMALISED = 0.30
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60  # standard reciprocal-rank-fusion damping constant
RETRIEVAL_METHODS = ("hybrid", "dense", "bm25")
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
# BM25 lexical ranker
# ============================================================

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an the and or of to in on for is are be it its i my me do does how what "
    "should can this that with as at by from".split()
)


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25:
    """Okapi BM25 over pre-tokenised documents (no external dependency)."""

    def __init__(self, documents: List[List[str]], k1=BM25_K1, b=BM25_B):
        self.k1, self.b = k1, b
        self.doc_freqs = [Counter(d) for d in documents]
        self.doc_len = np.array([len(d) for d in documents], dtype=float)
        self.avg_len = float(self.doc_len.mean()) if len(documents) else 0.0
        n = len(documents)
        df = Counter(t for d in documents for t in set(d))
        self.idf = {
            t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()
        }

    def scores(self, query_tokens: List[str]):
        """Return (raw_scores, max_attainable) for a tokenised query."""
        raw = np.zeros(len(self.doc_freqs))
        terms = [t for t in set(query_tokens) if t in self.idf]
        for i, freqs in enumerate(self.doc_freqs):
            for t in terms:
                f = freqs.get(t, 0)
                if f:
                    denom = f + self.k1 * (
                        1 - self.b + self.b * self.doc_len[i] / (self.avg_len or 1)
                    )
                    raw[i] += self.idf[t] * f * (self.k1 + 1) / denom
        ceiling = sum(self.idf[t] for t in terms) * (self.k1 + 1) or 1.0
        return raw, ceiling


def reciprocal_rank_fusion(rankings: List[List[int]], k: int = RRF_K):
    """Fuse several best-first index rankings into one {index: score} map."""
    fused: dict = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


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
        self.bm25 = None

        if not self.chunks:
            return

        documents = [chunk.text for chunk in self.chunks]
        self.bm25 = BM25([tokenize(f"{c.section} {c.text}") for c in self.chunks])
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

    def _vector_scores(self, query: str):
        """Dense (or TF-IDF fallback) cosine scores for every chunk."""
        if self.retrieval_mode == "semantic":
            q = self.model.encode(query, normalize_embeddings=True, show_progress_bar=False)
            return np.dot(np.asarray(self.embeddings), np.asarray(q))
        return cosine_similarity(self.tfidf_matrix, self.vectorizer.transform([query])).ravel()

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        method: str = "hybrid",
    ):
        """Return relevance-gated chunks, best first.

        method:
          * ``dense``  - embedding cosine (TF-IDF cosine when offline)
          * ``bm25``   - lexical BM25
          * ``hybrid`` - reciprocal-rank fusion of both rankers

        A chunk is returned only if at least one ranker scores it above that
        ranker's absolute relevance threshold, so off-topic questions return
        nothing instead of the "least bad" chunk.
        """
        if method not in RETRIEVAL_METHODS:
            raise ValueError(f"method must be one of {RETRIEVAL_METHODS}")
        if not self.chunks or not query or not query.strip():
            return []
        if self.embeddings is None and self.tfidf_matrix is None:
            return []

        top_k = max(1, min(int(top_k), len(self.chunks)))
        dense = self._vector_scores(query)
        raw_bm25, ceiling = self.bm25.scores(tokenize(query))
        lexical = raw_bm25 / ceiling

        dense_gate = MIN_RELEVANCE_SCORES[self.retrieval_mode]
        passes = (dense >= dense_gate) | (lexical >= MIN_BM25_NORMALISED)
        if method == "dense":
            passes = dense >= dense_gate
            order = np.argsort(dense)[::-1]
            fused = {int(i): float(dense[i]) for i in order}
        elif method == "bm25":
            passes = lexical >= MIN_BM25_NORMALISED
            order = np.argsort(lexical)[::-1]
            fused = {int(i): float(lexical[i]) for i in order}
        else:
            fused = reciprocal_rank_fusion([
                [int(i) for i in np.argsort(dense)[::-1]],
                [int(i) for i in np.argsort(lexical)[::-1]],
            ])

        ranked = sorted(fused, key=fused.get, reverse=True)
        results = []
        for index in ranked:
            if not passes[index]:
                continue
            chunk = self.chunks[index]
            results.append({
                "source": chunk.source,
                "section": chunk.section,
                "chunk_id": chunk.chunk_id,
                "score": float(max(dense[index], 0.0)),
                "dense_score": float(dense[index]),
                "bm25_score": float(lexical[index]),
                "fused_score": float(fused[index]),
                "text": chunk.text,
            })
            if len(results) == top_k:
                break
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

def retrieve_chunks(query: str, top_k: int = DEFAULT_TOP_K, method: str = "hybrid"):
    """Return relevance-filtered chunks as structured dictionaries."""
    return get_retriever().retrieve(query, top_k=top_k, method=method)


def format_context(results) -> str:
    """Render retrieved chunks as numbered Markdown so answers can cite [n]."""
    if not results:
        return NO_CONTEXT_MESSAGE

    return "\n\n---\n\n".join(
        f"### [{n}] {r['section']}\n"
        f"**Source:** {r['source']}  \n"
        f"**Relevance:** {r['score']:.3f}\n\n"
        f"{r['text']}"
        for n, r in enumerate(results, start=1)
    )


def format_sources(results) -> str:
    """Markdown footer listing the passages an answer was grounded on."""
    if not results:
        return ""
    lines = [f"- [{n}] **{r['section']}** — `{r['source']}`" for n, r in enumerate(results, start=1)]
    return "**Sources**\n" + "\n".join(lines)


def retrieve_context(query: str, top_k: int = DEFAULT_TOP_K, method: str = "hybrid") -> str:
    """Retrieve and format knowledge-base context for ``query``."""
    return format_context(retrieve_chunks(query, top_k=top_k, method=method))


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