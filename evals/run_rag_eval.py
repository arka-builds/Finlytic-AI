"""Evaluate retrieval quality on a labelled question set.

    python evals/run_rag_eval.py            # prints a table
    python evals/run_rag_eval.py --write    # also writes docs/RAG_EVAL.md

Metrics (in-scope questions): hit@1, hit@3, MRR.
Out-of-scope questions should retrieve nothing: "rejection rate".
Methods needing the embedding model are skipped when it cannot be loaded.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import rag_engine  # noqa: E402

EVAL_SET = ROOT / "evals" / "rag_eval_set.jsonl"


def load_set():
    return [json.loads(line) for line in EVAL_SET.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(retriever, method, rows, k=3):
    hits1 = hits3 = 0
    rr = 0.0
    in_scope = rejected = out_scope = 0
    misses = []
    for row in rows:
        results = retriever.retrieve(row["question"], top_k=k, method=method)
        sections = []
        for r in results:
            if r["section"] not in sections:
                sections.append(r["section"])
        if row["expected_section"] is None:
            out_scope += 1
            rejected += int(not results)
            continue
        in_scope += 1
        if row["expected_section"] in sections:
            rank = sections.index(row["expected_section"]) + 1
            hits1 += int(rank == 1)
            hits3 += int(rank <= 3)
            rr += 1.0 / rank
        else:
            misses.append(row["question"])
    return {
        "hit@1": hits1 / in_scope, "hit@3": hits3 / in_scope, "mrr": rr / in_scope,
        "rejection": rejected / out_scope if out_scope else float("nan"),
        "misses": misses,
    }


def build_retrievers():
    """Return {label: (retriever, method)} for every runnable configuration."""
    # Force the TF-IDF path so results are reproducible without a model download.
    original = rag_engine.get_embedding_model
    rag_engine.get_embedding_model = lambda: (_ for _ in ()).throw(RuntimeError("forced offline"))
    try:
        lexical = rag_engine.LocalRAGRetriever()
    finally:
        rag_engine.get_embedding_model = original
    configs = {
        "TF-IDF": (lexical, "dense"),
        "BM25": (lexical, "bm25"),
        "Hybrid (TF-IDF + BM25, RRF)": (lexical, "hybrid"),
    }
    try:
        semantic = rag_engine.LocalRAGRetriever()
        if semantic.retrieval_mode == "semantic":
            configs.update({
                "MiniLM dense": (semantic, "dense"),
                "Hybrid (MiniLM + BM25, RRF)": (semantic, "hybrid"),
            })
    except Exception:
        pass
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write docs/RAG_EVAL.md")
    args = parser.parse_args()
    rows = load_set()
    n_in = sum(r["expected_section"] is not None for r in rows)
    lines = [
        "| Method | hit@1 | hit@3 | MRR | Off-topic rejected |",
        "|---|---|---|---|---|",
    ]
    detail = []
    for label, (retriever, method) in build_retrievers().items():
        m = evaluate(retriever, method, rows)
        lines.append(f"| {label} | {m['hit@1']:.2f} | {m['hit@3']:.2f} | {m['mrr']:.2f} | {m['rejection']:.2f} |")
        if m["misses"]:
            detail.append(f"**{label}** missed: " + "; ".join(f"“{q}”" for q in m["misses"]))
    table = "\n".join(lines)
    print(table)
    print()
    print("\n".join(detail))
    if args.write:
        out = ROOT / "docs" / "RAG_EVAL.md"
        out.write_text(
            "# Retrieval evaluation\n\n"
            f"{n_in} in-scope questions (paraphrased, several with little keyword overlap) and "
            f"{len(rows) - n_in} off-topic questions, over the 6-section knowledge base. "
            "Regenerate with `python evals/run_rag_eval.py --write`.\n\n"
            f"{table}\n\n" + "\n\n".join(detail) + "\n\n"
            "Caveats: small, hand-written set from one author; the knowledge base is only six sections, "
            "so scores will be optimistic compared with a large corpus. MiniLM rows appear only when the "
            "embedding model can be loaded.\n",
            encoding="utf-8",
        )
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
