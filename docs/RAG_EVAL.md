# Retrieval evaluation

30 in-scope questions (paraphrased, several with little keyword overlap) and 6 off-topic questions, over the 6-section knowledge base. Regenerate with `python evals/run_rag_eval.py --write`.

| Method | hit@1 | hit@3 | MRR | Off-topic rejected |
|---|---|---|---|---|
| TF-IDF | 0.80 | 0.80 | 0.80 | 1.00 |
| BM25 | 0.80 | 0.80 | 0.80 | 1.00 |
| Hybrid (TF-IDF + BM25, RRF) | 0.90 | 0.90 | 0.90 | 1.00 |

**TF-IDF** missed: “If I earn 50000 and spend 40000 what do I save?”; “Is Netflix essential or discretionary spending?”; “Is the line between essential and optional spending the same for everyone?”; “What should I know before repaying a loan?”; “Is this app giving professional financial advice?”; “Will the assistant make up facts about my transactions?”

**BM25** missed: “How many months of expenses should I keep in reserve?”; “I lost my job suddenly, what money should I have set aside?”; “Why must the income and expense periods match?”; “Is Netflix essential or discretionary spending?”; “Is the line between essential and optional spending the same for everyone?”; “Will the assistant make up facts about my transactions?”

**Hybrid (TF-IDF + BM25, RRF)** missed: “Is Netflix essential or discretionary spending?”; “Is the line between essential and optional spending the same for everyone?”; “Will the assistant make up facts about my transactions?”

Caveats: small, hand-written set from one author; the knowledge base is only six sections, so scores will be optimistic compared with a large corpus. MiniLM rows appear only when the embedding model can be loaded.
