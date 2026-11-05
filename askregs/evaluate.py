"""Measure retrieval quality on the test set: python -m askregs.evaluate

For each question we know which paragraph answers it. We check where that paragraph lands
in each mode's ranking:
  hit@1  - share of questions where the right paragraph is the top result
  hit@5  - share where it is anywhere in the top 5 (what the answer step will read)
  MRR    - mean reciprocal rank: 1 for rank 1, 0.5 for rank 2, 0.33 for rank 3...
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .__main__ import CHUNKS, EMBEDDINGS, ROOT
from .search import Retriever

MODES = ("bm25", "dense", "hybrid")


def base_id(chunk_id: str) -> str:
    """'ADB-V1:3.30#2' and 'ADB-V1:3.30~2' both count as paragraph 'ADB-V1:3.30'."""
    return re.sub(r"[#~]\d+$", "", chunk_id)


def first_correct_rank(ranking, chunks, expected: set[str], limit: int = 50) -> int | None:
    for rank, (i, _) in enumerate(ranking[:limit], start=1):
        if base_id(chunks[i].id) in expected:
            return rank
    return None


def main() -> None:
    questions = [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").open(encoding="utf-8")]
    retriever = Retriever(CHUNKS, EMBEDDINGS)
    retriever.vectors  # build or load the embeddings before timing anything

    results = {}
    for mode in MODES:
        ranks, start = [], time.perf_counter()
        for item in questions:
            ranking = retriever.ranking(item["q"], mode)
            ranks.append(first_correct_rank(ranking, retriever.chunks, set(item["expected"])))
        elapsed_ms = (time.perf_counter() - start) / len(questions) * 1000
        results[mode] = {
            "hit@1": sum(r == 1 for r in ranks) / len(ranks),
            "hit@5": sum(r is not None and r <= 5 for r in ranks) / len(ranks),
            "mrr": sum(1 / r for r in ranks if r) / len(ranks),
            "ms_per_query": elapsed_ms,
            "misses": [q["q"] for q, r in zip(questions, ranks) if r is None or r > 5],
        }

    print(f"{len(questions)} questions, {len(retriever.chunks)} chunks\n")
    print(f"{'mode':<8}{'hit@1':>8}{'hit@5':>8}{'MRR':>8}{'ms/query':>10}")
    for mode, r in results.items():
        print(f"{mode:<8}{r['hit@1']:>8.0%}{r['hit@5']:>8.0%}{r['mrr']:>8.2f}{r['ms_per_query']:>10.0f}")
    for mode, r in results.items():
        if r["misses"]:
            print(f"\n{mode} missed the top 5 on:")
            for q in r["misses"]:
                print(f"  - {q}")

    out = ROOT / "eval" / "results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
