"""Command line: python -m askregs ingest | search "question" [-k 5] [--mode dense]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ingest import ingest
from .search import Retriever

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "chunks.jsonl"
EMBEDDINGS = ROOT / "data" / "embeddings.npy"


def load_sources() -> dict:
    return {s["id"]: s for s in json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))}


def cite(chunk, sources) -> str:
    para = f"para {chunk.para}" if chunk.para else "unnumbered text"
    return f"{sources[chunk.doc_id]['short']}, {para}, p.{chunk.page}"


def main() -> None:
    parser = argparse.ArgumentParser(prog="askregs")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="chunk the PDFs in data/raw into data/chunks.jsonl")
    search = sub.add_parser("search", help="search the chunks")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=5)
    search.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="dense")
    args = parser.parse_args()

    sources = load_sources()

    if args.command == "ingest":
        chunks = ingest(ROOT / "sources.json", ROOT / "data" / "raw", CHUNKS)
        EMBEDDINGS.unlink(missing_ok=True)  # chunks changed, so the cached vectors are stale
        for doc_id in sources:
            print(f"{doc_id}: {sum(c.doc_id == doc_id for c in chunks)} chunks")
        return

    retriever = Retriever(CHUNKS, EMBEDDINGS)
    for rank, hit in enumerate(retriever.search(args.query, args.k, args.mode), start=1):
        print(f"{rank}. [{cite(hit.chunk, sources)}] score {hit.score:.3f}")
        print(f"   {hit.chunk.text[:220]}...\n")


if __name__ == "__main__":
    main()
