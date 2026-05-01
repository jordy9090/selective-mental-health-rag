import json
import argparse
from pathlib import Path
from rank_bm25 import BM25Okapi


CHUNKS_PATH = Path("data/processed/chunks.jsonl")


def load_chunks(path: Path):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def tokenize(text: str):
    return text.lower().split()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument(
        "--source_family",
        type=str,
        choices=["coping", "psychoeducation", "safety"],
        default=None,
        help="Optional filter by source family"
    )
    args = parser.parse_args()

    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    if args.source_family is not None:
        chunks = [c for c in chunks if c.get("source_family") == args.source_family]
        print(f"Filtered to {len(chunks)} chunks for source_family={args.source_family}")

    if not chunks:
        print("[ERROR] No chunks available after filtering.")
        return

    corpus_tokens = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(corpus_tokens)

    query_tokens = tokenize(args.query)
    scores = bm25.get_scores(query_tokens)

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )[:args.top_k]

    print("\n=== QUERY ===")
    print(args.query)

    print("\n=== TOP RESULTS ===")
    for rank, (chunk, score) in enumerate(ranked, start=1):
        print(f"\n[Rank {rank}] score={score:.4f}")
        print(f"doc_id: {chunk['doc_id']}")
        print(f"title: {chunk['title']}")
        print(f"source_family: {chunk['source_family']}")
        print(f"file_name: {chunk['file_name']}")
        print(f"file_type: {chunk.get('file_type', 'unknown')}")
        print(f"chunk_id: {chunk['chunk_id']}")
        print(f"text: {chunk['text'][:1000]}")
        print("-" * 80)


if __name__ == "__main__":
    main()
