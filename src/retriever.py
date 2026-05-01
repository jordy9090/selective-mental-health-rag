import json
from typing import List, Optional, Sequence, Tuple

from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, chunks_path: str = "data/processed/chunks.jsonl"):
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]

        # BM25는 전체 코퍼스 기준으로 한 번만 구축
        self.tokenized_corpus = [self._tokenize(c.get("text", "")) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return (text or "").lower().split()

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        allowed_families: Optional[Sequence[str]] = None,
    ) -> List[Tuple[dict, float]]:
        """
        allowed_families가 주어지면 해당 source_family의 chunk만 후보로 사용.
        예:
            retrieve(q, top_k=3, allowed_families=["coping"])
            retrieve(q, top_k=3, allowed_families=["coping", "psychoeducation"])
        """
        query = (query or "").strip()
        if not query:
            return []

        scores = self.bm25.get_scores(self._tokenize(query))

        candidate_indices = list(range(len(self.chunks)))
        if allowed_families:
            allowed = {x.strip().lower() for x in allowed_families}
            candidate_indices = [
                i
                for i, c in enumerate(self.chunks)
                if str(c.get("source_family", "")).strip().lower() in allowed
            ]

        if not candidate_indices:
            return []

        ranked = sorted(candidate_indices, key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked]
