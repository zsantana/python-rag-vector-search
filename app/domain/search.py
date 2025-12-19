from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class SearchResult:
    id: int
    content: str
    score: float


class SearchStrategy(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int) -> List[SearchResult]:
        raise NotImplementedError


class VectorSearch(SearchStrategy):
    def __init__(self, embedding_model_name: str):
        self.embedding_model_name = embedding_model_name

    def search(self, query: str, top_k: int) -> List[SearchResult]:
        from app.infrastructure.embeddings import get_embedding_model
        from app.infrastructure.database import search_cosine_similarity

        embedding_model = get_embedding_model(self.embedding_model_name)
        query_embedding = embedding_model.embed_query(query)
        rows = search_cosine_similarity(query_embedding, limit=top_k)
        return [SearchResult(id=row[0], content=row[1], score=float(row[2])) for row in rows]


class SemanticSearch(SearchStrategy):
    def search(self, query: str, top_k: int) -> List[SearchResult]:
        from app.infrastructure.database import search_full_text

        rows = search_full_text(query, limit=top_k)
        return [SearchResult(id=row[0], content=row[1], score=float(row[2])) for row in rows]


class HybridSearch(SearchStrategy):
    def __init__(
        self,
        embedding_model_name: str,
        vector_weight: float,
        minimum_score: float,
    ):
        self.embedding_model_name = embedding_model_name
        self.vector_weight = float(vector_weight)
        self.minimum_score = float(minimum_score)

    def search(self, query: str, top_k: int) -> List[SearchResult]:
        vector_results = VectorSearch(self.embedding_model_name).search(query, top_k)
        semantic_results = SemanticSearch().search(query, top_k)

        def _normalize(results: List[SearchResult]) -> Dict[int, float]:
            if not results:
                return {}
            max_score = max(r.score for r in results) or 1.0
            return {r.id: (r.score / max_score) for r in results}

        vec_norm = _normalize(vector_results)
        sem_norm = _normalize(semantic_results)

        by_id: Dict[int, Dict[str, Any]] = {}
        for r in vector_results:
            by_id.setdefault(r.id, {"content": r.content, "vec": 0.0, "sem": 0.0})
            by_id[r.id]["vec"] = vec_norm.get(r.id, 0.0)

        for r in semantic_results:
            by_id.setdefault(r.id, {"content": r.content, "vec": 0.0, "sem": 0.0})
            by_id[r.id]["sem"] = sem_norm.get(r.id, 0.0)

        combined: List[SearchResult] = []
        for doc_id, d in by_id.items():
            score = (self.vector_weight * float(d["vec"])) + ((1.0 - self.vector_weight) * float(d["sem"]))
            if score >= self.minimum_score:
                combined.append(SearchResult(id=doc_id, content=str(d["content"]), score=float(score)))

        combined.sort(key=lambda r: r.score, reverse=True)
        return combined[: int(top_k)]
