"""Reranker semántico en dos etapas.

1. Recuperación densa (bi-encoder, por defecto BAAI/bge-m3): descarta lo claramente ajeno.
2. Cross-encoder (por defecto BAAI/bge-reranker-v2-m3): puntúa cada par
   (contexto de empresa, documento) con una probabilidad 0-1.

`query_context` permite desambiguar: "Apple, empresa tecnológica" en lugar de "Apple".
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from src.common.schemas import UnifiedMediaDocument

logger = logging.getLogger(__name__)

EMBED_MODEL = os.getenv("RERANK_EMBED_MODEL", "BAAI/bge-m3")
CROSS_MODEL = os.getenv("RERANK_CROSS_MODEL", "BAAI/bge-reranker-v2-m3")


class SemanticReranker:
    def __init__(
        self,
        embed_model: str = EMBED_MODEL,
        cross_model: str = CROSS_MODEL,
        device: Optional[str] = None,
        prefilter_k: int = 50,
        max_chars: int = 2000,
    ) -> None:
        self.embed_model_name = embed_model
        self.cross_model_name = cross_model
        self.device = device
        self.prefilter_k = prefilter_k
        self.max_chars = max_chars
        self._embedder = None
        self._cross = None

    # ---------- carga perezosa (los modelos pesan varios GB) ----------
    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Cargando bi-encoder %s", self.embed_model_name)
            self._embedder = SentenceTransformer(self.embed_model_name, device=self.device)
        return self._embedder

    @property
    def cross(self):
        if self._cross is None:
            from sentence_transformers import CrossEncoder
            logger.info("Cargando cross-encoder %s", self.cross_model_name)
            self._cross = CrossEncoder(self.cross_model_name, device=self.device, max_length=512)
        return self._cross

    # ---------- API del módulo ----------
    def encode_documents(self, docs: List[UnifiedMediaDocument]) -> np.ndarray:
        texts = [d.rerank_text(self.max_chars) for d in docs]
        return self.embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def dense_prefilter(self, query: str, docs: List[UnifiedMediaDocument]) -> List[UnifiedMediaDocument]:
        if len(docs) <= self.prefilter_k:
            return docs
        q = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = self.encode_documents(docs) @ q
        order = np.argsort(-sims)[: self.prefilter_k]
        return [docs[i] for i in order]

    def cross_score_relevance(self, query: str, docs: List[UnifiedMediaDocument]) -> List[float]:
        if not docs:
            return []
        pairs = [(query, d.rerank_text(self.max_chars)) for d in docs]
        raw = np.asarray(self.cross.predict(pairs, convert_to_numpy=True), dtype=float)
        # Si el modelo devuelve logits (fuera de [0,1]) los pasamos por una sigmoide
        if raw.min() < 0 or raw.max() > 1:
            raw = 1.0 / (1.0 + np.exp(-raw))
        return raw.tolist()

    @staticmethod
    def select_top_n(
        docs: List[UnifiedMediaDocument],
        n: int,
        balanced: bool = True,
        min_per_source: int = 1,
        min_score: float = 0.0,
    ) -> List[UnifiedMediaDocument]:
        """Top-N por relevance_score. Con balanced=True garantiza una cuota mínima por fuente
        (solo con documentos que superan min_score), y el resto se rellena por puntuación."""
        pool = sorted(
            (d for d in docs if (d.relevance_score or 0.0) >= min_score),
            key=lambda d: d.relevance_score or 0.0,
            reverse=True,
        )
        if not balanced:
            return pool[:n]
        by_source: Dict[str, List[UnifiedMediaDocument]] = defaultdict(list)
        for d in pool:
            by_source[d.source.value].append(d)
        chosen: List[UnifiedMediaDocument] = []
        for items in by_source.values():
            chosen += items[:min_per_source]
        chosen = sorted(chosen, key=lambda d: d.relevance_score or 0.0, reverse=True)[:n]
        ids = {d.doc_id for d in chosen}
        for d in pool:
            if len(chosen) >= n:
                break
            if d.doc_id not in ids:
                chosen.append(d)
                ids.add(d.doc_id)
        return sorted(chosen, key=lambda d: d.relevance_score or 0.0, reverse=True)

    def rerank(
        self,
        company: str,
        docs: List[UnifiedMediaDocument],
        n: int,
        query_context: Optional[str] = None,
        balanced: bool = True,
        min_score: float = 0.0,
    ) -> List[UnifiedMediaDocument]:
        """Flujo completo: prefiltro denso -> cross-encoder -> selección Top-N."""
        query = query_context or company
        candidates = self.dense_prefilter(query, docs)
        for d, s in zip(candidates, self.cross_score_relevance(query, candidates)):
            d.relevance_score = float(np.clip(s, 0.0, 1.0))
        return self.select_top_n(candidates, n, balanced=balanced, min_score=min_score)
