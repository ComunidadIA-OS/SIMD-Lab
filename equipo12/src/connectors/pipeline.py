"""Punto de entrada único de la ingesta.

    from src.connectors.pipeline import run_ingestion_sync
    res = run_ingestion_sync("Iberdrola", top_n=5, context="Iberdrola, empresa energética")
    res.documents  -> List[UnifiedMediaDocument]  (Contrato A)
    res.reports    -> estado de cada fuente

Flujo: búsqueda federada -> reranker -> descarga de multimedia.
Quien analice los documentos debe llamar después a purge(res.documents).
"""
from __future__ import annotations

import os

# El reranker usa PyTorch: que transformers no cargue TensorFlow
# (en Windows lo bloquea Smart App Control y no hace falta aquí)
os.environ.setdefault("USE_TF", "0")

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from dotenv import load_dotenv

load_dotenv()  # antes de importar el resto: claves de API, User-Agent y modelos

from src.common.schemas import MediaSource, UnifiedMediaDocument  # noqa: E402
from src.connectors.media_downloader import MediaDownloader  # noqa: E402
from src.connectors.orchestrator import FederatedSearchOrchestrator, SourceReport  # noqa: E402
from src.reranker.semantic_ranker import SemanticReranker  # noqa: E402

logger = logging.getLogger(__name__)

LIGHT_EMBED = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LIGHT_CROSS = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

_rerankers: dict = {}


def get_reranker(light: bool = False) -> SemanticReranker:
    """Carga los modelos una sola vez por proceso (pesan varios GB)."""
    if light not in _rerankers:
        kw = {"embed_model": LIGHT_EMBED, "cross_model": LIGHT_CROSS} if light else {}
        _rerankers[light] = SemanticReranker(**kw)
    return _rerankers[light]


@dataclass
class IngestionResult:
    documents: List[UnifiedMediaDocument]
    reports: List[SourceReport] = field(default_factory=list)


async def run_ingestion(
    company: str,
    top_n: int = 5,
    sources: Optional[Iterable[str | MediaSource]] = None,
    context: Optional[str] = None,
    pre_limit: int = 20,
    min_score: float = 0.5,
    rerank: bool = True,
    light_models: bool = False,
    download: bool = True,
    media_dir: Optional[str] = None,
    per_source_timeout: float = 60.0,
) -> IngestionResult:
    found = await FederatedSearchOrchestrator(per_source_timeout=per_source_timeout).gather_all_sources(
        company, pre_limit=pre_limit, sources=sources
    )
    docs = found.documents
    if rerank:
        docs = get_reranker(light_models).rerank(
            company, docs, top_n, query_context=context, min_score=min_score
        )
    else:
        docs = docs[:top_n]
    if download and docs:
        downloader = MediaDownloader(base_dir=media_dir) if media_dir else MediaDownloader()
        docs = await downloader.download_all(docs)
    return IngestionResult(documents=docs, reports=found.reports)


def run_ingestion_sync(company: str, **kwargs) -> IngestionResult:
    """Versión síncrona para Streamlit o scripts (no usar dentro de un bucle asyncio)."""
    return asyncio.run(run_ingestion(company, **kwargs))


def purge(docs: List[UnifiedMediaDocument], media_dir: Optional[str] = None) -> None:
    """Borra del disco la multimedia descargada (llamar tras analizar)."""
    downloader = MediaDownloader(base_dir=media_dir) if media_dir else MediaDownloader()
    downloader.purge_all(docs)