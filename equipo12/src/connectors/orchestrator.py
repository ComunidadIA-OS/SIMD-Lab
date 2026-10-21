"""Orquestador de búsqueda federada: lanza los conectores en paralelo."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import aiohttp

from src.common.schemas import MediaSource, UnifiedMediaDocument
from src.connectors.base import DEFAULT_USER_AGENT, BaseMediaConnector
from src.connectors.gdelt_client import GDELTConnector
from src.connectors.guardian_client import GuardianConnector
from src.connectors.internet_archive_client import InternetArchiveConnector
from src.connectors.wikimedia_client import WikimediaConnector
from src.connectors.youtube_client import YouTubeConnector

logger = logging.getLogger(__name__)

CONNECTOR_CLASSES = {
    MediaSource.WIKIMEDIA: WikimediaConnector,
    MediaSource.GUARDIAN: GuardianConnector,
    MediaSource.GDELT: GDELTConnector,
    MediaSource.YOUTUBE: YouTubeConnector,
    MediaSource.ARCHIVE: InternetArchiveConnector,
}


@dataclass
class SourceReport:
    source: str
    ok: bool
    n_docs: int = 0
    seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class FederatedResult:
    documents: List[UnifiedMediaDocument]
    reports: List[SourceReport] = field(default_factory=list)


class FederatedSearchOrchestrator:
    """Consulta varias fuentes a la vez. Si una falla o tarda, las demás siguen."""

    def __init__(self, per_source_timeout: float = 25.0) -> None:
        self.per_source_timeout = per_source_timeout

    async def _run_one(self, conn: BaseMediaConnector, query: str, limit: int):
        start = time.perf_counter()
        try:
            docs = await asyncio.wait_for(conn.fetch_top_candidates(query, limit), self.per_source_timeout)
            return docs, SourceReport(conn.source.value, True, len(docs), time.perf_counter() - start)
        except asyncio.TimeoutError:
            msg = f"timeout ({self.per_source_timeout}s)"
        except Exception as exc:  # noqa: BLE001 - registramos y seguimos con el resto
            msg = f"{type(exc).__name__}: {exc}"
        logger.warning("Fuente %s fallida: %s", conn.source.value, msg)
        return [], SourceReport(conn.source.value, False, 0, time.perf_counter() - start, msg)

    async def gather_all_sources(
        self,
        query: str,
        pre_limit: int = 20,
        sources: Optional[Iterable[str | MediaSource]] = None,
    ) -> FederatedResult:
        wanted = [MediaSource(s) for s in sources] if sources else list(CONNECTOR_CLASSES)
        async with aiohttp.ClientSession(headers={"User-Agent": DEFAULT_USER_AGENT}) as session:
            connectors = [CONNECTOR_CLASSES[s](session=session) for s in wanted]
            outcomes = await asyncio.gather(*(self._run_one(c, query, pre_limit) for c in connectors))

        seen: Dict[str, UnifiedMediaDocument] = {}
        for docs, _ in outcomes:
            for d in docs:
                seen.setdefault(d.doc_id, d)
        return FederatedResult(documents=list(seen.values()), reports=[r for _, r in outcomes])
