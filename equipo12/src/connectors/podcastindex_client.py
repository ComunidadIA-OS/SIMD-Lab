"""Conector de Podcast Index (episodios de podcast con su audio).

Busca episodios en los que se menciona la empresa (título, descripción, autor o
etiquetas de personas) con /search/byperson. Cada episodio trae `enclosureUrl`,
el MP3 que el propio podcast publica en su RSS: MediaDownloader recorta solo
los primeros minutos a WAV 16 kHz mono para el analizador de audio.

Requiere PODCASTINDEX_API_KEY y PODCASTINDEX_API_SECRET (registro gratuito en
https://api.podcastindex.org).
"""
from __future__ import annotations

import hashlib
import html
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.common.schemas import MediaSource, UnifiedMediaDocument
from src.connectors.base import BaseMediaConnector

logger = logging.getLogger(__name__)

API_URL = "https://api.podcastindex.org/api/1.0/search/byperson"
AUDIO_TYPES = ("audio/",)
_TAGS = re.compile(r"<[^>]+>")


def auth_headers(key: str, secret: str, now: Optional[int] = None) -> Dict[str, str]:
    """Cabeceras de autenticación: sha1(key + secret + fecha unix)."""
    epoch = str(int(now if now is not None else time.time()))
    digest = hashlib.sha1((key + secret + epoch).encode("utf-8")).hexdigest()
    return {"X-Auth-Key": key, "X-Auth-Date": epoch, "Authorization": digest}


def _clean(text: Optional[str]) -> str:
    """Quita etiquetas HTML y espacios sobrantes de la descripción."""
    return re.sub(r"\s+", " ", html.unescape(_TAGS.sub(" ", text or ""))).strip()


class PodcastIndexConnector(BaseMediaConnector):
    source = MediaSource.PODCAST
    max_concurrency = 2

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, **kw) -> None:
        super().__init__(**kw)
        self.api_key = api_key or os.getenv("PODCASTINDEX_API_KEY")
        self.api_secret = api_secret or os.getenv("PODCASTINDEX_API_SECRET")

    def parse_results(self, data: dict, query: str) -> List[UnifiedMediaDocument]:
        docs = []
        for ep in (data or {}).get("items", []) or []:
            audio = ep.get("enclosureUrl")
            if not audio or not str(ep.get("enclosureType", "audio/")).startswith(AUDIO_TYPES):
                continue  # sin audio descargable (o es un vídeo) no aporta a esta fuente
            url = ep.get("link") or audio
            ts = ep.get("datePublished")
            fecha = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat() if ts else None
            descripcion = _clean(ep.get("description"))
            docs.append(UnifiedMediaDocument(
                doc_id=self.make_doc_id(self.source, audio),
                source=self.source,
                company_query=query,
                url=url,
                title=_clean(ep.get("title")),
                published_date=fecha,
                language=ep.get("feedLanguage"),
                license="Publicado por el podcast en su RSS (uso para investigación)",
                author=ep.get("feedTitle") or ep.get("feedAuthor"),
                raw_text=descripcion or None,
                audio_urls=[audio],
                extra={
                    "feed_id": str(ep.get("feedId", "")),
                    "episode_id": str(ep.get("id", "")),
                    "duration_s": str(ep.get("duration") or ""),
                },
            ))
        return docs

    async def search_episodes(self, query: str, limit: int) -> List[UnifiedMediaDocument]:
        if not (self.api_key and self.api_secret):
            logger.warning("PODCASTINDEX_API_KEY/SECRET no definidas: se omite Podcast Index.")
            return []
        data = await self.get_json(
            API_URL,
            {"q": query, "max": min(limit * 2, 100), "fulltext": ""},
            headers=auth_headers(self.api_key, self.api_secret),
        )
        return self.parse_results(data, query)[:limit]

    async def fetch_top_candidates(self, query: str, limit: int) -> List[UnifiedMediaDocument]:
        return await self.search_episodes(query, limit)
