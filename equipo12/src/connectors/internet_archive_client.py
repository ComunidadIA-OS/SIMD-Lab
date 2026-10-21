"""Conector de Internet Archive (Advanced Search + Metadata API)."""
from __future__ import annotations

import asyncio
import logging
from typing import List, Optional, Sequence

from src.common.schemas import MediaSource, UnifiedMediaDocument
from src.connectors.base import BaseMediaConnector

logger = logging.getLogger(__name__)

SEARCH_URL = "https://archive.org/advancedsearch.php"
META_URL = "https://archive.org/metadata/{identifier}"
DOWNLOAD_URL = "https://archive.org/download/{identifier}/{name}"

VIDEO_FORMATS = ("h.264", "MPEG4", "512Kb MPEG4", "h.264 IA")
AUDIO_FORMATS = ("VBR MP3", "128Kbps MP3", "MP3", "Ogg Vorbis")


class InternetArchiveConnector(BaseMediaConnector):
    source = MediaSource.ARCHIVE
    max_concurrency = 4

    def __init__(self, mediatypes: Sequence[str] = ("movies", "audio"), **kw) -> None:
        super().__init__(**kw)
        self.mediatypes = list(mediatypes)

    @staticmethod
    def _first(value) -> Optional[str]:
        if isinstance(value, list):
            return str(value[0]) if value else None
        return str(value) if value is not None else None

    def pick_files(self, identifier: str, files: list) -> dict:
        """Elige un vídeo, un audio, subtítulos y miniatura de la lista de ficheros."""
        out = {"video": [], "audio": [], "subs": [], "thumb": []}
        for f in files:
            name, fmt = f.get("name", ""), f.get("format", "")
            url = DOWNLOAD_URL.format(identifier=identifier, name=name)
            if fmt in VIDEO_FORMATS or name.endswith(".mp4"):
                out["video"].append((float(f.get("size", 0) or 0), url))
            elif fmt in AUDIO_FORMATS or name.endswith(".mp3"):
                out["audio"].append((float(f.get("size", 0) or 0), url))
            elif name.endswith(".vtt") or name.endswith(".srt"):
                out["subs"].append(url)
            elif fmt == "Item Tile":
                out["thumb"].append(url)
        # Para vídeo/audio nos quedamos con el fichero más ligero (luego solo se lee un tramo)
        for k in ("video", "audio"):
            out[k] = [min(out[k])[1]] if out[k] else []
        return out

    async def _enrich(self, doc_meta: dict, query: str) -> Optional[UnifiedMediaDocument]:
        identifier = doc_meta["identifier"]
        meta = await self.get_json(META_URL.format(identifier=identifier))
        md = meta.get("metadata", {}) or {}
        files = self.pick_files(identifier, meta.get("files", []) or [])
        url = f"https://archive.org/details/{identifier}"
        desc = (self._first(md.get("description")) or "").strip()
        return UnifiedMediaDocument(
            doc_id=self.make_doc_id(self.source, url),
            source=self.source,
            company_query=query,
            url=url,
            title=self._first(md.get("title")) or identifier,
            published_date=self._first(md.get("date")),
            language=self._first(md.get("language")),
            license=self._first(md.get("licenseurl")),
            author=self._first(md.get("creator")),
            raw_text=desc or None,
            image_urls=files["thumb"],
            audio_urls=files["audio"],
            video_urls=files["video"],
            subtitle_urls=files["subs"],
            extra={"identifier": identifier, "mediatype": self._first(md.get("mediatype")) or ""},
        )

    async def fetch_historical_media(self, query: str, mediatype: Sequence[str], limit: int) -> List[UnifiedMediaDocument]:
        mt = " OR ".join(mediatype)
        params = [
            ("q", f'"{query}" AND mediatype:({mt})'),
            ("fl[]", "identifier"), ("fl[]", "title"), ("fl[]", "mediatype"),
            ("rows", str(limit)), ("output", "json"),
        ]
        data = await self.get_json(SEARCH_URL, params)
        hits = data.get("response", {}).get("docs", [])
        results = await asyncio.gather(*(self._enrich(h, query) for h in hits), return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Internet Archive: metadatos fallidos: %s", r)
        return [r for r in results if isinstance(r, UnifiedMediaDocument)][:limit]

    async def fetch_top_candidates(self, query: str, limit: int) -> List[UnifiedMediaDocument]:
        return await self.fetch_historical_media(query, self.mediatypes, limit)
