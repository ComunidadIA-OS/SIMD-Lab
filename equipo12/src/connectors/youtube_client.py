"""Conector de YouTube Data API v3 (solo metadatos).

Decisión del equipo: NO se descargan vídeos ni audio de YouTube (yt-dlp vulnera
los Términos de Servicio). Se recuperan título, descripción, canal, duración,
licencia y miniatura. Los vídeos con licencia Creative Commons se marcan en
`extra["license"]` por si en el futuro se decide tratarlos de otra forma.
"""
from __future__ import annotations

import logging
import os
import re
from typing import List, Optional
import html

from src.common.schemas import MediaSource, UnifiedMediaDocument
from src.connectors.base import BaseMediaConnector

logger = logging.getLogger(__name__)

SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
_DUR = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")


def iso_duration_to_seconds(value: str) -> int:
    m = _DUR.fullmatch(value or "")
    if not m:
        return 0
    h, mi, s = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + mi * 60 + s


class YouTubeConnector(BaseMediaConnector):
    source = MediaSource.YOUTUBE
    max_concurrency = 2

    def __init__(self, api_key: Optional[str] = None, **kw) -> None:
        super().__init__(**kw)
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY")

    def parse(self, search: dict, details: dict, query: str, max_duration: int) -> List[UnifiedMediaDocument]:
        info = {v["id"]: v for v in (details or {}).get("items", [])}
        docs = []
        for item in (search or {}).get("items", []):
            vid = item.get("id", {}).get("videoId")
            if not vid:
                continue
            sn = item.get("snippet", {})
            det = info.get(vid, {})
            seconds = iso_duration_to_seconds(det.get("contentDetails", {}).get("duration", ""))
            if max_duration and seconds > max_duration:
                continue
            thumbs = sn.get("thumbnails", {})
            thumb = next((thumbs[k]["url"] for k in ("maxres", "standard", "high", "medium", "default")
                          if thumbs.get(k, {}).get("url")), None)
            url = f"https://www.youtube.com/watch?v={vid}"
            lic = det.get("status", {}).get("license", "")
            docs.append(UnifiedMediaDocument(
                doc_id=self.make_doc_id(self.source, url),
                source=self.source,
                company_query=query,
                url=url,
                title=html.unescape(sn.get("title", "")),
                published_date=sn.get("publishedAt"),
                language=det.get("snippet", {}).get("defaultAudioLanguage"),
                license="Creative Commons BY" if lic == "creativeCommon" else "Licencia estándar de YouTube",
                author=sn.get("channelTitle"),
                raw_text=html.unescape(det.get("snippet", {}).get("description") or sn.get("description") or "") or None,
                image_urls=[thumb] if thumb else [],
                extra={"video_id": vid, "duration_s": str(seconds), "license": lic},
            ))
        return docs

    async def extract_multimedia(self, query: str, limit: int, max_duration: int = 1200) -> List[UnifiedMediaDocument]:
        if not self.api_key:
            logger.warning("YOUTUBE_API_KEY no definida: se omite YouTube.")
            return []
        search = await self.get_json(SEARCH_URL, {
            "part": "snippet", "q": query, "type": "video",
            "maxResults": min(limit, 50), "relevanceLanguage": "es", "key": self.api_key,
        })
        ids = [i["id"]["videoId"] for i in search.get("items", []) if i.get("id", {}).get("videoId")]
        details = {}
        if ids:
            details = await self.get_json(VIDEOS_URL, {
                "part": "snippet,contentDetails,status", "id": ",".join(ids), "key": self.api_key,
            })
        return self.parse(search, details, query, max_duration)[:limit]

    async def fetch_top_candidates(self, query: str, limit: int) -> List[UnifiedMediaDocument]:
        return await self.extract_multimedia(query, limit)
