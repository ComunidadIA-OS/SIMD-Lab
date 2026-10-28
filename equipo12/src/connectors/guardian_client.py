"""Conector de The Guardian Open Platform (endpoint /search)."""
from __future__ import annotations

import logging
import os
from typing import List, Optional

from src.common.schemas import MediaSource, UnifiedMediaDocument
from src.connectors.base import BaseMediaConnector

logger = logging.getLogger(__name__)

API_URL = "https://content.guardianapis.com/search"


class GuardianConnector(BaseMediaConnector):
    source = MediaSource.GUARDIAN
    max_concurrency = 2

    def __init__(self, api_key: Optional[str] = None, **kw) -> None:
        super().__init__(**kw)
        self.api_key = api_key or os.getenv("GUARDIAN_API_KEY")

    def parse_results(self, data: dict, query: str) -> List[UnifiedMediaDocument]:
        docs = []
        for item in data.get("response", {}).get("results", []):
            fields = item.get("fields", {}) or {}
            url = item.get("webUrl", "")
            thumb = fields.get("thumbnail")
            main_url, caption = self._main_image(item)
            docs.append(UnifiedMediaDocument(
                doc_id=self.make_doc_id(self.source, url),
                source=self.source,
                company_query=query,
                url=url,
                title=fields.get("headline") or item.get("webTitle", ""),
                published_date=item.get("webPublicationDate"),
                language="en",
                license="Guardian Open Platform (uso no comercial, ver términos)",
                author=fields.get("byline"),
                raw_text="\n".join(t for t in (caption, fields.get("bodyText")) if t) or None,
                image_urls=[main_url or thumb] if (main_url or thumb) else [],
                extra={"section": item.get("sectionId", ""), "guardian_id": item.get("id", "")},
            ))
        return docs

    @staticmethod
    def _main_image(item: dict):
        """Imagen principal en su mayor tamaño (1000 px en vez de la miniatura de 500) y su pie de foto."""
        for el in item.get("elements", []) or []:
            if el.get("relation") == "main" and el.get("type") == "image":
                assets = [a for a in el.get("assets", []) if a.get("file")]
                if not assets:
                    continue
                best = max(assets, key=lambda a: int(a.get("typeData", {}).get("width", 0) or 0))
                return best["file"], best.get("typeData", {}).get("caption")
        return None, None

    async def fetch_news(self, query: str, section: Optional[str], limit: int) -> List[UnifiedMediaDocument]:
        if not self.api_key:
            logger.warning("GUARDIAN_API_KEY no definida: se omite The Guardian.")
            return []
        params = {
            "q": f'"{query}"',
            "page-size": min(limit, 50),
            "order-by": "relevance",
            "show-fields": "headline,byline,bodyText,thumbnail",
            "show-elements": "image",
            "api-key": self.api_key,
        }
        if section:
            params["section"] = section
        data = await self.get_json(API_URL, params)
        return self.parse_results(data, query)[:limit]

    async def fetch_top_candidates(self, query: str, limit: int) -> List[UnifiedMediaDocument]:
        return await self.fetch_news(query, None, limit)
