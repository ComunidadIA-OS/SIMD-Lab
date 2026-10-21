"""Conector de GDELT DOC 2.0 API (modo artlist).

Se usa solo DOC 2.0: el Visual Global Knowledge Graph (VGKG) se consulta vía
BigQuery y queda fuera del alcance. GDELT no devuelve el cuerpo del artículo,
solo título, URL, idioma, fecha e imagen social.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from src.common.schemas import MediaSource, UnifiedMediaDocument
from src.connectors.base import BaseMediaConnector

logger = logging.getLogger(__name__)

API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


class GDELTConnector(BaseMediaConnector):
    source = MediaSource.GDELT
    max_concurrency = 1
    min_interval = 5.0  # GDELT pide no superar ~1 petición cada 5 s

    @staticmethod
    def _iso(seendate: Optional[str]) -> Optional[str]:
        if not seendate:
            return None
        try:
            return datetime.strptime(seendate, "%Y%m%dT%H%M%SZ").isoformat() + "Z"
        except ValueError:
            return seendate

    def parse_results(self, data: dict, query: str) -> List[UnifiedMediaDocument]:
        docs = []
        for art in (data or {}).get("articles", []):
            url = art.get("url", "")
            img = art.get("socialimage")
            title = art.get("title", "")
            docs.append(UnifiedMediaDocument(
                doc_id=self.make_doc_id(self.source, url),
                source=self.source,
                company_query=query,
                url=url,
                title=title,
                published_date=self._iso(art.get("seendate")),
                language=art.get("language"),
                author=art.get("domain"),
                raw_text=title or None,  # GDELT no da cuerpo; el texto completo exigiría scraping
                image_urls=[img] if img else [],
                extra={"domain": art.get("domain", ""), "sourcecountry": art.get("sourcecountry", "")},
            ))
        return docs

    async def query_doc(self, query: str, timespan: str, limit: int) -> List[UnifiedMediaDocument]:
        params = {
            "query": f'"{query}"' if " " in query else query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": min(limit, 250),
            "timespan": timespan,
            "sort": "hybridrel",
        }
        data = await self.get_json(API_URL, params)
        return self.parse_results(data, query)[:limit]

    async def fetch_top_candidates(self, query: str, limit: int) -> List[UnifiedMediaDocument]:
        return await self.query_doc(query, "3months", limit)
