"""Conector de Wikipedia (MediaWiki API) + imagen principal de Wikimedia Commons."""
from __future__ import annotations

import asyncio
import logging
from typing import List, Sequence

from src.common.schemas import MediaSource, UnifiedMediaDocument
from src.connectors.base import BaseMediaConnector

logger = logging.getLogger(__name__)


class WikimediaConnector(BaseMediaConnector):
    source = MediaSource.WIKIMEDIA
    max_concurrency = 3

    def __init__(self, languages: Sequence[str] = ("es", "en"), **kw) -> None:
        super().__init__(**kw)
        self.languages = list(languages)

    def _api(self, lang: str) -> str:
        return f"https://{lang}.wikipedia.org/w/api.php"

    async def _search(self, lang: str, query: str, limit: int) -> List[int]:
        data = await self.get_json(
            self._api(lang),
            {"action": "query", "list": "search", "srsearch": query,
             "srlimit": limit, "format": "json", "formatversion": 2},
        )
        return [hit["pageid"] for hit in data.get("query", {}).get("search", [])]

    async def _page(self, lang: str, pageid: int, query: str) -> UnifiedMediaDocument | None:
        # TextExtracts solo devuelve texto completo de una página por petición
        data = await self.get_json(
            self._api(lang),
            {"action": "query", "pageids": pageid,
             "prop": "extracts|pageimages|info|revisions|pageprops",
             "ppprop": "wikibase_item",
             "explaintext": 1, "piprop": "original", "inprop": "url",
             "rvprop": "timestamp", "format": "json", "formatversion": 2},
        )
        pages = data.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            return None
        p = pages[0]
        url = p.get("fullurl", f"https://{lang}.wikipedia.org/?curid={pageid}")
        image = p.get("original", {}).get("source")
        revs = p.get("revisions") or [{}]
        return UnifiedMediaDocument(
            doc_id=self.make_doc_id(self.source, url),
            source=self.source,
            company_query=query,
            url=url,
            title=p.get("title", ""),
            published_date=revs[0].get("timestamp"),
            language=lang,
            license="CC BY-SA 4.0 (texto de Wikipedia)",
            raw_text=p.get("extract") or None,
            image_urls=[image] if image else [],
            extra={"pageid": str(pageid),
                   "wikidata": (p.get("pageprops") or {}).get("wikibase_item", "")},
        )

    async def search_articles_and_commons(self, query: str, limit: int) -> List[UnifiedMediaDocument]:
        per_lang = max(1, limit // len(self.languages)) if limit >= len(self.languages) else 1
        ids_by_lang = await asyncio.gather(
            *(self._search(lang, query, per_lang) for lang in self.languages),
            return_exceptions=True,
        )
        tasks = []
        for lang, ids in zip(self.languages, ids_by_lang):
            if isinstance(ids, Exception):
                logger.warning("Wikipedia %s: búsqueda fallida: %s", lang, ids)
                continue
            tasks += [self._page(lang, pid, query) for pid in ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Filtrar los resultados que son documentos de medios unificados
        docs = [r for r in results if isinstance(r, UnifiedMediaDocument)]
        # La misma entidad en es.wikipedia y en.wikipedia comparte id de Wikidata:
        # nos quedamos con la primera (el orden de self.languages da prioridad al español)
        seen, unique = set(), []
        for d in docs:
            key = d.extra.get("wikidata") or d.url
            if key not in seen:
                seen.add(key)
                unique.append(d)
        return unique[:limit]

    async def fetch_top_candidates(self, query: str, limit: int) -> List[UnifiedMediaDocument]:
        return await self.search_articles_and_commons(query, limit)
