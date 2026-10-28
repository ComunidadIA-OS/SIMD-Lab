"""Conector de Flickr (fotos con licencia abierta).

Muchas empresas e instituciones publican en Flickr fotos de sus eventos (juntas,
ruedas de prensa, premios), que es donde más aparecen personas. Solo se piden
fotos con licencia Creative Commons o de dominio público, y en el tamaño grande
(1024 px) para que las caras se detecten bien.

Requiere FLICKR_API_KEY (https://www.flickr.com/services/apps/create/).
"""
from __future__ import annotations

import html
import logging
import os
from datetime import datetime
from typing import List, Optional

from src.common.schemas import MediaSource, UnifiedMediaDocument
from src.connectors.base import BaseMediaConnector

logger = logging.getLogger(__name__)

API_URL = "https://www.flickr.com/services/rest/"

# Licencias de Flickr (flickr.photos.licenses.getInfo). Se excluye la 0, "todos los derechos reservados".
LICENSES = {
    "1": "CC BY-NC-SA 2.0", "2": "CC BY-NC 2.0", "3": "CC BY-NC-ND 2.0", "4": "CC BY 2.0",
    "5": "CC BY-SA 2.0", "6": "CC BY-ND 2.0", "7": "Sin restricciones de copyright conocidas",
    "8": "Obra del Gobierno de EE. UU.", "9": "CC0 1.0", "10": "Dominio público",
}
# De mayor a menor: 1024, 800 y 640 px
SIZES = ("url_l", "url_c", "url_z")


class FlickrConnector(BaseMediaConnector):
    source = MediaSource.FLICKR
    max_concurrency = 2

    def __init__(self, api_key: Optional[str] = None, **kw) -> None:
        super().__init__(**kw)
        self.api_key = api_key or os.getenv("FLICKR_API_KEY")

    @staticmethod
    def _date(value: Optional[str]) -> Optional[str]:
        try:
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").isoformat() if value else None
        except ValueError:
            return value

    def parse_results(self, data: dict, query: str) -> List[UnifiedMediaDocument]:
        if (data or {}).get("stat") == "fail":
            raise RuntimeError(f"Flickr: {data.get('message', 'error desconocido')}")
        docs = []
        for p in (data or {}).get("photos", {}).get("photo", []) or []:
            image = next((p[k] for k in SIZES if p.get(k)), None)
            if not image:
                continue
            url = f"https://www.flickr.com/photos/{p.get('owner')}/{p.get('id')}"
            descripcion = html.unescape((p.get("description") or {}).get("_content", "")).strip()
            titulo = html.unescape(p.get("title") or "").strip()
            texto = "\n".join(t for t in (titulo, descripcion, p.get("tags", "")) if t)
            docs.append(UnifiedMediaDocument(
                doc_id=self.make_doc_id(self.source, url),
                source=self.source,
                company_query=query,
                url=url,
                title=titulo or f"Foto de {p.get('ownername', 'Flickr')}",
                published_date=self._date(p.get("datetaken")),
                license=LICENSES.get(str(p.get("license")), f"Licencia Flickr {p.get('license')}"),
                author=p.get("ownername"),
                raw_text=texto or None,
                image_urls=[image],
                extra={"photo_id": str(p.get("id", "")), "owner": str(p.get("owner", ""))},
            ))
        return docs

    async def search_photos(self, query: str, limit: int) -> List[UnifiedMediaDocument]:
        if not self.api_key:
            logger.warning("FLICKR_API_KEY no definida: se omite Flickr.")
            return []
        data = await self.get_json(API_URL, {
            "method": "flickr.photos.search",
            "api_key": self.api_key,
            "text": query,
            "license": ",".join(LICENSES),
            "content_types": "0",      # solo fotos (sin capturas ni ilustraciones)
            "media": "photos",
            "sort": "relevance",
            "extras": "description,license,owner_name,date_taken,tags," + ",".join(SIZES),
            "per_page": str(min(limit, 100)),
            "format": "json",
            "nojsoncallback": "1",
        })
        return self.parse_results(data, query)[:limit]

    async def fetch_top_candidates(self, query: str, limit: int) -> List[UnifiedMediaDocument]:
        return await self.search_photos(query, limit)
