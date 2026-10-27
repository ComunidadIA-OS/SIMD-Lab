"""Pruebas sin red de la parte de ingesta.

Ejecutar desde la raíz del proyecto:  PYTHONPATH=. pytest tests/test_ingesta.py -q
"""
import asyncio

from src.common.schemas import MediaSource, UnifiedMediaDocument
from src.connectors.gdelt_client import GDELTConnector
from src.connectors.guardian_client import GuardianConnector
from src.connectors.internet_archive_client import InternetArchiveConnector
from src.connectors.orchestrator import FederatedSearchOrchestrator
from src.connectors.youtube_client import YouTubeConnector, iso_duration_to_seconds
from src.reranker.semantic_ranker import SemanticReranker


def _doc(i, source, score):
    return UnifiedMediaDocument(doc_id=f"d{i}", source=source, company_query="X",
                                url=f"u{i}", title=f"t{i}", relevance_score=score)


def test_guardian_parse():
    data = {"response": {"results": [{
        "id": "business/1", "webTitle": "W", "webUrl": "https://g/1",
        "webPublicationDate": "2026-01-01T00:00:00Z", "sectionId": "business",
        "fields": {"headline": "H", "byline": "Ana", "bodyText": "Cuerpo", "thumbnail": "https://img"}}]}}
    d = GuardianConnector(api_key="x").parse_results(data, "Iberdrola")[0]
    assert d.title == "H" and d.author == "Ana" and d.raw_text == "Cuerpo"
    assert d.image_urls == ["https://img"] and d.source == MediaSource.GUARDIAN


def test_gdelt_parse_date():
    data = {"articles": [{"url": "https://n/1", "title": "T", "seendate": "20260115T103000Z",
                          "socialimage": "https://i", "domain": "n", "language": "Spanish"}]}
    d = GDELTConnector().parse_results(data, "Iberdrola")[0]
    assert d.published_date == "2026-01-15T10:30:00Z" and d.image_urls == ["https://i"]


def test_youtube_parse_and_duration_filter():
    assert iso_duration_to_seconds("PT1H2M3S") == 3723
    search = {"items": [
        {"id": {"videoId": "a"}, "snippet": {"title": "corto", "thumbnails": {"high": {"url": "t"}}}},
        {"id": {"videoId": "b"}, "snippet": {"title": "largo"}}]}
    details = {"items": [
        {"id": "a", "contentDetails": {"duration": "PT4M"}, "status": {"license": "creativeCommon"}},
        {"id": "b", "contentDetails": {"duration": "PT2H"}, "status": {"license": "youtube"}}]}
    docs = YouTubeConnector(api_key="x").parse(search, details, "X", max_duration=1200)
    assert [d.title for d in docs] == ["corto"]
    assert docs[0].video_paths == [] and docs[0].audio_urls == []  # nunca se descarga YouTube


def test_internet_archive_pick_lightest_video():
    files = [{"name": "a.HD.mov", "format": "h.264 HD", "size": "900"},
             {"name": "a.mp4", "format": "h.264", "size": "300"},
             {"name": "a.es.vtt", "format": "Web Video Text Tracks"},
             {"name": "__ia_thumb.jpg", "format": "Item Tile"}]
    out = InternetArchiveConnector().pick_files("id1", files)
    assert out["video"] == ["https://archive.org/download/id1/a.mp4"]
    assert out["subs"] and out["thumb"]


def test_select_top_n_balanced():
    docs = [_doc(1, MediaSource.GUARDIAN, .9), _doc(2, MediaSource.GUARDIAN, .8),
            _doc(3, MediaSource.GUARDIAN, .7), _doc(4, MediaSource.ARCHIVE, .2)]
    top = SemanticReranker.select_top_n(docs, 3, balanced=True)
    assert {d.source for d in top} == {MediaSource.GUARDIAN, MediaSource.ARCHIVE}
    top_unbalanced = SemanticReranker.select_top_n(docs, 3, balanced=False)
    assert all(d.source == MediaSource.GUARDIAN for d in top_unbalanced)
    # min_score excluye documentos irrelevantes aunque haya cuota por fuente
    top_min = SemanticReranker.select_top_n(docs, 3, balanced=True, min_score=0.5)
    assert all(d.source == MediaSource.GUARDIAN for d in top_min)


def test_orchestrator_survives_failing_source():
    class Ok:
        source = MediaSource.WIKIMEDIA
        async def fetch_top_candidates(self, q, n):
            return [_doc(1, self.source, None)]

    class Boom:
        source = MediaSource.GDELT
        async def fetch_top_candidates(self, q, n):
            raise RuntimeError("caída")

    orch = FederatedSearchOrchestrator(per_source_timeout=1)
    async def run():
        return await asyncio.gather(orch._run_one(Ok(), "X", 5), orch._run_one(Boom(), "X", 5))
    (docs_ok, rep_ok), (docs_ko, rep_ko) = asyncio.run(run())
    assert len(docs_ok) == 1 and rep_ok.ok
    assert docs_ko == [] and not rep_ko.ok and "caída" in rep_ko.error



# ---------- pruebas añadidas ----------
import pytest
from aiohttp import web

from src.common.schemas import normalize_language
from src.connectors.media_downloader import MediaDownloader
from src.connectors.wikimedia_client import WikimediaConnector


@pytest.mark.parametrize("raw, expected", [
    ("Spanish", "es"), ("spa", "es"), ("es-ES", "es"), ("es", "es"),
    ("English", "en"), ("eng", "en"), ("", None), (None, None), ("klingon", None),
])
def test_normalize_language(raw, expected):
    assert normalize_language(raw) == expected


def test_language_normalized_in_document():
    d = UnifiedMediaDocument(doc_id="x", source=MediaSource.GDELT, company_query="X",
                             url="u", title="t", language="Spanish")
    assert d.language == "es"


def test_downloader_rejects_svg():
    assert not MediaDownloader.is_supported_image_url("https://upload.wikimedia.org/logo.svg")
    assert not MediaDownloader.is_supported_image_url("https://x/anim.gif?w=200")
    assert MediaDownloader.is_supported_image_url("https://x/foto.JPG")
    assert MediaDownloader.is_supported_image_url("https://x/thumbnail")  # sin extensión: se decide por Content-Type


def test_wikimedia_dedupes_same_entity_across_languages():
    conn = WikimediaConnector()

    async def fake_search(lang, query, limit):
        return [1]

    async def fake_page(lang, pageid, query):
        return UnifiedMediaDocument(doc_id=lang, source=MediaSource.WIKIMEDIA, company_query=query,
                                    url=f"https://{lang}.wikipedia.org/wiki/Iberdrola", title="Iberdrola",
                                    language=lang, extra={"wikidata": "Q1"})

    conn._search, conn._page = fake_search, fake_page
    docs = asyncio.run(conn.search_articles_and_commons("Iberdrola", 10))
    assert len(docs) == 1 and docs[0].language == "es"


def _run_with_server(handler, coro_factory):
    """Levanta un servidor HTTP local, ejecuta la prueba y lo apaga."""
    async def run():
        app = web.Application()
        app.router.add_get("/", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            return await coro_factory(f"http://127.0.0.1:{port}/")
        finally:
            await runner.cleanup()
    return asyncio.run(run())


def test_get_json_retries_after_429():
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return web.Response(status=429)
        return web.json_response({"ok": True})

    async def check(url):
        async with GuardianConnector(api_key="x") as conn:
            return await conn.get_json(url)

    assert _run_with_server(handler, check) == {"ok": True}
    assert calls["n"] == 2


def test_get_json_non_json_raises_value_error():
    async def handler(request):
        return web.Response(text="<html>error</html>")

    async def check(url):
        async with GuardianConnector(api_key="x") as conn:
            with pytest.raises(ValueError):
                await conn.get_json(url)

    _run_with_server(handler, check)
