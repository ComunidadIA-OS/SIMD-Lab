"""Pruebas sin red de la parte de ingesta (Integrante 1).

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
