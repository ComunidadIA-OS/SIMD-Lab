from typing import List

from src.analyzers.multimodal_pipeline import MultimodalAnalyzerPipeline, load_payloads
from src.common.schemas import ExtractedFeaturesPayload
from tests.test_xai_fairness_mock import generate_mock_dataset

# "live": busca la empresa escrita, la analiza y borra la multimedia (tarda unos minutos)
# "json": lee un análisis ya guardado (siempre el mismo)
# "mock": datos inventados
DATA_SOURCE = "live"
ANALYZED_DATA_PATH = "tests/mocks/real_extracted_features.json"

_pipeline = None


def _get_pipeline() -> MultimodalAnalyzerPipeline:
    """Carga los analizadores una sola vez (spaCy, CLIP y audio tardan en cargar)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = MultimodalAnalyzerPipeline()
    return _pipeline


def _quitar_vacios(docs: List[ExtractedFeaturesPayload]) -> List[ExtractedFeaturesPayload]:
    """Descarta lo que no aporta información de género.

    - Un texto sin ninguna frase sobre una mujer o un hombre no cuenta como texto.
    - Un documento sin ninguna modalidad con datos se quita entero, para que no baje la nota.
    """
    utiles = []
    for d in docs:
        tm = d.text_metrics
        if tm is not None and tm.sentence_female_ratio + tm.sentence_male_ratio == 0:
            d.text_metrics = None  # sin personas en el texto: la nota de texto no cuenta
        if any(m is not None for m in (d.text_metrics, d.audio_metrics, d.image_metrics, d.video_metrics)):
            utiles.append(d)
    return utiles


def fetch_analyzed_documents(company: str, top_n: int, selected_sources: List[str]) -> List[ExtractedFeaturesPayload]:
    """Devuelve los documentos analizados filtrados por fuente y límite Top-N."""
    if DATA_SOURCE == "live":
        from src.connectors.pipeline import run_ingestion_sync
        result = run_ingestion_sync(company, top_n=top_n, sources=selected_sources)
        if not result.documents:
            raise RuntimeError(f"Ninguna fuente devolvió documentos para '{company}'")
        docs = _quitar_vacios(_get_pipeline().analyze_batch(result.documents))
    else:
        if DATA_SOURCE == "mock":
            all_docs = generate_mock_dataset(company)
        else:
            all_docs = load_payloads(ANALYZED_DATA_PATH)
        filtered = [d for d in all_docs if d.source.value in selected_sources]
        docs = _quitar_vacios((filtered or all_docs)[:top_n])

    if not docs:
        raise RuntimeError(f"Ningún documento de '{company}' menciona o muestra a personas: no hay nada que medir")
    return docs