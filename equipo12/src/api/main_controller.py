import logging
import os
import time
from typing import List

# GPU compartida entre TensorFlow (audio) y PyTorch (reranker y CLIP):
# por defecto TensorFlow reserva toda la memoria de la GPU y PyTorch se queda sin sitio.
# Tiene que definirse antes de que se importe TensorFlow.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from src.analyzers.multimodal_pipeline import MultimodalAnalyzerPipeline, load_payloads
from src.common.logging_setup import setup_logging
from src.common.schemas import ExtractedFeaturesPayload
from tests.test_xai_fairness_mock import generate_mock_dataset

setup_logging()
logger = logging.getLogger("auditoria")

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
        else:
            logger.info("   descartado (sin personas): %s", d.title[:60])
    return utiles


def _analizar(docs) -> List[ExtractedFeaturesPayload]:
    """Analiza documento a documento para poder ver el progreso. Si uno falla, sigue con el resto."""
    t0 = time.perf_counter()
    if _pipeline is None:
        logger.info("Cargando los analizadores (spaCy, CLIP, audio): la primera vez tarda...")
    pipe = _get_pipeline()
    payloads = []
    for i, d in enumerate(docs, 1):
        t = time.perf_counter()
        logger.info("Analizando %d/%d [%s] %s | img=%d audio=%d vídeo=%d",
                    i, len(docs), d.source.value, d.title[:60],
                    len(d.image_paths), len(d.audio_paths), len(d.video_paths))
        try:
            p = pipe.analyze(d)
            payloads.append(p)
            logger.info("   -> %s en %.1f s", p.available_modalities, time.perf_counter() - t)
        except Exception as exc:  # noqa: BLE001
            logger.warning("   -> fallo, se omite: %s: %s", type(exc).__name__, exc)
    logger.info("Análisis terminado: %d de %d documentos en %.1f s",
                len(payloads), len(docs), time.perf_counter() - t0)
    return payloads


def fetch_analyzed_documents(company: str, top_n: int, selected_sources: List[str]) -> List[ExtractedFeaturesPayload]:
    """Devuelve los documentos analizados filtrados por fuente y límite Top-N."""
    t0 = time.perf_counter()
    logger.info("===== Auditoría de '%s' | modo=%s | Top-%d | fuentes=%s =====",
                company, DATA_SOURCE, top_n, ", ".join(selected_sources))
    if DATA_SOURCE == "live":
        from src.connectors.pipeline import run_ingestion_sync
        result = run_ingestion_sync(company, top_n=top_n, sources=selected_sources)
        if not result.documents:
            raise RuntimeError(f"Ninguna fuente devolvió documentos para '{company}'")
        docs = _quitar_vacios(_analizar(result.documents))
    else:
        if DATA_SOURCE == "mock":
            all_docs = generate_mock_dataset(company)
        else:
            all_docs = load_payloads(ANALYZED_DATA_PATH)
        filtered = [d for d in all_docs if d.source.value in selected_sources]
        docs = _quitar_vacios((filtered or all_docs)[:top_n])

    logger.info("===== Fin: %d documentos útiles en %.1f s =====", len(docs), time.perf_counter() - t0)
    if not docs:
        raise RuntimeError(f"Ningún documento de '{company}' menciona o muestra a personas: no hay nada que medir")
    return docs
