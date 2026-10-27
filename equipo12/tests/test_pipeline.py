"""Prueba de extremo a extremo: ingesta -> descarga -> análisis -> metrica.
    PYTHONPATH=. python tests/test_pipeline.py
"""
import asyncio
import json
import logging

from dotenv import load_dotenv
load_dotenv()

from src.analyzers.multimodal_pipeline import MultimodalAnalyzerPipeline, save_payloads
from src.common.schemas import UnifiedMediaDocument
from src.connectors.media_downloader import MediaDownloader
from src.explainability.scoring_model import CorporateGenderScorer

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

with open("tests/mocks/sample_top_n_docs.json", encoding="utf-8") as f:
    docs = [UnifiedMediaDocument.model_validate(d) for d in json.load(f)]

print("1. Descargar la multimedia ==")
docs = asyncio.run(MediaDownloader().download_all(docs))
for d in docs:
    print(f"  {d.title[:50]:<50} imágenes={len(d.image_paths)} audios={len(d.audio_paths)} vídeos={len(d.video_paths)}")

print("\n2. Analizar ==")
pipeline = MultimodalAnalyzerPipeline()
payloads = pipeline.analyze_batch(docs)
for p in payloads:
    print(f"  {p.title[:50]:<50} {p.available_modalities}  subgrupo={p.demographic_subgroup}")

print("\n3. Borrado efímero ==")
for r in pipeline.purge_receipts:
    print(f"  {r['doc_id']}: {r['ficheros_borrados']} ficheros borrados")

save_payloads(payloads, "tests/mocks/real_extracted_features.json")
print("\nGuardado tests/mocks/real_extracted_features.json")

print("\n4. Puntuación con el modelo del dashboard ==")
resultado = CorporateGenderScorer().predict_importance_score(payloads)
print("  Puntuación global:", resultado["global_score"])
print("  Por modalidad:", resultado["by_modality"])
