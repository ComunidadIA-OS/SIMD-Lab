"""Prueba del pipeline completo con documentos sintéticos (texto, foto, audio y vídeo).

    PYTHONPATH=. python tests/test_pipeline_mock.py
"""
import logging
import math
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from skimage import data

load_dotenv()

from src.analyzers.multimodal_pipeline import MultimodalAnalyzerPipeline, save_payloads
from src.common.schemas import MediaSource, UnifiedMediaDocument
from src.explainability.scoring_model import CorporateGenderScorer

MEDIA_DIR = Path(os.getenv("MEDIA_DIR", "/tmp/ramdisk"))
EMPRESA = "EcoRed"


def carpeta(doc_id):
    d = MEDIA_DIR / doc_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def cara(nombre):
    """'astronaut' (una mujer) o 'camera' (un hombre), imágenes incluidas en scikit-image."""
    img = getattr(data, nombre)()
    return np.stack([img] * 3, axis=-1) if img.ndim == 2 else img


def foto_grupo(ruta):
    """Una mujer a la izquierda y un hombre a la derecha."""
    Image.fromarray(np.concatenate([cara("astronaut"), cara("camera")], axis=1)).save(ruta)


def video(ruta, segundos=15, fps=10):
    """8 s de la mujer moviéndose y luego 7 s del hombre casi quieto."""
    mujer, hombre = (cv2.cvtColor(cara(n), cv2.COLOR_RGB2BGR) for n in ("astronaut", "camera"))
    salida = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"MJPG"), fps, (512, 512))
    for n in range(segundos * fps):
        t = n / fps
        if t < 8:
            salida.write(np.roll(mujer, int(12 * math.sin(2 * math.pi * t)), axis=1))
        else:
            salida.write(np.roll(hombre, int(3 * math.sin(2 * math.pi * t)), axis=1))
    salida.release()


def documentos():
    docs = []

    # 1) Noticia con mujeres protagonistas y una foto de grupo 
    d = carpeta("mock_noticia_directora")
    foto_grupo(d / "image_0.jpg")
    docs.append(UnifiedMediaDocument(
        doc_id="mock_noticia_directora", source=MediaSource.GUARDIAN, company_query=EMPRESA,
        url="https://example.org/noticia-directora", language="es",
        title="Laura Martín presenta el plan de inversión de EcoRed",
        raw_text=("La directora general de EcoRed, Laura Martín, presentó ayer el nuevo plan de inversión. "
                  "«Vamos a duplicar la capacidad renovable», explicó Martín. "
                  "La ingeniera jefa, Carmen Ruiz, detalló el calendario de obras. "
                  "El consejero Pedro Gómez asistió a la presentación."),
        image_urls=["https://example.org/foto-grupo.jpg"], image_paths=[str(d / "image_0.jpg")]))

    # 2) Entrevista en vídeo
    d = carpeta("mock_entrevista_video")
    shutil.copy("tests/mocks/lamartine.wav", d / "audio_0.wav")
    video(d / "video_0.avi")
    docs.append(UnifiedMediaDocument(
        doc_id="mock_entrevista_video", source=MediaSource.ARCHIVE, company_query=EMPRESA,
        url="https://example.org/entrevista", language="es",
        title="Entrevista sobre redes eléctricas",
        raw_text=("Entrevista a la investigadora Ana Torres sobre redes eléctricas. "
                  "La investigadora explicó los retos del almacenamiento."),
        video_urls=["https://example.org/entrevista.mp4"],
        audio_paths=[str(d / "audio_0.wav")], video_paths=[str(d / "video_0.avi")]))

    # 3) Noticia con hombres protagonistas, solo texto
    docs.append(UnifiedMediaDocument(
        doc_id="mock_noticia_presidente", source=MediaSource.GDELT, company_query=EMPRESA,
        url="https://example.org/noticia-presidente", language="es",
        title="El presidente de EcoRed presenta los resultados",
        raw_text=("El presidente de EcoRed, Javier López, anunció los resultados del trimestre. "
                  "«Ha sido un año excelente», afirmó López. "
                  "Según el director financiero, Miguel Sanz, el beneficio creció un 12 %. "
                  "La empresa también agradeció el trabajo de su secretaria.")))
    return docs


def resumen(m) -> str:
    datos = m.model_dump(exclude={"audio_track", "frame_set_visuals"})
    return ", ".join(f"{k}={getattr(v, 'value', v)}" for k, v in datos.items())


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    print("1. Analizar 3 documentos sintéticos ==")
    pipeline = MultimodalAnalyzerPipeline()
    payloads = pipeline.analyze_batch(documentos())
    for p in payloads:
        print(f"\n{p.title}\n  modalidades={p.available_modalities}  subgrupo={p.demographic_subgroup}")
        for nombre, m in [("Texto", p.text_metrics), ("Audio", p.audio_metrics),
                          ("Imagen", p.image_metrics), ("Vídeo", p.video_metrics)]:
            if m is not None:
                print(f"  {nombre}: {resumen(m)}")

    print("\n2. Borrado efímero ==")
    for r in pipeline.purge_receipts:
        print(f"  {r['doc_id']}: {r['ficheros_borrados']} ficheros borrados")
    print("  tests/mocks/lamartine.wav sigue ahí:", Path("tests/mocks/lamartine.wav").exists())

    save_payloads(payloads, "tests/mocks/mock_pipeline_features.json")
    print("\nGuardado tests/mocks/mock_pipeline_features.json")

    print("\n3. Puntuación con el modelo del dashboard ==")
    resultado = CorporateGenderScorer().predict_importance_score(payloads)
    tabla = resultado["dataframe"][["title", "score_texto", "score_audio", "score_imagen", "score_video",
                                    "importance_score"]]
    print(tabla.to_string(index=False))
    print("\n  Puntuación global:", resultado["global_score"])
