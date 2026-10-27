"""Pipeline multimodal
Para cada documento:
  1. Texto                     -> TextMetrics  
  2. Cada audio                -> AudioMetrics
  3. Cada vídeo, con su audio  -> VideoMetrics
  4. Las fotos                 -> ImageMetrics
  5. Borra la multimedia descargada
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import List

import numpy as np

from src.analyzers.privacy_scrubber import EphemeralPrivacyScrubber
from src.common.schemas import AudioSample, ExtractedFeaturesPayload, ImageSample, VideoSample

logger = logging.getLogger(__name__)   


class MultimodalAnalyzerPipeline:
    """Junta los tres analizadores (texto, audio, imagen/vídeo) y el borrado de la multimedia."""

    def __init__(self, purge: bool = True):
        self.purge = purge
        self.scrubber = EphemeralPrivacyScrubber()
        self.purge_receipts: List[dict] = [] 
        self._texto = self._audio = self._vision = None  

    # Cada analizador se crea la primera vez que se usa y luego se reutiliza
    @property
    def texto(self):
        """Analizador de texto (spaCy)"""
        if self._texto is None:
            from src.analyzers.text_analyzer import TextGenderAnalyzer
            self._texto = TextGenderAnalyzer()
        return self._texto

    @property
    def audio(self):
        """Analizador de audio (inaSpeechSegmenter)"""
        if self._audio is None:
            from src.analyzers.audio_analyzer import AudioGenderAnalyzer
            self._audio = AudioGenderAnalyzer()
        return self._audio

    @property
    def vision(self):
        """Analizador de imagen y vídeo (YuNet + OpenCLIP)"""
        if self._vision is None:
            from src.analyzers.vision_analyzer import VisionGenderAnalyzer
            self._vision = VisionGenderAnalyzer()
        return self._vision


    def analyze_batch(self, docs) -> List[ExtractedFeaturesPayload]:
        """Analiza una lista de documentos. Si uno falla, se avisa y se sigue con el resto."""
        payloads = []
        for d in docs:
            try:
                payloads.append(self.analyze(d))
            except Exception as exc:  
                logger.warning("%s: no se ha podido analizar y se omite: %s", d.doc_id, exc)
        return payloads

    def analyze(self, doc) -> ExtractedFeaturesPayload:
        """Analiza un documento y devuelve su resultado"""
        t0 = time.perf_counter()   
        recibo = None
        try:
            for campo in ("image_paths", "audio_paths", "video_paths", "frame_dirs"):
                setattr(doc, campo, [r for r in dict.fromkeys(getattr(doc, campo)) if Path(r).exists()])

            # Cada modalidad por separado
            text_metrics, text_samples = self._analizar_texto(doc)
            audios = self._analizar_audios(doc)                       
            video_metrics, video_samples, caras_video = self._analizar_videos(doc, audios)
            image_metrics, image_samples, caras_fotos = self._analizar_imagenes(doc)

            urls_audio = doc.video_urls or doc.audio_urls             # el audio sale del vídeo si lo hay
            audio_samples = [AudioSample(audio_path=self._url_publica(i, urls_audio), metrics=a.metrics)
                             for i, a in audios.items()]
            audio_metrics = next((a.metrics for a in audios.values() if a.metrics), None)
            subgrupo = self._subgrupo(doc, list(audios.values()), caras_video + caras_fotos)

            payload = ExtractedFeaturesPayload(
                doc_id=doc.doc_id,
                title=doc.title,
                url=doc.url,
                source=doc.source,
                company_query=doc.company_query,
                raw_text_snippet=(doc.raw_text or doc.title or "")[:500],   
                text_samples=text_samples,
                media_image_samples=image_samples,
                media_audio_samples=audio_samples,
                media_video_samples=video_samples,
                demographic_subgroup=subgrupo,
                text_metrics=text_metrics,
                audio_metrics=audio_metrics,
                image_metrics=image_metrics,
                video_metrics=video_metrics,
                biometric_purged=False,   
            )
        finally:
            if self.purge: # El borrado se hace siempre
                recibo = self._borrar_multimedia(doc)

        payload.biometric_purged = bool(recibo and recibo["completo"])
        logger.info("%s analizado en %.1f s: %s", doc.doc_id, time.perf_counter() - t0, payload.available_modalities)
        return payload

    def _borrar_multimedia(self, doc) -> dict:
        try:
            recibo = self.scrubber.purge_biometric_tensors(doc)
        except Exception as exc: 
            logger.warning("%s: no se pudo borrar la multimedia: %s", doc.doc_id, exc)
            recibo = {"doc_id": doc.doc_id, "ficheros_borrados": 0, "no_borrados": [], "completo": False,
                      "error": str(exc)}
        if recibo["no_borrados"]:
            logger.warning("%s: ficheros fuera de su carpeta temporal, no borrados: %s",
                           doc.doc_id, recibo["no_borrados"])
        self.purge_receipts.append(recibo)
        return recibo


    def _analizar_texto(self, doc):
        if not (doc.raw_text or doc.title):
            return None, []
        try:
            r = self.texto.analyze_document(doc)
            return r.metrics, r.samples
        except Exception as exc:  
            logger.warning("%s: fallo en el texto: %s", doc.doc_id, exc)
            return None, []

    def _analizar_audios(self, doc) -> dict:
        resultados = {}
        for ruta in doc.audio_paths:
            try:
                resultados[self._indice(ruta)] = self.audio.analyze(ruta)
            except Exception as exc:  
                logger.warning("%s: fallo en el audio %s: %s", doc.doc_id, ruta, exc)
        return resultados

    def _analizar_videos(self, doc, audios: dict):
        metrics, samples, caras = None, [], []
        for ruta in doc.video_paths:
            i = self._indice(ruta)
            try:
                r = self.vision.analyze_video_frames(ruta, audios.get(i))   
            except Exception as exc:  
                logger.warning("%s: fallo en el vídeo %s: %s", doc.doc_id, ruta, exc)
                continue
            caras += r.frames.faces
            if r.metrics:
                metrics = metrics or r.metrics   
                samples.append(VideoSample(video_path=self._url_publica(i, doc.video_urls), metrics=r.metrics))
        return metrics, samples, caras

    def _analizar_imagenes(self, doc):
        if not doc.image_paths:
            return None, [], []
        try:
            from src.analyzers.vision_analyzer import compute_image_metrics
            r = self.vision.analyze_images(doc.image_paths)
        except Exception as exc:  
            logger.warning("%s: fallo en las imágenes: %s", doc.doc_id, exc)
            return None, [], []
        samples = [ImageSample(image_path=self._url_publica(self._indice(ruta), doc.image_urls),
                               metrics=compute_image_metrics([f for f in r.faces if f.image == n]))
                   for n, ruta in enumerate(doc.image_paths)]
        return r.metrics, samples, r.faces

    @staticmethod
    def _indice(ruta) -> int:
        m = re.search(r"_(\d+)\.\w+$", Path(ruta).name)   
        return int(m.group(1)) if m else 0

    @staticmethod
    def _url_publica(indice: int, urls: List[str]) -> str:
        return urls[indice] if indice < len(urls) else ""

    @staticmethod
    def _subgrupo(doc, audios, caras) -> str:
        """Subgrupo técnico para auditar sesgos (idioma y calidad del audio y la imagen).
        No se infiere etnia ni edad: el AI Act prohíbe deducir el origen étnico a partir
        de datos biométricos (art. 5.1.g).
        """
        from src.analyzers.text_analyzer import TextGenderAnalyzer
        # Idioma del documento ("es", "en" u "otro")
        partes = [TextGenderAnalyzer._idioma(doc.language, doc.raw_text or doc.title or "") or "otro"]
        # Audio ruidoso si en alguno de sus audios hay más de un 30 % de música o ruido
        if audios:
            ruido = max(a.music_noise_ratio for a in audios)
            partes.append("audio-ruidoso" if ruido > 0.3 else "audio-limpio")
        # Caras pequeñas si, de media, son planos lejanos 
        con_genero = [f for f in caras if f.gender]
        if con_genero:
            partes.append("caras-pequenas" if np.mean([f.size for f in con_genero]) < 0.3 else "caras-grandes")
        if len(partes) == 1:
            partes.append("solo-texto")   # ni audio ni caras
        return "_".join(partes)


def save_payloads(payloads: List[ExtractedFeaturesPayload], ruta: str) -> None:
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)  
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump([p.model_dump(mode="json") for p in payloads], f, ensure_ascii=False, indent=2)


def load_payloads(ruta: str) -> List[ExtractedFeaturesPayload]:
    with open(ruta, encoding="utf-8") as f:
        return [ExtractedFeaturesPayload.model_validate(d) for d in json.load(f)]
