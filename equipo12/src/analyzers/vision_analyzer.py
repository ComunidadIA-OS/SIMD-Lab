"""Análisis de imagen y vídeo: cuántas mujeres aparecen y cómo se las muestra.

Recibe fotos y devuelve ImageMetrics.
Recibe vídeos y devuelve VideoMetrics.

Pasos:
  1. detect_faces: OpenCV (modelo YuNet) encuentra las caras y 5 puntos de cada una
     (ojos, nariz y comisuras de la boca).
  2. classify_visual_context: OpenCLIP mira cada cara para estimar si se percibe como
     mujer u hombre, y su entorno para decir si el contexto es de liderazgo (atril,
     reunión, conferencia...) o secundario (público, fondo...).
  3. compute_spatial_prominence: tamaño del plano, centralidad, regla de tercios,
     ángulo de cámara e iluminación de cada cara.
  4. compute_image_metrics: cuenta y devuelve las métricas.

Vídeo (analyze_video_frames): se toma un fotograma por segundo y se analiza como
una foto. Además, 0.2 s después se toma otro fotograma para medir cuánto se mueve
cada persona y cuánto cambia su silueta. Con el audio se mira si, cuando se oye una voz de mujer, 
se ve a una mujer en pantalla.
"""
from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from src.common.schemas import ImageMetrics, VideoMetrics

# Modelo de detección de caras (YuNet): se descarga la primera vez y se guarda en ~/.cache/simd
MODELO_CARAS_URL = "https://github.com/ShiqiYu/libfacedetection.train/raw/master/onnx/yunet_s_640_640.onnx"
MODELO_CARAS = Path.home() / ".cache" / "simd" / "yunet_s_640_640.onnx"
LADO_MAXIMO = 1280        # las fotos más grandes se reducen 
CARA_MINIMA = 24          # píxeles de alto: las caras más pequeñas no se clasifican
CONFIANZA_GENERO = 0.6    # por debajo, el género queda como desconocido
SEGUNDOS_ENTRE_MUESTRAS = 1.0   # vídeo: un fotograma por segundo
DESFASE_MOVIMIENTO = 0.2        # vídeo: segundos hasta el fotograma con el que se mide el movimiento

# Frases que OpenCLIP compara con cada cara para estimar el género percibido
TEXTOS_GENERO = ["a photo of a woman", "a photo of a man"]
# Frases de contexto de liderazgo (persona protagonista)
TEXTOS_LIDERAZGO = [
    "a person giving a speech at a podium",
    "an executive leading a business meeting",
    "a person presenting at a conference",
    "an expert being interviewed on television",
    "a scientist or engineer at work",
]
# y de contexto secundario (persona de fondo)
TEXTOS_SECUNDARIO = [
    "a person in the audience",
    "a person in the background of a photo",
    "a person standing in a crowd",
    "an assistant serving other people",
    "a person posing for a casual photo",
]


@dataclass
class Face:
    """Una cara encontrada. Solo guarda números, nunca la imagen de la cara."""
    image: int                
    gender: Optional[str]     
    size: float               # tamaño del plano: 1 = primer plano
    centrality: float         # 1 = en el centro de la imagen
    thirds: float             # 1 = sobre un punto de la regla de tercios
    angle: float              # >0.5 = cámara desde abajo (plano de poder)
    lighting: float           # >0.5 = cara más iluminada que el resto
    leadership: float         # probabilidad de contexto de liderazgo
    box: tuple                # (x, y, ancho, alto) en píxeles


@dataclass
class ImageResult:
    metrics: Optional[ImageMetrics]   
    faces: List[Face]                 # todas las caras encontradas



def compute_spatial_prominence(box, puntos, gris) -> dict:
    """Prominencia de una cara a partir de su caja, sus 5 puntos y la imagen en gris."""
    alto_img, ancho_img = gris.shape
    x, y, w, h = box
    cx, cy = (x + w / 2) / ancho_img, (y + h / 2) / alto_img   # centro de la cara (0-1)

    # Tamaño del plano: alto de la cara respecto al alto de la imagen
    size = min(1.0, (h / alto_img) / 0.4)   # cara del 40 % del alto o más = primer plano
    # Centralidad: 1 en el centro de la imagen, 0 en una esquina
    centrality = 1 - min(1.0, np.hypot(cx - 0.5, cy - 0.5) / np.hypot(0.5, 0.5))
    # Regla de tercios: 1 si la cara está sobre uno de los 4 cruces de los tercios
    tercios = [(1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3)]
    distancia = min(np.hypot(cx - a, cy - b) for a, b in tercios)   
    thirds = 1 - min(1.0, distancia / np.hypot(1 / 3, 1 / 3))

    # Ángulo de cámara
    ojos_y = (puntos[0][1] + puntos[1][1]) / 2
    nariz_y = puntos[2][1]
    boca_y = (puntos[3][1] + puntos[4][1]) / 2
    r = (nariz_y - ojos_y) / max(boca_y - ojos_y, 1.0)  
    angle = float(np.clip(0.5 + (0.55 - r) * 2.5, 0, 1))  

    # Iluminación
    cara = gris[max(int(y), 0): int(y + h), max(int(x), 0): int(x + w)]
    lighting = float(np.clip(0.5 + (cara.mean() - gris.mean()) / 100, 0, 1)) if cara.size else 0.5

    return {"size": round(float(size), 4), "centrality": round(float(centrality), 4),
            "thirds": round(float(thirds), 4), "angle": round(angle, 4), "lighting": round(lighting, 4)}


def compute_image_metrics(faces: List[Face]) -> Optional[ImageMetrics]:
    """Cuenta las caras y devuelve las métricas"""
    con_genero = [f for f in faces if f.gender]   # solo caras con género claro
    if not con_genero:
        return None  
    mujeres = [f for f in con_genero if f.gender == "F"]

    def media_mujeres(campo: str) -> float:
        return round(float(np.mean([getattr(f, campo) for f in mujeres])), 4) if mujeres else 0.0

    imagenes_con_caras = {f.image for f in con_genero}   # nº de imagen con alguna cara
    imagenes_con_mujeres = {f.image for f in mujeres}    # nº de imagen con alguna mujer
    return ImageMetrics(
        visual_gender_female_ratio=round(len(mujeres) / len(con_genero), 4),   # caras de mujer / todas
        shot_scale_importance_score=media_mujeres("size"),          # tamaño del plano de las mujeres
        camera_angle_power_score=media_mujeres("angle"),            # ángulo de cámara
        rule_of_thirds_alignment=media_mujeres("thirds"),           # regla de tercios
        shadow_lighting_prominence=media_mujeres("lighting"),       # iluminación
        cross_image_appearance_count=len(imagenes_con_mujeres),     # en cuántas imágenes sale una mujer
        # De las imágenes con personas, en qué parte sale alguna mujer
        cross_image_persistence_ratio=round(len(imagenes_con_mujeres) / len(imagenes_con_caras), 4),
        spatial_centrality_score=media_mujeres("centrality"),       # centralidad
        leadership_visual_context=media_mujeres("leadership"),      # contexto de liderazgo
    )


# Videos
@dataclass
class Motion:
    """Cuánto se mueve una persona entre dos fotogramas seguidos."""
    gender: str
    flow: float               
    silhouette: float         # cuánto cambia su silueta (0 = nada, 1 = mucho)


@dataclass
class VideoResult:
    metrics: Optional[VideoMetrics]   
    frames: ImageResult       
    seconds_with_women: List[float]  # segundos en los que se ve al menos a una mujer
    motions: List[Motion]             # movimiento de cada persona en cada par de fotogramas


def _relativo(mujeres: List[float], hombres: List[float]) -> float:
    media_m = float(np.mean(mujeres)) if mujeres else 0.0
    media_h = float(np.mean(hombres)) if hombres else 0.0
    return round(media_m / (media_m + media_h), 4) if media_m + media_h > 0 else 0.5


def _cuerpo(box, ancho, alto):
    x, y, w, h = box
    return (int(max(0, x - w)), int(max(0, y - h * 0.3)), int(min(ancho, x + 2 * w)), int(min(alto, y + 3 * h)))


def measure_motion(gris1, gris2, caras: List[Face]) -> List[Motion]:
    """Movimiento y cambio de silueta de cada persona entre dos fotogramas."""
    flujo = cv2.calcOpticalFlowFarneback(gris1, gris2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitud = np.linalg.norm(flujo, axis=2)  
    alto, ancho = gris1.shape
    resultado = []
    for c in caras:
        if not c.gender:
            continue  
        x0, y0, x1, y1 = _cuerpo(c.box, ancho, alto)
        if x1 <= x0 or y1 <= y0:
            continue   
        movimiento = float(magnitud[y0:y1, x0:x1].mean()) / max(c.box[3], 1)
        h1 = cv2.calcHist([gris1[y0:y1, x0:x1]], [0], None, [32], [0, 256])
        h2 = cv2.calcHist([gris2[y0:y1, x0:x1]], [0], None, [32], [0, 256])
        cambio = float(cv2.compareHist(cv2.normalize(h1, h1), cv2.normalize(h2, h2), cv2.HISTCMP_BHATTACHARYYA))
        resultado.append(Motion(c.gender, round(movimiento, 4), round(cambio, 4)))
    return resultado


def compute_video_metrics(frames: ImageResult, tiempos: List[float], motions: List[Motion],
                          audio) -> Optional[VideoMetrics]:
    """Junta fotogramas, movimiento y audio en las métricas del Contrato B.

    tiempos: segundo del vídeo de cada fotograma analizado (en el mismo orden).
    audio: AudioResult del audio del vídeo.
    """
    if frames.metrics is None or audio is None or audio.metrics is None:
        return None   # sin caras o sin voz no se puntúa el vídeo

    # Tiempo en pantalla: de todo el vídeo analizado
    con_mujeres = {tiempos[f.image] for f in frames.faces if f.gender == "F"}

    # Coincidencia voz-imagen: cuando se oye una voz de mujer
    habla_mujer = [t for t in tiempos
                   if any(label == "female" and ini <= t < fin for label, ini, fin in audio.segments)]
    sincronia = sum(t in con_mujeres for t in habla_mujer)

    mov_m = [m.flow for m in motions if m.gender == "F"]   # movimiento de cada mujer
    mov_h = [m.flow for m in motions if m.gender == "M"]   # movimiento de cada hombre
    total_mov = sum(mov_m) + sum(mov_h)
    return VideoMetrics(
        audio_track=audio.metrics,          # métricas del audio del vídeo
        frame_set_visuals=frames.metrics,   # métricas de imagen de los fotogramas
        optical_flow_motion_score=_relativo(mov_m, mov_h),
        kinetic_dominance_ratio=round(sum(mov_m) / total_mov, 4) if total_mov > 0 else 0.5,
        silhouette_change_histogram_score=_relativo([m.silhouette for m in motions if m.gender == "F"],
                                                    [m.silhouette for m in motions if m.gender == "M"]),
        female_screen_time_ratio=round(len(con_mujeres) / len(tiempos), 4) if tiempos else 0.0,
        active_speaker_sync_score=round(sincronia / len(habla_mujer), 4) if habla_mujer else 0.5,
    )


class VisionGenderAnalyzer:

    def __init__(self, modelo: str = "ViT-B-32", pesos: str = "laion2b_s34b_b79k"):
        """Carga OpenCLIP (modelo ViT-B-32 entrenado con LAION-2B) y el detector de caras YuNet."""
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("HF_HUB_VERBOSITY", "error")
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

        import open_clip
        import torch
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"   
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
            modelo, pretrained=pesos, device=self.device)
        self.clip.eval()   # modo de uso (no de entrenamiento)
        tokenizer = open_clip.get_tokenizer(modelo)
        self.textos_genero = self._codificar_textos(tokenizer, TEXTOS_GENERO)
        self.textos_contexto = self._codificar_textos(tokenizer, TEXTOS_LIDERAZGO + TEXTOS_SECUNDARIO)

        if not MODELO_CARAS.exists():
            MODELO_CARAS.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MODELO_CARAS_URL, MODELO_CARAS)
        # 0.7: solo se aceptan caras con al menos un 70 % de seguridad
        self.detector = cv2.FaceDetectorYN.create(str(MODELO_CARAS), "", (320, 320), 0.7)

    def analyze_images(self, imagenes) -> ImageResult:
        faces: List[Face] = []
        for i, imagen in enumerate(imagenes):
            rgb = self._leer(imagen)
            if rgb is not None:   # None = no se ha podido abrir: se salta
                faces += self._caras_de_imagen(rgb, i)
        return ImageResult(compute_image_metrics(faces), faces)

    def analyze_video_frames(self, video_path, audio) -> VideoResult:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0   
        paso = max(1, round(fps * SEGUNDOS_ENTRE_MUESTRAS))   
        desfase = max(1, min(paso - 1, round(fps * DESFASE_MOVIMIENTO))) if paso > 1 else 0
        faces: List[Face] = []
        tiempos: List[float] = []   
        motions: List[Motion] = []
        anterior = None   
        n = 0             
        while True:
            ok, bgr = cap.read()
            if not ok:
                break   # fin del vídeo
            if n % paso == 0:
                rgb = self._leer(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                caras = self._caras_de_imagen(rgb, len(tiempos))
                tiempos.append(n / fps)
                faces += caras
                anterior = (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), caras) if caras else None
            elif anterior is not None and n % paso == desfase:
                gris = cv2.cvtColor(self._leer(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), cv2.COLOR_RGB2GRAY)
                motions += measure_motion(anterior[0], gris, anterior[1])
                anterior = None
            n += 1
        cap.release()

        frames = ImageResult(compute_image_metrics(faces), faces)
        metrics = compute_video_metrics(frames, tiempos, motions, audio)
        segundos_mujeres = sorted({tiempos[f.image] for f in faces if f.gender == "F"})
        return VideoResult(metrics, frames, segundos_mujeres, motions)

    # encontrar caras
    def detect_faces(self, bgr) -> np.ndarray:
        alto, ancho = bgr.shape[:2]
        self.detector.setInputSize((ancho, alto))   
        _, caras = self.detector.detect(bgr)
        return np.zeros((0, 15)) if caras is None else caras   # sin caras: tabla vacía

    # género y contexto con OpenCLIP
    def classify_visual_context(self, recortes, textos) -> np.ndarray:
        with self.torch.no_grad():   
            x = self.torch.stack([self.preprocess(r) for r in recortes]).to(self.device)
            emb = self.clip.encode_image(x)                 
            emb = emb / emb.norm(dim=-1, keepdim=True)      
            return (100 * emb @ textos.T).softmax(dim=-1).cpu().numpy()

    def _codificar_textos(self, tokenizer, textos):
        with self.torch.no_grad():
            t = self.clip.encode_text(tokenizer(textos).to(self.device))
            return t / t.norm(dim=-1, keepdim=True)

    def _caras_de_imagen(self, rgb, indice: int) -> List[Face]:
        """Encuentra las caras de una imagen y calcula todo lo de cada una"""
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gris = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        alto, ancho = gris.shape
        caras = [c for c in self.detect_faces(bgr) if c[3] >= CARA_MINIMA]   
        if not caras:
            return []
        foto = Image.fromarray(rgb)
        recortes_cara = [foto.crop(self._ampliar(c[:4], 1.6, ancho, alto)) for c in caras]
        recortes_entorno = [foto.crop(self._ampliar(c[:4], 4.0, ancho, alto, con_cuerpo=True)) for c in caras]
        p_genero = self.classify_visual_context(recortes_cara, self.textos_genero)
        p_entorno = self.classify_visual_context(recortes_entorno, self.textos_contexto)

        resultado = []
        for c, pg, pe in zip(caras, p_genero, p_entorno):
            box = tuple(int(v) for v in c[:4])
            gender = "F" if pg[0] >= CONFIANZA_GENERO else "M" if pg[1] >= CONFIANZA_GENERO else None
            prominencia = compute_spatial_prominence(box, c[4:14].reshape(5, 2), gris)   
            liderazgo = round(float(pe[: len(TEXTOS_LIDERAZGO)].sum()), 4)
            resultado.append(Face(indice, gender, leadership=liderazgo, box=box, **prominencia))
        return resultado

    @staticmethod
    def _ampliar(box, factor, ancho, alto, con_cuerpo=False):
        x, y, w, h = box
        cx = x + w / 2   
        izquierda, derecha = cx - w * factor / 2, cx + w * factor / 2
        if con_cuerpo:
            arriba, abajo = y - h * 0.5, y + h * factor
        else:
            cy = y + h / 2
            arriba, abajo = cy - h * factor / 2, cy + h * factor / 2
        return (int(max(0, izquierda)), int(max(0, arriba)), int(min(ancho, derecha)), int(min(alto, abajo)))

    @staticmethod
    def _leer(imagen):
        """Ruta o array -> array RGB, reducido si es muy grande."""
        try:
            if isinstance(imagen, (str, Path)):
                rgb = np.array(Image.open(imagen).convert("RGB"))
            else:  
                rgb = np.asarray(Image.fromarray(np.asarray(imagen, dtype=np.uint8)).convert("RGB"))
        except OSError:
            return None   
        escala = LADO_MAXIMO / max(rgb.shape[:2])
        if escala < 1:  
            rgb = cv2.resize(rgb, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(rgb)   

    # ---------------------------------------------- para LIME
    def predict_proba(self, imagenes) -> np.ndarray:
        """[[P(contexto secundario), P(contexto de liderazgo)], ...] para LimeImageExplainer.

        LIME tapa zonas de la imagen y mira cómo cambia esta probabilidad para decir
        qué zonas pesan más en que se vea como contexto de liderazgo.
        """
        recortes = [Image.fromarray(np.asarray(im, dtype=np.uint8)) for im in imagenes]
        p = self.classify_visual_context(recortes, self.textos_contexto)[:, : len(TEXTOS_LIDERAZGO)].sum(axis=1)
        return np.stack([1 - p, p], axis=1)
