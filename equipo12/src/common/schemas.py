from enum import Enum
from typing import Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

# Nombres e ISO 639-2 que devuelven las fuentes -> ISO 639-1 de dos letras
_LANG_ALIASES = {
    "spanish": "es", "español": "es", "spa": "es",
    "english": "en", "eng": "en",
    "catalan": "ca", "català": "ca", "cat": "ca",
    "french": "fr", "fre": "fr", "fra": "fr",
    "german": "de", "ger": "de", "deu": "de",
    "portuguese": "pt", "por": "pt",
    "italian": "it", "ita": "it",
}
# Normalizamos el idioma de las fuentes a un código ISO 639-1 de dos letras
def normalize_language(value: Optional[str]) -> Optional[str]:
    """'Spanish', 'spa', 'es-ES' o 'es' -> 'es'. Devuelve None si no se reconoce."""
    if not value:
        return None
    v = str(value).strip().lower().replace("_", "-").split("-")[0]
    if v in _LANG_ALIASES:
        return _LANG_ALIASES[v]
    return v if len(v) == 2 and v.isalpha() else None

class MediaSource(str, Enum):
    WIKIMEDIA = "Wikimedia"
    ARCHIVE = "Internet Archive"
    YOUTUBE = "YouTube"
    GDELT = "GDELT"
    GUARDIAN = "The Guardian"
    PODCAST = "Podcast Index"

class UnifiedMediaDocument(BaseModel):
    """Documento normalizado que sale de la ingesta (conectores + reranker + descarga).

    Los campos doc_id, title, url, source y company_query se copian tal cual
    en ExtractedFeaturesPayload.
    """
    doc_id: str = Field(..., description="Hash estable de source + url")
    source: MediaSource
    company_query: str = Field(..., description="Empresa o término buscado")
    url: str
    title: str
    published_date: Optional[str] = Field(None, description="Fecha ISO 8601 si la fuente la proporciona")
    language: Optional[str] = Field(None, description="Código o nombre de idioma según la fuente")
    license: Optional[str] = Field(None, description="Licencia declarada por la fuente, si existe")
    author: Optional[str] = Field(None, description="Firma, canal o autor")
    raw_text: Optional[str] = Field(None, description="Texto completo disponible (cuerpo, extracto o descripción)")

    # URLs remotas (las rellenan los conectores; aún no se ha descargado nada)
    image_urls: List[str] = Field(default_factory=list)
    audio_urls: List[str] = Field(default_factory=list)
    video_urls: List[str] = Field(default_factory=list)
    subtitle_urls: List[str] = Field(default_factory=list)

    # Rutas locales en el volumen efímero (las rellena MediaDownloader tras el reranking)
    image_paths: List[str] = Field(default_factory=list)
    audio_paths: List[str] = Field(default_factory=list, description=".wav 16 kHz mono")
    video_paths: List[str] = Field(default_factory=list, description="Clip .mp4 recortado y reescalado")
    frame_dirs: List[str] = Field(default_factory=list, description="Carpetas con fotogramas a 1 fps")

    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Puntuación del reranker")
    extra: Dict[str, str] = Field(default_factory=dict, description="Metadatos propios de cada fuente")
    
    @field_validator("language", mode="before")
    @classmethod
    def _normalize_language(cls, v):
        return normalize_language(v)

    @property
    def available_modalities(self) -> List[str]:
        mods = []
        if self.raw_text:
            mods.append("Texto")
        if self.image_urls or self.image_paths:
            mods.append("Imagen")
        if self.audio_urls or self.audio_paths or self.video_urls:
            mods.append("Audio")
        if self.video_urls or self.video_paths:
            mods.append("Vídeo")
        return mods

    def rerank_text(self, max_chars: int = 2000) -> str:
        """Texto compacto (título + inicio del cuerpo) que usa el reranker."""
        body = (self.raw_text or "")[:max_chars]
        return f"{self.title}\n{body}".strip()


class GenderTarget(str, Enum):
    FEMALE = "female"
    MALE = "male"
    NEUTRAL = "neutral"


# 1. MÉTRICAS DE TEXTO
class TextMetrics(BaseModel):
    sentence_female_ratio: float = Field(..., ge=0.0, le=1.0, description="Frases que se refieren a mujer")
    sentence_male_ratio: float = Field(..., ge=0.0, le=1.0, description="Frases que se refieren a hombre")
    sentence_neutral_ratio: float = Field(..., ge=0.0, le=1.0, description="Frases neutras")
    lexical_root_diversity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Extracción de raíces léxicas asociadas a sujetos femeninos (None si no se puede calcular)")
    feminine_suffix_ratio: Optional[float] = Field(None, ge=0.0, le=1.0, description="Análisis de terminaciones léxicas femeninas (-a, -ora, -triz) (None si no se puede calcular)")
    female_mention_frequency: int = Field(..., ge=0, description="Frecuencia absoluta de menciones")
    relative_frequency_score: float = Field(..., ge=0.0, le=1.0, description="Frecuencia relativa normalizada")
    syntactic_agency_ratio: Optional[float] = Field(None, ge=0.0, le=1.0, description="Agencia sintáctica (mujer como sujeto activo) (None si no se puede calcular)")
    expert_quote_ratio: Optional[float] = Field(None, ge=0.0, le=1.0, description="Citas directas como experta (None si no se puede calcular)")
    headline_presence: float = Field(..., ge=0.0, le=1.0, description="Presencia en titular")


# 2. MÉTRICAS DE AUDIO
class AudioMetrics(BaseModel):
    opened_by: GenderTarget = Field(..., description="Quién abre el audio")
    closed_by: GenderTarget = Field(..., description="Quién cierra el audio")
    female_opened_score: float = Field(..., ge=0.0, le=1.0, description="Puntuación de apertura femenina")
    female_closed_score: float = Field(..., ge=0.0, le=1.0, description="Puntuación de cierre femenino")
    middle_segment_female_ratio: Optional[float] = Field(None, ge=0.0, le=1.0, description="Presencia en el 20%-80% central del audio (None si no se puede calcular)")
    female_speaking_time_ratio: Optional[float] = Field(None, ge=0.0, le=1.0, description="Ratio de duración de voz femenina (None si no se puede calcular)")
    female_turn_count: int = Field(..., ge=0, description="Conteo de turnos de palabra femeninos")
    speaking_turns_ratio: Optional[float] = Field(None, ge=0.0, le=1.0, description="Proporción de turnos de habla femeninos (None si no se puede calcular)")
    acoustic_gender_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confianza del reconocimiento de género por voz (None si no se puede calcular)")
    no_interruption_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Ausencia de interrupciones en turno femenino (None si no se puede calcular)")


# 3. MÉTRICAS DE IMAGEN
class ImageMetrics(BaseModel):
    visual_gender_female_ratio: float = Field(..., ge=0.0, le=1.0, description="Reconocimiento de género en imagen")
    shot_scale_importance_score: float = Field(..., ge=0.0, le=1.0, description="Detección de plano de importancia")
    camera_angle_power_score: float = Field(..., ge=0.0, le=1.0, description="Ángulo de cámara")
    rule_of_thirds_alignment: float = Field(..., ge=0.0, le=1.0, description="Alineación con regla de tercios")
    shadow_lighting_prominence: float = Field(..., ge=0.0, le=1.0, description="Análisis de sombras e iluminación")
    cross_image_appearance_count: int = Field(..., ge=0, description="Apariciones entre el conjunto de imágenes")
    cross_image_persistence_ratio: float = Field(..., ge=0.0, le=1.0, description="Ratio de persistencia visual")
    spatial_centrality_score: float = Field(..., ge=0.0, le=1.0, description="Centralidad espacial")
    leadership_visual_context: float = Field(..., ge=0.0, le=1.0, description="Contexto visual de liderazgo")


# 4. MÉTRICAS DE VÍDEO
class VideoMetrics(BaseModel):
    audio_track: AudioMetrics = Field(..., description="Métricas de audio del vídeo")
    frame_set_visuals: ImageMetrics = Field(..., description="Métricas de fotogramas del vídeo")
    optical_flow_motion_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Puntuación de movimiento por flujo óptico (None si no se puede calcular)")
    kinetic_dominance_ratio: Optional[float] = Field(None, ge=0.0, le=1.0, description="Dominancia cinética femenina (None si no se puede calcular)")
    silhouette_change_histogram_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Histograma de cambios por silueta (None si no se puede calcular)")
    female_screen_time_ratio: float = Field(..., ge=0.0, le=1.0, description="Tiempo total en pantalla")
    active_speaker_sync_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Sincronía de habla y presencia activa (None si no se puede calcular)")


# --- SUB-MODELOS ENRIQUECIDOS POR MUESTRA INDIVIDUAL ---
class TextSample(BaseModel):
    text: str
    metrics: Optional[TextMetrics] = None


class ImageSample(BaseModel):
    image_path: str
    metrics: Optional[ImageMetrics] = None


class AudioSample(BaseModel):
    audio_path: str
    metrics: Optional[AudioMetrics] = None


class VideoSample(BaseModel):
    video_path: str
    metrics: Optional[VideoMetrics] = None


# ESQUEMA PRINCIPAL DEL DOCUMENTO AUDITADO
class ExtractedFeaturesPayload(BaseModel):
    doc_id: str
    title: str
    url: str
    source: MediaSource
    company_query: str
    raw_text_snippet: Optional[str] = None
    
    # Listas enriquecidas con soporte multi-muestra y métricas individuales
    text_samples: List[TextSample] = Field(default_factory=list)
    media_image_samples: List[ImageSample] = Field(default_factory=list)
    media_audio_samples: List[AudioSample] = Field(default_factory=list)
    media_video_samples: List[VideoSample] = Field(default_factory=list)

    demographic_subgroup: str
    available_modalities: List[str] = Field(default_factory=list)
    
    # Métricas agregadas (para el modelo global LightGBM y SHAP global)
    text_metrics: Optional[TextMetrics] = None
    audio_metrics: Optional[AudioMetrics] = None
    image_metrics: Optional[ImageMetrics] = None
    video_metrics: Optional[VideoMetrics] = None
    
    biometric_purged: bool = Field(default=True)

    @model_validator(mode="after")
    def auto_detect_modalities(self):
        # Sincronización retrocompatible para textos planos
        if self.raw_text_snippet and not self.text_samples:
            self.text_samples = [TextSample(text=self.raw_text_snippet, metrics=self.text_metrics)]
        elif self.text_samples and not self.raw_text_snippet:
            self.raw_text_snippet = self.text_samples[0].text

        detected = []
        if self.text_metrics is not None or len(self.text_samples) > 0:
            detected.append("Texto")
        if self.audio_metrics is not None or len(self.media_audio_samples) > 0:
            detected.append("Audio")
        if self.image_metrics is not None or len(self.media_image_samples) > 0:
            detected.append("Imagen")
        if self.video_metrics is not None or len(self.media_video_samples) > 0:
            detected.append("Vídeo")

        self.available_modalities = detected
        return self

    def to_feature_dict(self) -> Dict[str, float]:
        """Extrae el vector plano agregado para alimentar LightGBM y SHAP global."""
        feats = {}

        # 1. Texto
        tm = self.text_metrics
        feats["txt_sentence_female_ratio"] = tm.sentence_female_ratio if tm else np.nan
        feats["txt_sentence_male_ratio"] = tm.sentence_male_ratio if tm else np.nan
        feats["txt_sentence_neutral_ratio"] = tm.sentence_neutral_ratio if tm else np.nan
        feats["txt_lexical_root_diversity"] = tm.lexical_root_diversity if tm else np.nan
        feats["txt_feminine_suffix_ratio"] = tm.feminine_suffix_ratio if tm else np.nan
        feats["txt_relative_frequency_score"] = tm.relative_frequency_score if tm else np.nan
        feats["txt_syntactic_agency_ratio"] = tm.syntactic_agency_ratio if tm else np.nan
        feats["txt_expert_quote_ratio"] = tm.expert_quote_ratio if tm else np.nan
        feats["txt_headline_presence"] = tm.headline_presence if tm else np.nan
        feats["txt_female_mention_frequency"] = float(tm.female_mention_frequency) if tm else np.nan
        
        # 2. Audio
        am = self.audio_metrics
        feats["aud_female_opened_score"] = am.female_opened_score if am else np.nan
        feats["aud_female_closed_score"] = am.female_closed_score if am else np.nan
        feats["aud_middle_segment_female_ratio"] = am.middle_segment_female_ratio if am else np.nan
        feats["aud_female_speaking_time_ratio"] = am.female_speaking_time_ratio if am else np.nan
        feats["aud_speaking_turns_ratio"] = am.speaking_turns_ratio if am else np.nan
        feats["aud_acoustic_gender_confidence"] = am.acoustic_gender_confidence if am else np.nan
        feats["aud_no_interruption_score"] = am.no_interruption_score if am else np.nan
        feats["aud_female_turn_count"] = float(am.female_turn_count) if am else np.nan

        # 3. Imagen
        im = self.image_metrics
        feats["img_visual_gender_female_ratio"] = im.visual_gender_female_ratio if im else np.nan
        feats["img_shot_scale_importance_score"] = im.shot_scale_importance_score if im else np.nan
        feats["img_camera_angle_power_score"] = im.camera_angle_power_score if im else np.nan
        feats["img_rule_of_thirds_alignment"] = im.rule_of_thirds_alignment if im else np.nan
        feats["img_shadow_lighting_prominence"] = im.shadow_lighting_prominence if im else np.nan
        feats["img_cross_image_persistence_ratio"] = im.cross_image_persistence_ratio if im else np.nan
        feats["img_spatial_centrality_score"] = im.spatial_centrality_score if im else np.nan
        feats["img_leadership_visual_context"] = im.leadership_visual_context if im else np.nan
        feats["img_cross_image_appearance_count"] = float(im.cross_image_appearance_count) if im else np.nan

        # 4. Vídeo
        vm = self.video_metrics
        feats["vid_optical_flow_motion_score"] = vm.optical_flow_motion_score if vm else np.nan
        feats["vid_kinetic_dominance_ratio"] = vm.kinetic_dominance_ratio if vm else np.nan
        feats["vid_silhouette_change_histogram_score"] = vm.silhouette_change_histogram_score if vm else np.nan
        feats["vid_female_screen_time_ratio"] = vm.female_screen_time_ratio if vm else np.nan
        feats["vid_active_speaker_sync_score"] = vm.active_speaker_sync_score if vm else np.nan

        # Las cuotas sin dato llegan como None: se pasan a NaN para pandas
        return {k: (np.nan if v is None else v) for k, v in feats.items()}


