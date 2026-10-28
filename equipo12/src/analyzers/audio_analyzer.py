"""Análisis de audio: cuánto hablan las mujeres y en qué momentos.

Recibe un .wav y devuelve AudioMetrics.

Pasos:
  1. segment_gender_speech: inaSpeechSegmenter parte el audio en trozos
     'female', 'male', 'music', 'noise' y 'noEnergy' (silencio).
  2. detect_interruptions_and_turns: junta los trozos seguidos de voz del mismo género
     en turnos de palabra y marca los relevos sin pausa.
  3. compute_audio_metrics: cuenta y devuelve las métricas.
"""
from __future__ import annotations

import contextlib
import io
import logging
import os
import shutil
import warnings
from dataclasses import dataclass
from typing import List, Optional

from src.common.schemas import AudioMetrics, GenderTarget

NEUTRO = None   # una proporción que no se puede calcular queda sin dato
PAUSA_MISMO_TURNO = 1.0    # si la misma voz para menos segundos que esto, sigue siendo su turno 
PAUSA_INTERRUPCION = 0.1   # si entra otra voz antes de esto, cuenta como corte (segundos)
TURNO_ESTABLE = 1.0        # los turnos más cortos son menos fiables (segundos)

# Puntos de quién abre o cierra el audio: mujer 1, neutro 0.5, hombre 0
PUNTOS = {GenderTarget.FEMALE: 1.0, GenderTarget.NEUTRAL: 0.5, GenderTarget.MALE: 0.0}

@dataclass
class Turn:
    """Un turno: voz del mismo género seguida, sin que entre la del otro género.
    (No se distingue a dos mujeres distintas que hablan seguidas.)"""
    gender: str               # "female" o "male"
    start: float              # segundo en que empieza
    end: float                # segundo en que acaba
    interrupted: bool = False  # True si es de mujer y le sigue un hombre sin pausa

    @property
    def duration(self) -> float:
        """Duración del turno en segundos."""
        return self.end - self.start


@dataclass
class AudioResult:
    metrics: Optional[AudioMetrics]   # las 10 métricas 
    turns: List[Turn]                 # los turnos de palabra
    segments: list            # salida de inaSpeechSegmenter
    music_noise_ratio: float  # parte del audio con música o ruido 


def _ratio(parte: float, total: float) -> Optional[float]:
    return round(parte / total, 4) if total > 0 else NEUTRO


def build_turns(segments) -> List[Turn]:
    turns: List[Turn] = []
    for label, start, end in segments:
        if label not in ("female", "male"):
            continue   # música, ruido y silencio no son turnos de palabra
        # Misma voz que el turno anterior y pausa corta: se alarga ese turno
        if turns and turns[-1].gender == label and start - turns[-1].end < PAUSA_MISMO_TURNO:
            turns[-1].end = end
        else:   # si no, empieza un turno nuevo
            turns.append(Turn(label, start, end))
    # Recorre los turnos de dos en dos (cada uno con el siguiente)
    for actual, siguiente in zip(turns, turns[1:]):
        # Mujer -> hombre casi sin pausa: se marca como interrumpida
        if (actual.gender == "female" and siguiente.gender == "male"
                and siguiente.start - actual.end < PAUSA_INTERRUPCION):
            actual.interrupted = True
    return turns

# Métricas
def compute_audio_metrics(turns: List[Turn], segments, duracion: float) -> Optional[AudioMetrics]:
    if not turns:
        return None   # nadie habla: no se puntúa
    mujeres = [t for t in turns if t.gender == "female"]   # turnos de mujer
    voz = [(label, start, end) for label, start, end in segments if label in ("female", "male")]
    seg_mujeres = sum(end - start for label, start, end in voz if label == "female")   # segundos de voz de mujer
    seg_total = sum(end - start for label, start, end in voz)                          # segundos de voz en total

    # Parte central del audio (del 20 % al 80 %). Quién habla en el "cuerpo"
    ini, fin = 0.2 * duracion, 0.8 * duracion
    def dentro(start: float, end: float) -> float:
        """Segundos de un trozo que caen dentro de la parte central."""
        return max(0.0, min(end, fin) - max(start, ini))
    centro_mujeres = sum(dentro(s, e) for label, s, e in voz if label == "female")
    centro_total = sum(dentro(s, e) for label, s, e in voz)

    # Quién abre (primer turno) y quién cierra (último turno)
    abre = GenderTarget.FEMALE if turns[0].gender == "female" else GenderTarget.MALE
    cierra = GenderTarget.FEMALE if turns[-1].gender == "female" else GenderTarget.MALE
    seg_turnos = sum(t.duration for t in turns)
    seg_estables = sum(t.duration for t in turns if t.duration >= TURNO_ESTABLE)   # turnos de 1 s o más

    return AudioMetrics(
        opened_by=abre,                                   
        closed_by=cierra,                                 
        female_opened_score=PUNTOS[abre],                 # 1 si abre una mujer, 0 si un hombre
        female_closed_score=PUNTOS[cierra],               # 1 si cierra una mujer, 0 si un hombre
        middle_segment_female_ratio=_ratio(centro_mujeres, centro_total),   # voz de mujer en la parte central
        female_speaking_time_ratio=_ratio(seg_mujeres, seg_total),          # tiempo de voz de mujer / total
        female_turn_count=len(mujeres),                   # nº de turnos de mujer
        speaking_turns_ratio=_ratio(len(mujeres), len(turns)),              # turnos de mujer 
        acoustic_gender_confidence=_ratio(seg_estables, seg_turnos), # turnos de 1 s o más 
        no_interruption_score=_ratio(sum(not t.interrupted for t in mujeres), len(mujeres)), # turnos de mujer sin interrupción 
    )


class AudioGenderAnalyzer:

    def __init__(self):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", module="keras")

        from inaSpeechSegmenter import Segmenter
        ffmpeg = "ffmpeg" if shutil.which("ffmpeg") else None
        self.segmenter = Segmenter(vad_engine="smn", detect_gender=True, ffmpeg=ffmpeg)

    def segment_gender_speech(self, wav_path: str) -> list:
        """Trozos del audio con su etiqueta: [('female', 0.0, 7.9), ('male', 8.1, 14.6), ...]"""
        with contextlib.redirect_stdout(io.StringIO()):  
            segments = self.segmenter(wav_path)
        return [(label, float(start), float(end)) for label, start, end in segments]

    def detect_interruptions_and_turns(self, segments) -> List[Turn]:
        """Turnos de palabra e interrupciones."""
        return build_turns(segments)

    def analyze(self, wav_path: str) -> AudioResult:
        """Analiza un .wav y devuelve métricas, turnos, trozos y cuánto hay de música o ruido."""
        segments = self.segment_gender_speech(wav_path)                
        turns = self.detect_interruptions_and_turns(segments)          
        duracion = segments[-1][2] if segments else 0.0                
        musica_ruido = sum(end - start for label, start, end in segments if label in ("music", "noise"))
        return AudioResult(compute_audio_metrics(turns, segments, duracion), turns, segments,   # paso 3
                           round(musica_ruido / duracion, 4) if duracion > 0 else 0.0)
