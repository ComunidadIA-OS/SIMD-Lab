
import json
import math
import os
import tempfile

import cv2
import numpy as np

from src.analyzers.text_analyzer import TextGenderAnalyzer
from src.common.schemas import UnifiedMediaDocument
from src.analyzers.audio_analyzer import AudioGenderAnalyzer
from src.analyzers.vision_analyzer import VisionGenderAnalyzer
from skimage import data


def crear_video_prueba(segundos=15, fps=10):
    """Vídeo sintético: la astronauta moviéndose 8 s y después un plano sin personas (voz en off)."""
    ruta = os.path.join(tempfile.gettempdir(), "video_prueba.avi")
    astronauta = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    cafe = cv2.resize(cv2.cvtColor(data.coffee(), cv2.COLOR_RGB2BGR), (512, 512))
    salida = cv2.VideoWriter(ruta, cv2.VideoWriter_fourcc(*"MJPG"), fps, (512, 512))
    for n in range(segundos * fps):
        t = n / fps
        if t < 8:
            salida.write(np.roll(astronauta, int(12 * math.sin(2 * math.pi * t)), axis=1))
        else:
            salida.write(cafe)
    salida.release()
    return ruta


analyzer = TextGenderAnalyzer()

print("1. Frase de prueba")
res = analyzer.analyze(
    "La consejera delegada de Iberdrola, Ana Pérez, explicó que la red necesita almacenamiento. "
    "El presidente, Juan García, fue reelegido. «Vamos a invertir», dijo ella.",
    title="Ana Pérez lidera la transición energética", language="es", company_query="Iberdrola")
for m in res.mentions:
    print(f"  {m.text:<15} género={m.gender}  hace_la_acción={m.active}  citada={m.quoted}")
print(" ", res.metrics)

print("\n2. Documentos de la ingesta (tests/mocks/sample_top_n_docs.json)")
with open("tests/mocks/sample_top_n_docs.json", encoding="utf-8") as f:
    docs = [UnifiedMediaDocument.model_validate(d) for d in json.load(f)]
for d in docs:
    res = analyzer.analyze_document(d)
    mujeres = sum(m.gender == "F" for m in res.mentions)
    hombres = sum(m.gender == "M" for m in res.mentions)
    print(f"\n{d.title}  ({mujeres} menciones de mujeres, {hombres} de hombres)")
    print(" ", res.metrics)

print("\n3. Audio de prueba (mujer y luego hombre)")
audio = AudioGenderAnalyzer().analyze("tests/mocks/lamartine.wav")
print("  Trozos detectados:")
for label, start, end in audio.segments:
    print(f"    {label:<9} {start:5.1f}s - {end:5.1f}s")
print("  Turnos:")
for t in audio.turns:
    print(f"  {t.gender:<6} {t.start:5.1f}s - {t.end:5.1f}s  cortada={t.interrupted}")
print(" ", audio.metrics)

print("\n4. Imagen de prueba (la astronauta Eileen Collins)")
vision = VisionGenderAnalyzer()
imagen = vision.analyze_images([data.astronaut()])
for f in imagen.faces:
    print(f"  cara en {f.box}: género={f.gender}  plano={f.size}  centro={f.centrality}  "
          f"ángulo={f.angle}  luz={f.lighting}  liderazgo={f.leadership}")
print(" ", imagen.metrics)

print("\n5. Vídeo de prueba (la astronauta 8 s, luego un plano sin personas; audio de la parte 3)")
video = vision.analyze_video_frames(crear_video_prueba(), audio)
print("  Segundos con una mujer en pantalla:", video.seconds_with_women)
print(" ", video.metrics.model_dump(exclude={"audio_track", "frame_set_visuals"}))
