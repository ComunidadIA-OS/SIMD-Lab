"""Tests automáticos de los analizadores, el borrado y el pipeline.

    PYTHONPATH=. pytest tests/test_analyzers_unit.py -v
"""
import contextlib
import io
import math

import cv2
import numpy as np
import pytest
from PIL import Image
from skimage import data

from src.analyzers.audio_analyzer import AudioGenderAnalyzer, AudioResult, build_turns, compute_audio_metrics
from src.analyzers.multimodal_pipeline import MultimodalAnalyzerPipeline, load_payloads, save_payloads
from src.analyzers.privacy_scrubber import EphemeralPrivacyScrubber
from src.analyzers.text_analyzer import Mention, TextGenderAnalyzer, TextResult, compute_text_metrics
from src.analyzers.vision_analyzer import MODELO_CARAS, MODELO_CARAS_URL, VisionGenderAnalyzer
from src.common.schemas import ExtractedFeaturesPayload, MediaSource, UnifiedMediaDocument
from tests.fakes_spacy import Doc, FakeNames, Fem, Masc, toy_doc_en, toy_doc_es, toy_doc_es_2


@pytest.fixture
def texto():
    a = object.__new__(TextGenderAnalyzer)
    a.nombres = FakeNames()
    return a


class SegmentadorFalso:
    def __init__(self, trozos):
        self.trozos = trozos

    def __call__(self, ruta):
        print("19/19 - 1s - barra de progreso")  
        return self.trozos


def audio_con(trozos):
    a = object.__new__(AudioGenderAnalyzer)
    a.segmenter = SegmentadorFalso(trozos)
    return a


@pytest.fixture(scope="module")
def vision():
    if not MODELO_CARAS.exists():
        import urllib.request
        MODELO_CARAS.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODELO_CARAS_URL, MODELO_CARAS)
    v = object.__new__(VisionGenderAnalyzer)
    v.detector = cv2.FaceDetectorYN.create(str(MODELO_CARAS), "", (320, 320), 0.7)
    v.textos_genero, v.textos_contexto = "genero", "contexto"

    def clip_falso(recortes, textos):
        filas = []
        for r in recortes:
            a = np.asarray(r).astype(int)
            gris = np.abs(a[..., 0] - a[..., 2]).mean() < 8
            if textos == "genero":
                filas.append([0.1, 0.9] if gris else [0.9, 0.1])
            else:
                p = np.full(10, 0.05)
                p[0] = 0.3 if gris else 0.6
                filas.append(p)
        return np.array(filas)

    v.classify_visual_context = clip_falso
    return v


def foto(nombre):
    img = getattr(data, nombre)()
    return np.stack([img] * 3, axis=-1) if img.ndim == 2 else img


def video(ruta, escenas, fps=10):
    salida = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"MJPG"), fps, (512, 512))
    fondo = cv2.resize(cv2.cvtColor(data.coffee(), cv2.COLOR_RGB2BGR), (512, 512))
    n = 0
    for segundos, img, amplitud in escenas:
        for _ in range(int(segundos * fps)):
            t = n / fps
            if img is None:
                salida.write(fondo)
            else:
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                salida.write(np.roll(bgr, int(amplitud * math.sin(2 * math.pi * t)), axis=1))
            n += 1
    salida.release()
    return str(ruta)


def documento(doc_id, **campos):
    base = dict(doc_id=doc_id, source=MediaSource.ARCHIVE, company_query="EcoRed", url="https://x", title="")
    return UnifiedMediaDocument(**{**base, **campos})


def test_texto_mujer_protagonista(texto):
    texto.nlp = {"es": lambda t: toy_doc_es()}
    r = texto.analyze("x", title="Ana Pérez lidera la transición energética", language="es",
                      company_query="Iberdrola")
    generos = {m.text: m.gender for m in r.mentions}
    assert generos["Ana Pérez"] == "F" and generos["Juan García"] == "M" and generos["ella"] == "F"
    m = r.metrics
    assert m.relative_frequency_score == 0.75          
    assert m.syntactic_agency_ratio == 1.0             
    assert m.expert_quote_ratio == 1.0                
    assert m.feminine_suffix_ratio == 0.5              
    assert m.headline_presence == 1.0
    assert len(r.samples) == 3


def test_texto_mujer_citada_en_ingles(texto):
    texto.nlp = {"en": lambda t: toy_doc_en()}
    r = texto.analyze("x", language="en")
    generos = {m.text: m.gender for m in r.mentions}
    assert generos["Jane Smith"] == "F"               
    assert generos["Brown"] == "M"                    
    assert r.metrics.expert_quote_ratio == 0.5        


def test_texto_apellido_que_parece_cargo(texto):
    texto.nlp = {"es": lambda t: toy_doc_es_2()}
    r = texto.analyze("x", language="Spanish")
    assert all(m.gender == "F" for m in r.mentions if m.gender)   


def frase_cargo(cargo, nombre=False):
    if not nombre:
        return Doc([[("La", "el", "DET", "det", 1, Fem), (cargo, None, "NOUN", "nsubj", 2, Fem),
                     ("explicó", "explicar", "VERB", "ROOT", 2, None), ("el", "el", "DET", "det", 4, Masc),
                     ("plan", None, "NOUN", "obj", 2, Masc), (".", None, "PUNCT", "punct", 2, None)]], [])
    return Doc([[("La", "el", "DET", "det", 1, Fem), (cargo, None, "NOUN", "nsubj", 4, Fem),
                 ("Margarita", None, "PROPN", "appos", 1, None), ("Salas", None, "PROPN", "flat", 2, None),
                 ("explicó", "explicar", "VERB", "ROOT", 4, None), ("el", "el", "DET", "det", 6, Masc),
                 ("plan", None, "NOUN", "obj", 4, Masc), (".", None, "PUNCT", "punct", 4, None)]],
               [(2, 4, "PER")])


@pytest.mark.parametrize("cargo, cuenta", [
    ("arquitecta", True),     
    ("operadora", False),     
    ("química", False),       
])
def test_texto_cargo_suelto(texto, cargo, cuenta):
    texto.nlp = {"es": lambda t: frase_cargo(cargo)}
    r = texto.analyze("x", language="es")
    if cuenta:
        assert [(m.text, m.gender) for m in r.mentions] == [(cargo, "F")]
        assert r.metrics.expert_quote_ratio == 1.0
    else:
        assert r.mentions == [] and r.metrics is None


def test_texto_palabra_ambigua_pegada_a_un_nombre(texto):
    texto.nlp = {"es": lambda t: frase_cargo("química", nombre=True)}
    r = texto.analyze("x", language="es")
    assert [(m.text, m.gender) for m in r.mentions] == [("Margarita Salas", "F")]


def test_texto_sin_personas():
    assert compute_text_metrics([], [], [0], False) is None
    m = Mention("cargo", "x", None, 0, 0, 0)                      
    assert compute_text_metrics([m], [], [0], False) is None


def test_texto_vacio_o_idioma_no_soportado(texto):
    texto.nlp = {}
    assert texto.analyze("", "", "es").metrics is None
    assert texto.analyze("Bonjour", language="fr").metrics is None


@pytest.mark.parametrize("idioma, esperado", [
    ("es", "es"), ("spa", "es"), ("Spanish", "es"), ("es-419", "es"),
    ("en", "en"), ("eng", "en"), ("English", "en"), ("en-GB", "en"), ("ca", None),
])
def test_texto_idioma(idioma, esperado):
    assert TextGenderAnalyzer._idioma(idioma, "") == esperado


def test_texto_nombres_ambiguos_no_cuentan(texto):
    class Nombres:
        def get_gender(self, n):
            return {"andrea": "mostly_female", "alex": "andy"}.get(n.lower(), "unknown")
    texto.nombres = Nombres()
    assert texto._genero_nombre("Andrea") is None
    assert texto._genero_nombre("Alex") is None


def test_audio_tiempo_de_voz_sin_pausas():
    a = audio_con([("noEnergy", 0, 1.0), ("female", 1.0, 2.2), ("noEnergy", 2.2, 2.5),
                   ("female", 2.5, 8.2), ("noEnergy", 8.2, 8.9), ("male", 8.9, 14.6)])
    r = a.analyze("x.wav")
    assert [t.gender for t in r.turns] == ["female", "male"]      
    assert r.metrics.female_speaking_time_ratio == round(6.9 / 12.6, 4)
    assert (r.metrics.opened_by.value, r.metrics.closed_by.value) == ("female", "male")


def test_audio_oculta_barras_de_progreso():
    salida = io.StringIO()
    with contextlib.redirect_stdout(salida):
        audio_con([("female", 0, 2)]).analyze("x.wav")
    assert "barra" not in salida.getvalue()


def test_audio_dos_mujeres_seguidas_cuentan_como_un_turno():
    turnos = build_turns([("female", 0, 2), ("female", 2.1, 5)])
    assert len(turnos) == 1


def test_audio_relevo_sin_pausa():
    turnos = build_turns([("female", 0, 5), ("male", 5.0, 9), ("female", 9.5, 12)])
    assert turnos[0].interrupted and not turnos[2].interrupted


def test_audio_sin_voz():
    assert audio_con([]).analyze("x.wav").metrics is None
    assert audio_con([]).analyze("x.wav").music_noise_ratio == 0.0
    r = audio_con([("music", 0, 10)]).analyze("x.wav")
    assert r.metrics is None and r.music_noise_ratio == 1.0


def test_imagen_formatos(vision, tmp_path):
    Image.fromarray(foto("astronaut")).convert("RGBA").save(tmp_path / "transparente.png")
    (tmp_path / "logo.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>")
    (tmp_path / "roto.jpg").write_bytes(b"esto no es una imagen")
    r = vision.analyze_images([foto("astronaut"), str(tmp_path / "transparente.png"),
                               str(tmp_path / "logo.svg"), str(tmp_path / "roto.jpg"),
                               str(tmp_path / "no_existe.jpg")])
    assert len(r.faces) == 2 and r.metrics.visual_gender_female_ratio == 1.0


def test_imagen_blanco_y_negro(vision):
    r = vision.analyze_images([data.camera()])          
    assert len(r.faces) == 1 and r.faces[0].gender == "M"


def test_imagen_sin_caras(vision):
    assert vision.analyze_images([data.coffee()]).metrics is None


def test_video_mujer_solo_parte_del_tiempo(vision, tmp_path):
    ruta = video(tmp_path / "v.avi", [(1, foto("astronaut"), 0), (9, None, 0)])
    trozos = [("female", 0, 10)]
    audio = AudioResult(compute_audio_metrics(build_turns(trozos), trozos, 10), build_turns(trozos), trozos, 0.0)
    m = vision.analyze_video_frames(ruta, audio).metrics
    assert m.female_screen_time_ratio == 0.1             


def test_video_mujer_y_hombre(vision, tmp_path):
    ruta = video(tmp_path / "v.avi", [(8, foto("astronaut"), 12), (7, foto("camera"), 3)])
    trozos = [("noEnergy", 0, 1.0), ("female", 1.0, 8.3), ("noEnergy", 8.3, 9.0), ("male", 9.0, 14.6)]
    audio = audio_con(trozos).analyze("x.wav")
    m = vision.analyze_video_frames(ruta, audio).metrics
    assert m.female_screen_time_ratio == round(8 / 15, 4)
    assert m.optical_flow_motion_score > 0.5 and m.kinetic_dominance_ratio > 0.5   # ella se mueve más
    assert m.active_speaker_sync_score == 0.875          # se la ve 7 de los 8 s en que se oye su voz


def test_video_sin_audio_o_inexistente(vision, tmp_path):
    ruta = video(tmp_path / "v.avi", [(2, foto("astronaut"), 0)])
    assert vision.analyze_video_frames(ruta, None).metrics is None
    audio = audio_con([("female", 0, 2)]).analyze("x.wav")
    assert vision.analyze_video_frames(str(tmp_path / "no_existe.mp4"), audio).metrics is None


def test_borrado_dentro_y_fuera(tmp_path):
    base = tmp_path / "ramdisk"
    (base / "d1" / "frames_0").mkdir(parents=True)
    (base / "d1" / "image_0.jpg").write_bytes(b"x")
    (base / "d1" / "frames_0" / "f.jpg").write_bytes(b"y")
    fuera = tmp_path / "fuera.wav"
    fuera.write_bytes(b"no tocar")
    doc = documento("d1", image_paths=[str(base / "d1" / "image_0.jpg")],
                    audio_paths=[str(fuera), str(base / ".." / "fuera.wav")],
                    frame_dirs=[str(base / "d1" / "frames_0")])
    recibo = EphemeralPrivacyScrubber(str(base)).purge_biometric_tensors(doc)
    assert not (base / "d1").exists()
    assert fuera.read_bytes() == b"no tocar"
    assert recibo["completo"] is False and len(recibo["no_borrados"]) == 2
    assert doc.image_paths == [] and doc.audio_paths == []


def test_borrado_no_sigue_enlaces_simbolicos(tmp_path):
    base = tmp_path / "ramdisk"
    (base / "d1").mkdir(parents=True)
    ajeno = tmp_path / "ajeno.txt"
    ajeno.write_text("DATOS DE OTRA PERSONA")
    try:
        (base / "d1" / "enlace.jpg").symlink_to(ajeno)
    except OSError:
        pytest.skip("este sistema no deja crear enlaces simbólicos")
    doc = documento("d1", image_paths=[str(base / "d1" / "enlace.jpg")])
    EphemeralPrivacyScrubber(str(base)).purge_biometric_tensors(doc)
    assert not (base / "d1").exists()
    assert ajeno.read_text() == "DATOS DE OTRA PERSONA"


def test_borrado_no_toca_la_carpeta_de_otro_documento(tmp_path):
    base = tmp_path / "ramdisk"
    (base / "d1").mkdir(parents=True)
    (base / "d2").mkdir()
    (base / "d2" / "audio_0.wav").write_bytes(b"de otro documento")
    doc = documento("d1", audio_paths=[str(base / "d2" / "audio_0.wav")])
    recibo = EphemeralPrivacyScrubber(str(base)).purge_biometric_tensors(doc)
    assert (base / "d2" / "audio_0.wav").exists() and recibo["completo"] is False


def test_borrado_con_doc_id_peligroso(tmp_path):
    base = tmp_path / "ramdisk"
    base.mkdir()
    (tmp_path / "importante.txt").write_text("no tocar")
    recibo = EphemeralPrivacyScrubber(str(base)).purge_biometric_tensors(documento(".."))
    assert (tmp_path / "importante.txt").exists() and base.exists()
    assert recibo["ficheros_borrados"] == 0


class TextoFalso:
    def analyze_document(self, doc):
        if "rompe" in (doc.raw_text or ""):
            raise RuntimeError("fallo simulado del analizador de texto")
        m = Mention("nombre", "x", "F", 0, 0, 0, active=True, verb="decir", quoted=True)
        return TextResult(compute_text_metrics([m], [], [0], False), [], [m])


@pytest.fixture
def pipeline(vision, tmp_path):
    base = tmp_path / "ramdisk"
    base.mkdir()
    p = MultimodalAnalyzerPipeline()
    p.scrubber = EphemeralPrivacyScrubber(str(base))
    p._texto, p._vision = TextoFalso(), vision
    p._audio = audio_con([("female", 0, 5), ("male", 5.5, 9)])
    p.base = base
    return p


def test_pipeline_rutas_repetidas_y_url_publica(pipeline):
    d = pipeline.base / "d1"
    d.mkdir()
    Image.fromarray(foto("astronaut")).save(d / "image_0.jpg")
    ruta = str(d / "image_0.jpg")
    doc = documento("d1", raw_text="algo", image_urls=["https://img/0.jpg"],
                    image_paths=[ruta, ruta, "/otra/maquina/image_0.jpg"])
    p = pipeline.analyze(doc)
    assert [s.image_path for s in p.media_image_samples] == ["https://img/0.jpg"]
    assert p.biometric_purged is True and not d.exists()


def test_pipeline_si_falla_un_analizador_sigue(pipeline):
    p = pipeline.analyze(documento("d1", raw_text="rompe"))
    assert p.text_metrics is None and p.doc_id == "d1"


def test_pipeline_borra_aunque_falle_y_el_lote_sigue(pipeline):
    d = pipeline.base / "d1"
    d.mkdir()
    (d / "audio_0.wav").write_bytes(b"x")
    pipeline._subgrupo = lambda *a: (_ for _ in ()).throw(RuntimeError("fallo simulado"))
    with pytest.raises(RuntimeError):
        pipeline.analyze(documento("d1", audio_paths=[str(d / "audio_0.wav")]))
    assert not d.exists()                                   
    resultado = pipeline.analyze_batch([documento("d2", raw_text="a"), documento("d3", raw_text="b")])
    assert resultado == []                                   


def test_pipeline_no_marca_borrado_si_queda_algo(pipeline, tmp_path):
    fuera = tmp_path / "fuera.wav"
    fuera.write_bytes(b"x")
    p = pipeline.analyze(documento("d1", raw_text="a", audio_paths=[str(fuera)]))
    assert p.biometric_purged is False and fuera.exists()


def test_pipeline_json_valido(pipeline, tmp_path):
    d = pipeline.base / "d1"
    d.mkdir()
    Image.fromarray(foto("astronaut")).save(d / "image_0.jpg")
    (d / "audio_0.wav").write_bytes(b"x")
    doc = documento("d1", raw_text="algo", title="Titular", image_urls=["https://img/0.jpg"],
                    audio_urls=["https://audio/0.wav"],
                    image_paths=[str(d / "image_0.jpg")], audio_paths=[str(d / "audio_0.wav")])
    payloads = pipeline.analyze_batch([doc])
    save_payloads(payloads, tmp_path / "salida.json")
    vuelta = load_payloads(tmp_path / "salida.json")
    assert ExtractedFeaturesPayload.model_validate(vuelta[0].model_dump()) == vuelta[0]
    assert set(vuelta[0].available_modalities) == {"Texto", "Audio", "Imagen"}
    assert all(0 <= v <= 1 for k, v in vuelta[0].to_feature_dict().items()
               if not math.isnan(v) and not k.endswith(("frequency", "count")))
