"""Pruebas del preprocesado de texto y del vocabulario por género.

    PYTHONPATH=. pytest tests/test_vocabulario.py -q
Las que usan spaCy se saltan si no está instalado el modelo es_core_news_md.
"""
from collections import Counter

import pandas as pd
import pytest

from src.analyzers.text_preprocessing import limpiar_texto, vectorizar_bow, vectorizar_tfidf
from src.analyzers.vocabulario_genero import log_odds_dirichlet


def test_limpiar_texto():
    assert limpiar_texto("  ¡Hola,   MUNDO! 😀 ") == "hola mundo"


def test_vectorizar_disperso_y_denso():
    textos = pd.Series(["beneficio dividendo", "conciliación igualdad beneficio"])
    dtm, cv = vectorizar_bow(textos)
    assert dtm.shape == (2, 4) and isinstance(dtm.dtypes.iloc[0], pd.SparseDtype)
    tf, _ = vectorizar_tfidf(textos, densa=True)
    assert tf.loc[1, "conciliación"] > 0 and tf.loc[0, "conciliación"] == 0


def test_log_odds_signo_y_minimo():
    mujeres = Counter({"conciliación": 8, "beneficio": 1, "rara": 1})
    hombres = Counter({"beneficio": 9, "conciliación": 1, "rara": 1})
    t = log_odds_dirichlet(mujeres, hombres, min_total=3)
    z = dict(zip(t["lema"], t["z"]))
    assert z["conciliación"] > 0 > z["beneficio"]
    assert "rara" not in z          # menos de 3 apariciones: no se compara


@pytest.fixture(scope="module")
def analizador():
    spacy = pytest.importorskip("spacy")
    try:
        spacy.load("es_core_news_md")
    except OSError:
        pytest.skip("falta el modelo es_core_news_md")
    from src.analyzers.text_analyzer import TextGenderAnalyzer
    return TextGenderAnalyzer()


def test_frases_por_genero_quita_nombres_cargos_y_empresa(analizador):
    from src.analyzers.vocabulario_genero import frases_por_genero
    texto = ("La directora Ana Pérez defendió el plan de conciliación de Inditex. "
             "Juan Martín anunció un beneficio récord.")
    mujeres, hombres = frases_por_genero(texto, analizador, "es", "Inditex")
    assert any("conciliación" in f for f in mujeres) and any("beneficio" in f for f in hombres)
    todas = {w for f in mujeres + hombres for w in f}
    assert not todas & {"ana", "pérez", "juan", "martín", "director", "directora", "inditex"}
