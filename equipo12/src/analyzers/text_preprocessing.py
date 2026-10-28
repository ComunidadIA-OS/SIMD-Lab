"""Preprocesado de texto para análisis de vocabulario: limpieza, lematización y vectores.

Sirve para estudiar DE QUÉ se habla (palabras y temas), por ejemplo para comparar el
vocabulario de las frases sobre mujeres y sobre hombres (vocabulario_genero.py) o como
entrada de BERTopic.

IMPORTANTE: no se usa en text_analyzer.py. Para medir la presencia de mujeres hacen falta
justo las palabras que aquí se quitan: los pronombres ("ella", "él") y los artículos
("la directora") son palabras vacías, y las mayúsculas y la puntuación sirven para encontrar
nombres de persona y separar frases.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Mismos modelos que el analizador de texto (así no se carga un segundo modelo en memoria)
MODELOS = {"es": "es_core_news_md", "en": "en_core_web_md"}
_nlp: dict = {}

# Etiquetas de entidad de persona en los modelos de spaCy (español: PER, inglés: PERSON)
ENTIDADES_PERSONA = {"PER", "PERSON"}


def get_nlp(lang: str = "es"):
    """Carga el modelo de spaCy la primera vez que se pide (no al importar el módulo)."""
    if lang not in _nlp:
        import spacy
        _nlp[lang] = spacy.load(MODELOS[lang])
    return _nlp[lang]


def limpiar_texto(texto: str) -> str:
    """Minúsculas, sin signos de puntuación ni emojis y sin espacios repetidos.

    Se aplica a los lemas ya extraídos, no al texto original: si se pasa a minúsculas y
    se quita la puntuación ANTES de spaCy, se estropean el lematizador y la separación
    de frases.
    """
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def es_token_util(token, excluir: Optional[set] = None) -> bool:
    """Decide si una palabra aporta contenido.

    Fuera: espacios, puntuación, palabras vacías, números, URLs, palabras de menos de 3
    letras, nombres de persona (si no, "Ana" o "García" saldrían como vocabulario) y
    lo que se pase en `excluir` (lemas en minúscula).
    """
    if token.is_space or token.is_punct or token.is_stop or token.like_num or token.like_url:
        return False
    if not token.is_alpha or len(token.text) < 3:
        return False
    if token.ent_type_ in ENTIDADES_PERSONA:
        return False
    return not (excluir and token.lemma_.lower() in excluir)


def lemas_utiles(tokens: Iterable, excluir: Optional[set] = None) -> List[str]:
    """Lemas limpios de una frase o documento de spaCy."""
    lemas = (limpiar_texto(t.lemma_) for t in tokens if es_token_util(t, excluir))
    return [lema for lema in lemas if lema]


def lematizar_por_lotes(series_textos: pd.Series, lang: str = "es", batch_size: int = 300,
                        n_process: int = 1, excluir: Optional[set] = None) -> pd.Series:
    """Recibe una Series de textos ORIGINALES y devuelve otra con los lemas útiles unidos por espacios.

    n_process=1 por defecto: con varios procesos, cada uno carga su propia copia del
    modelo, y en Windows y en libretas de Jupyter suele colgarse.
    """
    textos = series_textos.fillna("").astype(str).tolist()
    salida = [" ".join(lemas_utiles(doc, excluir))
              for doc in get_nlp(lang).pipe(textos, batch_size=batch_size, n_process=n_process)]
    return pd.Series(salida, index=series_textos.index, name=getattr(series_textos, "name", None))


def pln_pipeline(texto: str, lang: str = "es", excluir: Optional[set] = None) -> str:
    """Un solo texto: lemas útiles unidos por espacios."""
    return " ".join(lemas_utiles(get_nlp(lang)(texto), excluir))


def _como_dataframe(matriz, columnas, textos, densa: bool):
    """Matriz documento-término. Por defecto se queda dispersa: con muchos textos, una
    matriz densa (toarray) puede ocupar varios GB."""
    indice = textos.index if isinstance(textos, pd.Series) else None
    if densa:
        return pd.DataFrame(matriz.toarray(), columns=columnas, index=indice)
    return pd.DataFrame.sparse.from_spmatrix(matriz, columns=columnas, index=indice)


def vectorizar_bow(textos, densa: bool = False, **vectorizer_kwargs):
    """Bolsa de palabras (recuentos). Devuelve (DataFrame documento-término, vectorizador)."""
    cv = CountVectorizer(**vectorizer_kwargs)
    dtm = cv.fit_transform(textos)
    return _como_dataframe(dtm, cv.get_feature_names_out(), textos, densa), cv


def vectorizar_tfidf(textos, densa: bool = False, **vectorizer_kwargs):
    """TF-IDF. Devuelve (DataFrame documento-término, vectorizador)."""
    tf = TfidfVectorizer(**vectorizer_kwargs)
    tfidf = tf.fit_transform(textos)
    return _como_dataframe(tfidf, tf.get_feature_names_out(), textos, densa), tf
