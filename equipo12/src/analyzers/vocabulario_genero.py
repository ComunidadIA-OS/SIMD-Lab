"""Vocabulario asociado a mujeres y a hombres en los textos de una empresa (encuadre).

El análisis de texto mide QUIÉN aparece y CÓMO (sujeto, citas). Esto mide DE QUÉ se habla:
qué palabras aparecen más en las frases sobre mujeres que en las frases sobre hombres, y al
revés. Por ejemplo, "conciliación" frente a "beneficio".

Método:
  1. El analizador de texto encuentra las menciones a personas y su género.
  2. Cada frase se asigna a "mujeres" si menciona más mujeres que hombres, y a "hombres"
     al revés (el mismo criterio que sentence_female_ratio). Las frases empatadas o sin
     personas no cuentan.
  3. Se lematizan las palabras útiles (text_preprocessing): fuera palabras vacías, nombres
     de persona, cargos, pronombres y el nombre de la empresa, para no medir lo obvio.
  4. Se comparan los dos grupos con el log-odds ratio con prior de Dirichlet informativo
     (Monroe, Colaresi y Quinn, 2008, "Fightin' Words"). Da una puntuación z por palabra:
     |z| > 1,96 indica una diferencia significativa al 95 %. Es más fiable que comparar
     frecuencias o TF-IDF cuando hay pocas frases, porque no exagera las palabras raras.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.analyzers.text_analyzer import MAX_CHARS, PRONOMBRES, ROLES, TRATAMIENTOS
from src.analyzers.text_preprocessing import lemas_utiles

logger = logging.getLogger(__name__)
Z_SIGNIFICATIVO = 1.96


def _excluir(lang: str, empresa: str) -> set:
    """Palabras que delatan el género por sí mismas o que salen en todo: no son encuadre."""
    return ({w.lower() for w in ROLES[lang]} | set(PRONOMBRES[lang]) | set(TRATAMIENTOS[lang])
            | {w.lower() for w in (empresa or "").split()})


def frases_por_genero(texto: str, analyzer, language: Optional[str] = None,
                      empresa: str = "") -> Tuple[List[List[str]], List[List[str]]]:
    """Lemas útiles de cada frase sobre mujeres y de cada frase sobre hombres de un texto."""
    lang = analyzer._idioma(language, texto or "")
    if not texto or lang not in analyzer.nlp:
        return [], []
    doc = analyzer.nlp[lang](texto[:MAX_CHARS])
    mentions, _ = analyzer.extract_entities_and_coref(doc, lang, empresa)
    cuenta = Counter((m.sent, m.gender) for m in mentions if m.gender in ("F", "M"))
    excluir = _excluir(lang, empresa)
    mujeres, hombres = [], []
    for k, frase in enumerate(doc.sents):
        f, h = cuenta[(k, "F")], cuenta[(k, "M")]
        if f == h:
            continue
        (mujeres if f > h else hombres).append(lemas_utiles(frase, excluir))
    return mujeres, hombres


def log_odds_dirichlet(c_mujeres: Counter, c_hombres: Counter, min_total: int = 3) -> pd.DataFrame:
    """Log-odds ratio con prior de Dirichlet informativo (Monroe et al., 2008).

    El prior es la frecuencia de cada palabra en los dos grupos juntos. z > 0: más
    asociada a las frases sobre mujeres; z < 0: a las frases sobre hombres.
    """
    vocab = [w for w in set(c_mujeres) | set(c_hombres) if c_mujeres[w] + c_hombres[w] >= min_total]
    if not vocab:
        return pd.DataFrame(columns=["lema", "frases_mujeres", "frases_hombres", "z"])
    yf = np.array([c_mujeres[w] for w in vocab], dtype=float)
    ym = np.array([c_hombres[w] for w in vocab], dtype=float)
    alpha = yf + ym                              # prior: frecuencia conjunta
    a0, nf, nm = alpha.sum(), yf.sum(), ym.sum()
    delta = (np.log((yf + alpha) / (nf + a0 - yf - alpha))
             - np.log((ym + alpha) / (nm + a0 - ym - alpha)))
    z = delta / np.sqrt(1.0 / (yf + alpha) + 1.0 / (ym + alpha))
    return (pd.DataFrame({"lema": vocab, "frases_mujeres": yf.astype(int), "frases_hombres": ym.astype(int),
                          "z": np.round(z, 2)})
            .sort_values("z", ascending=False).reset_index(drop=True))


def perfil_lexico(textos: Iterable[Tuple[str, Optional[str]]], analyzer, empresa: str = "",
                  top: int = 12, min_total: int = 3) -> dict:
    """Vocabulario más asociado a mujeres y a hombres en un conjunto de textos.

    textos: pares (texto, idioma). Cada palabra cuenta una vez por frase.
    """
    c_mujeres, c_hombres = Counter(), Counter()
    n_mujeres = n_hombres = n_textos = 0
    for texto, idioma in textos:
        try:
            fm, fh = frases_por_genero(texto, analyzer, idioma, empresa)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vocabulario: texto omitido: %s", exc)
            continue
        n_textos += 1
        n_mujeres += len(fm)
        n_hombres += len(fh)
        for lemas in fm:
            c_mujeres.update(set(lemas))
        for lemas in fh:
            c_hombres.update(set(lemas))

    tabla = log_odds_dirichlet(c_mujeres, c_hombres, min_total)
    logger.info("Vocabulario: %d textos, %d frases sobre mujeres, %d sobre hombres, %d lemas comparables",
                n_textos, n_mujeres, n_hombres, len(tabla))
    return {
        "textos": n_textos,
        "frases_mujeres": n_mujeres,
        "frases_hombres": n_hombres,
        "mujeres": tabla[tabla["z"] > 0].head(top).to_dict("records"),
        "hombres": tabla[tabla["z"] < 0].sort_values("z").head(top).to_dict("records"),
        "tabla": tabla,
    }
