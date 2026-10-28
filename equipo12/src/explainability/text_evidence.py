"""Evidencia del análisis de texto: qué personas ha detectado el analizador y cómo.

Sustituye a LIME de texto. LIME quitaba palabras al azar y medía cómo cambiaba la
salida de un clasificador sustituto basado en listas de palabras, por eso salían
palabras como "la" o "global" con peso, aunque no tienen nada que ver con el género.

Aquí se muestra lo que de verdad usa el índice: cada mención a una persona que el
analizador de texto ha encontrado (nombre, cargo o pronombre), con su género y si
es sujeto de la acción o está citada. Las métricas de texto se calculan solo con esto.
"""
from __future__ import annotations

import html
from collections import Counter
from typing import Optional

COLOR = {"F": "#C2185B", "M": "#1565C0"}
FONDO = {"F": "#FCE4EC", "M": "#E3F2FD"}
NOMBRE_GENERO = {"F": "mujer", "M": "hombre"}
TIPO = {"nombre": "nombre", "cargo": "cargo", "pronombre": "pronombre"}


def explain_text(text: str, analyzer, language: Optional[str] = None, company_query: str = "") -> str:
    """Devuelve HTML con el texto, las menciones resaltadas y un resumen.

    analyzer es el TextGenderAnalyzer del pipeline (el mismo que calcula las métricas).
    """
    resultado = analyzer.analyze(text, title="", language=language, company_query=company_query)
    menciones = sorted([m for m in resultado.mentions if m.gender in ("F", "M")], key=lambda m: m.start)

    # Texto con las menciones resaltadas (sin solapes)
    trozos, cursor = [], 0
    for m in menciones:
        ini, fin = m.start, m.start + len(m.text)
        if ini < cursor or text[ini:fin] != m.text:
            continue
        trozos.append(html.escape(text[cursor:ini]))
        marcas = []
        if m.active:
            marcas.append("sujeto")
        if m.quoted:
            marcas.append("citada" if m.gender == "F" else "citado")
        etiqueta = f"{NOMBRE_GENERO[m.gender]} · {TIPO.get(m.kind, m.kind)}" + (f" · {', '.join(marcas)}" if marcas else "")
        borde = "2px solid" if marcas else "1px dashed"
        trozos.append(
            f"<span title='{html.escape(etiqueta)}' style='background:{FONDO[m.gender]};color:{COLOR[m.gender]};"
            f"border-bottom:{borde} {COLOR[m.gender]};padding:1px 3px;border-radius:3px;font-weight:600;'>"
            f"{html.escape(m.text)}<sup style='font-size:0.65em;margin-left:2px;'>{'/'.join(marcas)}</sup></span>"
        )
        cursor = fin
    trozos.append(html.escape(text[cursor:]))

    # Resumen
    c = Counter(m.gender for m in menciones)
    activas = Counter(m.gender for m in menciones if m.active)
    citadas = Counter(m.gender for m in menciones if m.quoted)

    def fila(nombre, cont):
        f, h = cont.get("F", 0), cont.get("M", 0)
        cuota = f"{100 * f / (f + h):.0f} %" if f + h else "sin dato"
        return f"<tr><td>{nombre}</td><td style='color:{COLOR['F']}'>{f}</td><td style='color:{COLOR['M']}'>{h}</td><td><b>{cuota}</b></td></tr>"

    tabla = (
        "<table style='border-collapse:collapse;font-size:0.85rem;margin-bottom:10px;'>"
        "<tr style='text-align:left;color:#475569'><th style='padding-right:18px'>En este fragmento</th>"
        "<th style='padding-right:18px'>Mujeres</th><th style='padding-right:18px'>Hombres</th><th>Cuota de mujeres</th></tr>"
        + fila("Menciones a personas", c) + fila("Como sujeto de la acción", activas) + fila("Citadas / citados", citadas)
        + "</table>"
    )
    leyenda = (
        f"<div style='font-size:0.78rem;color:#475569;margin-bottom:8px'>"
        f"<span style='color:{COLOR['F']};font-weight:700'>■ Mujer</span> &nbsp; "
        f"<span style='color:{COLOR['M']};font-weight:700'>■ Hombre</span> &nbsp; "
        "Subrayado continuo: es sujeto de la acción o está citada/o. Pasa el ratón por encima para ver el detalle.</div>"
    )
    if not menciones:
        cuerpo = "<p style='color:#64748B'>El analizador no ha encontrado ninguna mención a mujeres ni a hombres en este fragmento, así que no aporta nada al índice.</p>"
    else:
        cuerpo = ""
    return (
        "<div style='font-family:Montserrat,sans-serif'>" + tabla + leyenda + cuerpo
        + "<div style='line-height:1.9;font-size:0.92rem;background:#FFFFFF;border:1px solid #E2E8F0;"
          "border-radius:8px;padding:12px 14px'>" + "".join(trozos) + "</div></div>"
    )
