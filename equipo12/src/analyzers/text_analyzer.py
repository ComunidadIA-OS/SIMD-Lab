"""Análisis de texto: cómo aparecen las mujeres en un texto.

Recibe el texto de un UnifiedMediaDocument y devuelve TextMetrics.

Pasos:
  1. extract_entities_and_coref: busca personas (nombres, cargos como "la directora"
     y pronombres como "ella") y decide si son mujer u hombre.
  2. analyze_syntactic_agency: marca quién es sujeto de un verbo activo (quién "hace").
  3. measure_expert_quotes: marca a quién se cita ("explicó la directora", "según X").
  4. compute_text_metrics: cuenta y devuelve las métricas.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.common.schemas import TextMetrics, TextSample

F, M = "F", "M"   # así se escribe mujer (F) y hombre (M) en todo el archivo
NEUTRO = None     # una proporción que no se puede calcular (p. ej. nadie citado) queda sin dato
MAX_CHARS = 40000  # los artículos muy largos se recortan para evitar tiempos de análisis excesivos

# Cargos y palabras de persona por parejas (masculino/femenino)
_PAREJAS_ES = """director directora presidente presidenta consejero consejera ministro ministra
fundador fundadora ingeniero ingeniera científico científica investigador investigadora
profesor profesora experto experta jefe jefa empresario empresaria ejecutivo ejecutiva
directivo directiva socio socia trabajador trabajadora empleado empleada abogado abogada
diputado diputada alcalde alcaldesa secretario secretaria doctor doctora médico médica
autor autora escritor escritora actor actriz candidato candidata rey reina hombre mujer
señor señora padre madre hijo hija hermano hermana esposo esposa niño niña
operador operadora técnico técnica mecánico mecánica conductor conductora administrador
administradora entrenador entrenadora locutor locutora arquitecto arquitecta astrónomo astrónoma
barbero barbera camarero camarera capitán capitana encargado encargada químico química
coreógrafo coreógrafa obrero obrera cocinero cocinera conservador conservadora bailarín bailarina
diseñador diseñadora editor editora vendedor vendedora grabador grabadora animador animadora
constructor constructora educador educadora cazador cazadora juez jueza maestro maestra
bibliotecario bibliotecaria enfermero enfermera matemático matemática lector lectora músico música
reportero reportera asesor asesora farmacéutico farmacéutica promotor promotora psicólogo psicóloga
desarrollador desarrolladora estadístico estadística redactor redactora veterinario veterinaria"""

# Igual en masculino y femenino: el género lo dice el artículo
_SIN_GENERO_ES = """portavoz periodista líder gerente representante analista economista
dirigente responsable especialista"""

# Parejas de persona en inglés
_PAREJAS_EN = """chairman chairwoman spokesman spokeswoman businessman businesswoman
actor actress king queen man woman boy girl father mother husband wife son daughter
brother sister
congressman congresswoman councilman councilwoman policeman policewoman salesman saleswoman
statesman stateswoman prince princess duke duchess emperor empress uncle aunt nephew niece
grandfather grandmother grandson granddaughter stepfather stepmother boyfriend girlfriend
widower widow"""
# Palabras que también nombran una empresa, una disciplina, un objeto o un órgano. Solo cuentan pegadas a un nombre de persona ("la química Margarita Salas").
AMBIGUAS = {
    "es": set("""operador operadora constructora promotor promotora desarrollador desarrolladora
        farmacéutica técnica mecánica química matemática estadística música grabadora cazadora
        lector lectora conservador conservadora directiva ejecutiva""".split()),
    "en": set(),
}


def _parejas(texto: str) -> dict:
    """Convierte "director directora ..." en {"director": "M", "directora": "F", ...}"""
    palabras = texto.split()
    return {**{w: M for w in palabras[0::2]}, **{w: F for w in palabras[1::2]}}


# Todos los cargos de cada idioma: palabra -> género ("directora" -> F).
ROLES = {
    "es": {**{w: None for w in _SIN_GENERO_ES.split()}, **_parejas(_PAREJAS_ES)},
    "en": _parejas(_PAREJAS_EN),
}
# Artículos que dicen el género del cargo que acompañan
ARTICULOS_ES = {"la": F, "una": F, "esta": F, "el": M, "un": M, "este": M, "del": M, "al": M}
# Tratamientos delante del nombre ("Sra. García", "Mr Brown")
TRATAMIENTOS = {
    "es": {"sra": F, "señora": F, "doña": F, "sr": M, "señor": M, "don": M},
    "en": {"mrs": F, "ms": F, "miss": F, "lady": F, "mr": M, "sir": M},
}
# Pronombres de persona con género
PRONOMBRES = {
    "es": {"ella": F, "él": M},
    "en": {"she": F, "her": F, "herself": F, "he": M, "him": M, "his": M, "himself": M},
}
# Verbos que introducen lo que alguien dice, en forma de diccionario ("dijo" -> "decir").
# Si una persona es sujeto de uno de ellos, se la está citando.
VERBOS_HABLA = {
    "es": set("""decir afirmar explicar señalar asegurar declarar indicar añadir comentar apuntar
        subrayar destacar advertir reconocer manifestar precisar recordar insistir anunciar opinar
        confirmar defender lamentar criticar responder""".split()),
    "en": set("""say tell explain add note state claim argue warn announce comment insist suggest
        admit confirm report describe reveal respond reply""".split()),
}



@dataclass
class Mention:
    """Una vez que se nombra a una persona en el texto (una "ficha" por mención)."""
    kind: str                 # "nombre", "cargo" o "pronombre"
    text: str                 
    gender: Optional[str]     
    sent: int                 
    token_i: int              
    start: int               
    cluster: str = ""         # misma persona = mismo cluster ("ana perez")
    votes: list = field(default_factory=list)   
    active: bool = False      
    verb: Optional[str] = None  
    quoted: bool = False      
    in_title: bool = False    


@dataclass
class TextResult:
    """Lo que devuelve analyze()."""
    metrics: Optional[TextMetrics]   # las 10 métricas (None si el texto no nombra a nadie)
    samples: List[TextSample]        
    mentions: List[Mention]          # todas las menciones encontradas 



def _ratio(parte: float, total: float) -> Optional[float]:
    return round(parte / total, 4) if total > 0 else NEUTRO


def _decidir(votos: list) -> Optional[str]:
    if votos.count(F) == votos.count(M):
        return None
    return F if votos.count(F) > votos.count(M) else M


# ===================================================================== métricas
def compute_text_metrics(mentions, roles, frases, titular_con_mujer) -> Optional[TextMetrics]:
    """Cuenta las menciones de las frases indicadas y devuelve las métricas.

    mentions: todas las menciones (Mention).
    roles: cargos encontrados, como (género, número de frase).
    frases: números de las frases que se cuentan.
    titular_con_mujer: True si en el titular sale alguna mujer.
    """
    frases = set(frases)
    con_genero = [m for m in mentions if m.gender and m.sent in frases]
    if not con_genero:
        return None   # el texto no nombra a nadie: no se puntúa
    mujeres = [m for m in con_genero if m.gender == F]

    # Cada frase es "femenina" si nombra más mujeres que hombres, y al revés
    por_frase = defaultdict(lambda: [0, 0])   
    for m in con_genero:
        por_frase[m.sent][0 if m.gender == F else 1] += 1
    fem = sum(1 for s in frases if por_frase[s][0] > por_frase[s][1])
    mas = sum(1 for s in frases if por_frase[s][1] > por_frase[s][0])

    activas = [m for m in con_genero if m.active]     
    citadas = [m for m in con_genero if m.quoted]     
    verbos_f = {m.verb for m in activas if m.gender == F}   
    verbos_m = {m.verb for m in activas if m.gender == M}  
    cargos = [g for g, s in roles if s in frases]     

    return TextMetrics(
        # Qué parte de las frases nombra más mujeres / más hombres / ni una cosa ni otra
        sentence_female_ratio=round(fem / len(frases), 4),
        sentence_male_ratio=round(mas / len(frases), 4),
        sentence_neutral_ratio=round((len(frases) - fem - mas) / len(frases), 4),
        lexical_root_diversity=_ratio(len(verbos_f), len(verbos_f) + len(verbos_m)),
        feminine_suffix_ratio=_ratio(cargos.count(F), len(cargos)),
        female_mention_frequency=len(mujeres),
        relative_frequency_score=_ratio(len(mujeres), len(con_genero)),
        syntactic_agency_ratio=_ratio(sum(m.gender == F for m in activas), len(activas)),
        expert_quote_ratio=_ratio(sum(m.gender == F for m in citadas), len(citadas)),
        headline_presence=1.0 if titular_con_mujer else 0.0,
    )


class TextGenderAnalyzer:
    """Analizador de texto. Se crea una vez (carga los modelos) y se usa con muchos textos."""

    # Modelo de spaCy de cada idioma
    MODELOS = {"es": "es_core_news_md", "en": "en_core_web_md"}

    def __init__(self):
        import spacy
        import gender_guesser.detector as gg
        self.nlp = {lang: spacy.load(nombre) for lang, nombre in self.MODELOS.items()}
        self.nombres = gg.Detector(case_sensitive=False)   # "Ana" -> female, "Juan" -> male

    def analyze_document(self, doc) -> TextResult:
        """Analiza un UnifiedMediaDocument."""
        return self.analyze(doc.raw_text or "", doc.title, doc.language, doc.company_query)

    def analyze(self, text: str, title: str = "", language: str = None, company_query: str = "") -> TextResult:
        """Analiza un texto con su titular y devuelve métricas, frases de ejemplo y menciones."""
        lang = self._idioma(language, text)
        if lang is None or not (text or title):
            return TextResult(None, [], [])

        text = text[:MAX_CHARS]
        if title and not text.startswith(title):
            text = f"{title}.\n\n{text}"
        largo_titular = len(title) + 1 if title else 0   
        doc = self.nlp[lang](text)   

        mentions, roles = self.extract_entities_and_coref(doc, lang, company_query)   
        self.analyze_syntactic_agency(doc, mentions)                                 
        self.measure_expert_quotes(doc, mentions, lang)                               

  
        for m in mentions:
            m.in_title = m.start < largo_titular
        titular_con_mujer = any(m.in_title and m.gender == F for m in mentions)
        n_frases = len(list(doc.sents))
        metrics = compute_text_metrics(mentions, roles, range(n_frases), titular_con_mujer)   # paso 4
        samples = self._ejemplos(doc, mentions, roles, titular_con_mujer) if metrics else []
        return TextResult(metrics, samples, mentions)

    def extract_entities_and_coref(self, doc, lang, company_query=""):
        """Busca las personas del texto y su género.

        Tres formas de nombrar a una persona:
          a) por su nombre ("Ana Pérez"), b) por un pronombre ("ella"),
          c) por un cargo suelto ("la directora explicó...").
        """
        roles_lex, tratamientos, pronombres = ROLES[lang], TRATAMIENTOS[lang], PRONOMBRES[lang]
        frase_de = {t.i: k for k, s in enumerate(doc.sents) for t in s}   # palabra -> nº de frase
        empresa = (company_query or "").lower()
        mentions, roles, usados = [], [], set()   

        # a) Personas con nombre ("Ana Pérez", "Mr Brown")
        personas = []
        for ent in doc.ents:
            if ent.label_ not in ("PER", "PERSON") or (empresa and empresa in ent.text.lower()):
                continue
            votos = []
            for t in doc[max(ent.start - 2, 0): ent.end]:          
                g = tratamientos.get(t.lower_.strip("."))
                if g:
                    votos += [g] * 3
            cargos = self._cargos_pegados(ent, roles_lex)          
            for r in cargos:
                g = self._genero_cargo(r, lang)
                if g and r.i not in usados:
                    votos += [g] * 3
                    roles.append((g, frase_de[r.i]))
                usados.add(r.i)
            palabras = [t for t in ent if t.is_alpha and t.lower_.strip(".") not in tratamientos
                        and not (t.lower_ in roles_lex and (t.i == ent.start or t.text[0].islower()))]
            if len(palabras) >= 2:                                 
                g = self._genero_nombre(palabras[0].text)
                if g:
                    votos.append(g)
            nucleo = self._nucleo(ent, cargos)   
            personas.append(Mention("nombre", ent.text, _decidir(votos), frase_de[nucleo.i], nucleo.i,
                                    ent.start_char, " ".join(t.lower_ for t in palabras), votos))
        self._misma_persona_mismo_genero(personas)
        mentions += personas
        en_personas = {t.i for ent in doc.ents if ent.label_ in ("PER", "PERSON") for t in ent}

        for tok in doc:
            # b) Pronombres ("ella", "she")
            if tok.pos_ == "PRON" and tok.lower_ in pronombres:
                mentions.append(Mention("pronombre", tok.text, pronombres[tok.lower_],
                                        frase_de[tok.i], tok.i, tok.idx))
            # c) Cargos sueltos ("la directora explicó...")
            elif (tok.pos_ == "NOUN" and tok.lower_ in roles_lex and tok.lower_ not in AMBIGUAS[lang]
                  and tok.i not in usados and tok.i not in en_personas
                  and tok.morph.get("Number") != ["Plur"]):
                g = self._genero_cargo(tok, lang)
                if g:
                    roles.append((g, frase_de[tok.i]))
                    mentions.append(Mention("cargo", tok.text, g, frase_de[tok.i], tok.i, tok.idx))

        mentions.sort(key=lambda m: m.start)   
        return mentions, roles

    def _genero_cargo(self, tok, lang):
        if lang == "es":
            for c in tok.children:
                if c.lower_ in ARTICULOS_ES:
                    return ARTICULOS_ES[c.lower_]
        return ROLES[lang].get(tok.lower_)

    def _genero_nombre(self, nombre):
        """Solo nombres claramente de mujer o de hombre; 'mostly_*' o ambiguos -> desconocido."""
        r = self.nombres.get_gender(nombre)
        return F if r == "female" else M if r == "male" else None

    @staticmethod
    def _cargos_pegados(ent, roles_lex):
        """Cargos unidos a la persona: 'la presidenta Ana Botín', 'Ana Botín, presidenta del banco'."""
        raiz = ent.root   
        candidatos = [t for t in ent if t.lower_ in roles_lex and (t.i == ent.start or t.text[0].islower())]
        if raiz.dep_ in ("appos", "flat", "compound") and raiz.head.i != raiz.i:
            candidatos.append(raiz.head)
        candidatos += [c for c in raiz.children if c.dep_ in ("appos", "compound")]
        return [t for t in candidatos if t.lower_ in roles_lex and t.pos_ in ("NOUN", "PROPN")
                and t.morph.get("Number") != ["Plur"]]

    @staticmethod
    def _nucleo(ent, cargos):
        """En 'la presidenta Ana Botín dijo...' el sujeto de 'dijo' es 'presidenta', no 'Ana'."""
        raiz = ent.root
        for r in cargos:
            if raiz.head.i == r.i and raiz.dep_ in ("appos", "flat", "compound"):
                return r
        return raiz

    @staticmethod
    def _misma_persona_mismo_genero(personas):
        """'Ana Botín' y luego 'Botín': se unen por el apellido y comparten el género."""
        completas = {m.cluster for m in personas if " " in m.cluster}
        for m in personas:
            if " " not in m.cluster:
                encaja = [c for c in completas if m.cluster in c.split()]
                if len(encaja) == 1:
                    m.cluster = encaja[0]
        votos = defaultdict(list)
        for m in personas:
            votos[m.cluster] += m.votes
        for m in personas:
            m.gender = _decidir(votos[m.cluster]) or m.gender

    def analyze_syntactic_agency(self, doc, mentions):
        """Marca como activa la mención que es sujeto de un verbo en voz activa ("Ana explicó")."""
        for m in mentions:
            tok = doc[m.token_i]
            verbo = tok.head  
            if tok.dep_ == "nsubj" and verbo.pos_ == "VERB" and not self._es_pasiva(verbo):
                m.active, m.verb = True, verbo.lemma_.lower()

    @staticmethod
    def _es_pasiva(verbo):
        for c in verbo.children:
            if c.dep_ in ("aux:pass", "auxpass", "nsubj:pass", "nsubjpass"):
                return True
            if c.dep_.startswith("aux") and c.lemma_ in ("ser", "be") and "Part" in verbo.morph.get("VerbForm"):
                return True
        return False

    def measure_expert_quotes(self, doc, mentions, lang):
        for m in mentions:
            tok = doc[m.token_i]
            antes = " ".join(t.lower_ for t in doc[max(tok.i - 3, 0): tok.i])   
            if (tok.dep_ == "nsubj" and tok.head.lemma_.lower() in VERBOS_HABLA[lang]) \
                    or "según" in antes or "according to" in antes:
                m.quoted = True

    @staticmethod
    def _idioma(language, text):
        """'es', 'Spanish', 'spa', 'en', 'English'... -> 'es' / 'en'. Si no viene, se adivina."""
        if language:
            return {"es": "es", "sp": "es", "en": "en"}.get(language.lower()[:2])
        # Sin idioma: inglés si "the" sale más que "el" + "la"
        texto = f" {text[:3000].lower()} "
        return "en" if texto.count(" the ") > texto.count(" el ") + texto.count(" la ") else "es"

    @staticmethod
    def _ejemplos(doc, mentions, roles, titular_con_mujer):
        """Frases de ejemplo para el dashboard: hasta 3 con mujeres y 2 solo con hombres."""
        frases = list(doc.sents)
        con_mujer = sorted({m.sent for m in mentions if m.gender == F and not m.in_title})
        con_hombre = sorted({m.sent for m in mentions if m.gender == M and not m.in_title} - set(con_mujer))
        return [TextSample(text=frases[s].text.strip()[:600],
                           metrics=compute_text_metrics(mentions, roles, [s], titular_con_mujer))
                for s in sorted(con_mujer[:3] + con_hombre[:2])]


   
    def predict_proba(self, texts, language="es"):
        """[[P(rol secundario), P(rol protagonista)], ...] para LimeTextExplainer.

        LIME quita palabras del texto y mira cómo cambia esta puntuación para decir qué
        palabras pesan más. La puntuación es la media de cuatro métricas (frecuencia,
        agencia, citas y frases), o 0.05 si el texto no nombra a nadie.
        """
        lang = self._idioma(language, "")
        salida = []
        for doc in self.nlp[lang].pipe(list(texts)):   # pipe: analiza muchos textos de golpe
            mentions, roles = self.extract_entities_and_coref(doc, lang)
            self.analyze_syntactic_agency(doc, mentions)
            self.measure_expert_quotes(doc, mentions, lang)
            tm = compute_text_metrics(mentions, roles, range(len(list(doc.sents))), False)
            valores = [] if tm is None else [v for v in (tm.relative_frequency_score, tm.syntactic_agency_ratio,
                                                           tm.expert_quote_ratio, tm.sentence_female_ratio) if v is not None]
            p = float(np.mean(valores)) if valores else 0.05
            salida.append([1 - p, p])
        return np.array(salida)
