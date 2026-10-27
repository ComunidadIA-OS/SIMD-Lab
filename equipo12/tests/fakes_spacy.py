"""Documentos de spaCy simulados para probar text_analyzer sin cargar el modelo.
"""
class Morph:
    def __init__(self, d): self.d = d or {}
    def get(self, k): return [self.d[k]] if k in self.d else []

class Tok:
    def __init__(self, doc, i, text, lemma, pos, dep, head, morph=None, tag=""):
        self.doc, self.i, self.text, self.lemma_, self.pos_, self.dep_ = doc, i, text, lemma or text.lower(), pos, dep
        self._head, self.morph, self.tag_ = head, Morph(morph), tag
        self.lower_ = text.lower(); self.is_alpha = text.isalpha(); self.idx = 0
    @property
    def head(self): return self.doc.toks[self._head]
    @property
    def children(self): return [t for t in self.doc.toks if t._head == self.i and t.i != self.i]
    def __repr__(self): return self.text

class Span:
    def __init__(self, doc, start, end, label=""):
        self.doc, self.start, self.end, self.label_ = doc, start, end, label
    def __iter__(self): return iter(self.doc.toks[self.start:self.end])
    @property
    def start_char(self): return self.doc.toks[self.start].idx
    @property
    def end_char(self): t = self.doc.toks[self.end-1]; return t.idx + len(t.text)
    @property
    def text(self): return self.doc.text[self.start_char:self.end_char]
    @property
    def root(self):
        for t in self:
            if not (self.start <= t._head < self.end) or t._head == t.i: return t
        return self.doc.toks[self.start]

class Doc:
    def __init__(self, sents_spec, ents):
        self.toks, pos, sent_bounds = [], 0, []
        for sent in sents_spec:
            s0 = len(self.toks)
            for (text, lemma, p, dep, head, morph) in sent:
                self.toks.append(Tok(self, len(self.toks), text, lemma, p, dep, head, morph))
            sent_bounds.append((s0, len(self.toks)))
        out = []
        for t in self.toks:
            t.idx = pos; out.append(t.text); pos += len(t.text) + 1
        self.text = " ".join(out)
        self._sents = [Span(self, a, b) for a, b in sent_bounds]
        self.ents = [Span(self, a, b, l) for a, b, l in ents]
    def __len__(self): return len(self.toks)
    def __getitem__(self, k):
        if isinstance(k, slice): return Span(self, k.start, k.stop)
        return self.toks[k]
    @property
    def sents(self): return iter(self._sents)

Fem, Masc = {"Gender": "Fem", "Number": "Sing"}, {"Gender": "Masc", "Number": "Sing"}
def toy_doc_es():
    S0 = [("Ana","Ana","PROPN","nsubj",2,None),("Pérez","Pérez","PROPN","flat",0,None),
          ("lidera","liderar","VERB","ROOT",2,None),("la","el","DET","det",4,Fem),
          ("transición",None,"NOUN","obj",2,Fem),("energética",None,"ADJ","amod",4,Fem),(".",None,"PUNCT","punct",2,None)]
    S1 = [("La","el","DET","det",8,Fem),("consejera",None,"NOUN","nsubj",16,Fem),("delegada",None,"ADJ","amod",8,Fem),
          ("de",None,"ADP","case",11,None),("Iberdrola",None,"PROPN","nmod",8,None),(",",None,"PUNCT","punct",13,None),
          ("Ana",None,"PROPN","appos",8,None),("Pérez",None,"PROPN","flat",13,None),(",",None,"PUNCT","punct",13,None),
          ("explicó","explicar","VERB","ROOT",16,None),("que",None,"SCONJ","mark",20,None),("la","el","DET","det",19,Fem),
          ("red",None,"NOUN","nsubj",20,Fem),("necesita","necesitar","VERB","ccomp",16,None),
          ("almacenamiento",None,"NOUN","obj",20,Masc),(".",None,"PUNCT","punct",16,None)]
    S2 = [("El","el","DET","det",24,Masc),("presidente",None,"NOUN","nsubj",30,Masc),(",",None,"PUNCT","punct",26,None),
          ("Juan",None,"PROPN","appos",24,None),("García",None,"PROPN","flat",26,None),(",",None,"PUNCT","punct",26,None),
          ("fue","ser","AUX","aux",30,None),("reelegido","reelegir","VERB","ROOT",30,{"VerbForm":"Part","Gender":"Masc","Number":"Sing"}),
          (".",None,"PUNCT","punct",30,None)]
    S3 = [("«",None,"PUNCT","punct",33,None),("Vamos","ir","VERB","ccomp",38,None),("a",None,"ADP","mark",35,None),
          ("invertir",None,"VERB","xcomp",33,None),("»",None,"PUNCT","punct",33,None),(",",None,"PUNCT","punct",38,None),
          ("dijo","decir","VERB","ROOT",38,None),("ella","él","PRON","nsubj",38,Fem),(".",None,"PUNCT","punct",38,None)]
    return Doc([S0,S1,S2,S3], [(0,2,"PER"),(11,12,"ORG"),(13,15,"PER"),(26,28,"PER")])

def toy_doc_es_2():
    """'Según la directora , Botín fue nombrada presidenta . Ana Duque ha sido nombrada jefa .'"""
    S0 = [("Según",None,"ADP","case",2,None),("la","el","DET","det",2,Fem),("directora",None,"NOUN","obl",6,Fem),
          (",",None,"PUNCT","punct",2,None),("Botín",None,"PROPN","nsubj",6,None),("fue","ser","AUX","aux",6,None),
          ("nombrada","nombrar","VERB","ROOT",6,{"VerbForm":"Part","Gender":"Fem","Number":"Sing"}),
          ("presidenta",None,"NOUN","xcomp",6,Fem),(".",None,"PUNCT","punct",6,None)]
    S1 = [("Ana",None,"PROPN","nsubj",13,None),("Duque",None,"PROPN","flat",9,None),("ha","haber","AUX","aux",13,None),
          ("sido","ser","AUX","aux",13,None),("nombrada","nombrar","VERB","ROOT",13,{"VerbForm":"Part","Gender":"Fem","Number":"Sing"}),
          ("jefa",None,"NOUN","xcomp",13,Fem),(".",None,"PUNCT","punct",13,None)]
    return Doc([S0,S1], [(4,5,"PER"),(9,11,"PER")])

def toy_doc_en():
    S0 = [("Jane",None,"PROPN","compound",1,None),("Smith",None,"PROPN","nsubj",6,None),(",",None,"PUNCT","punct",1,None),
          ("the",None,"DET","det",4,None),("CEO",None,"NOUN","appos",1,None),(",",None,"PUNCT","punct",1,None),
          ("said","say","VERB","ROOT",6,None),("profits","profit","NOUN","nsubj",8,{"Number":"Plur"}),("rose","rise","VERB","ccomp",6,None),
          (".",None,"PUNCT","punct",6,None)]
    S1 = [("According","accord","VERB","prep",16,None),("to",None,"ADP","prep",10,None),("Mr",None,"PROPN","compound",13,None),
          ("Brown",None,"PROPN","pobj",11,None),(",",None,"PUNCT","punct",16,None),("sales","sale","NOUN","nsubj",16,None),
          ("fell","fall","VERB","ROOT",16,None),(".",None,"PUNCT","punct",16,None)]
    S2 = [("She","she","PRON","nsubjpass",20,None),("was","be","AUX","auxpass",20,None),("named","name","VERB","ROOT",20,None),
          ("chairwoman",None,"NOUN","oprd",20,None),(".",None,"PUNCT","punct",20,None)]
    return Doc([S0,S1,S2], [(0,2,"PERSON"),(13,14,"PERSON")])

class FakeNames:
    def get_gender(self, n): return {"jane": "female", "john": "male"}.get(n.lower(), "unknown")
