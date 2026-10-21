
import json

from src.analyzers.text_analyzer import TextGenderAnalyzer
from src.common.schemas import UnifiedMediaDocument

analyzer = TextGenderAnalyzer()

print("== 1. Frase de prueba ==")
res = analyzer.analyze(
    "La consejera delegada de Iberdrola, Ana Pérez, explicó que la red necesita almacenamiento. "
    "El presidente, Juan García, fue reelegido. «Vamos a invertir», dijo ella.",
    title="Ana Pérez lidera la transición energética", language="es", company_query="Iberdrola")
for m in res.mentions:
    print(f"  {m.text:<15} género={m.gender}  hace_la_acción={m.active}  citada={m.quoted}")
print(" ", res.metrics)

with open("tests/mocks/sample_top_n_docs.json", encoding="utf-8") as f:
    docs = [UnifiedMediaDocument.model_validate(d) for d in json.load(f)]
for d in docs:
    res = analyzer.analyze_document(d)
    mujeres = sum(m.gender == "F" for m in res.mentions)
    hombres = sum(m.gender == "M" for m in res.mentions)
    print(f"\n{d.title}  ({mujeres} menciones de mujeres, {hombres} de hombres)")
    print(" ", res.metrics)