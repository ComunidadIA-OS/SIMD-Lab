"""Prueba de extremo a extremo de la ingesta (requiere red).

    PYTHONPATH=. python tests/test_ingesta_rerank.py --query Santander --n 5
    PYTHONPATH=. python tests/test_ingesta_rerank.py --query Apple --context "Apple Inc., empresa tecnológica"

Genera tests/mocks/sample_top_n_docs.json (Contrato A) para la Integrante 2.
Opciones útiles:
    --no-rerank      salta el reranker (no descarga modelos)
    --no-download    no descarga multimedia
    --light-models   usa modelos pequeños para probar en CPU
    --min-score      relevancia mínima para entrar en el Top (por defecto 0.5)
"""
import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # antes de importar src, para que se lean todas las variables

from src.connectors.media_downloader import MediaDownloader
from src.connectors.orchestrator import FederatedSearchOrchestrator
from src.reranker.semantic_ranker import SemanticReranker

LIGHT_EMBED = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LIGHT_CROSS = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


async def main(args):
    t0 = time.perf_counter()
    result = await FederatedSearchOrchestrator(per_source_timeout=60).gather_all_sources(
        args.query, pre_limit=args.pre_limit
    )
    t_search = time.perf_counter() - t0

    print(f"\n== Búsqueda federada: {len(result.documents)} candidatos en {t_search:.1f}s ==")
    for r in result.reports:
        estado = f"{r.n_docs} docs" if r.ok else f"FALLO: {r.error}"
        print(f"  {r.source:<17} {r.seconds:5.1f}s  {estado}")

    docs = result.documents
    if args.no_rerank:
        top = docs[: args.n]
    else:
        kw = {"embed_model": LIGHT_EMBED, "cross_model": LIGHT_CROSS} if args.light_models else {}
        top = SemanticReranker(**kw).rerank(
            args.query, docs, args.n, query_context=args.context, min_score=args.min_score
        )

    if not args.no_download:
        top = await MediaDownloader(base_dir=args.media_dir).download_all(top)

    print(f"\n== Top {len(top)} ==")
    for d in top:
        score = f"{d.relevance_score:.2f}" if d.relevance_score is not None else "  - "
        print(f"  [{score}] {d.source.value:<17} {d.title[:70]}  modalidades={d.available_modalities}")

    out = Path("tests/mocks/sample_top_n_docs.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([d.model_dump() for d in top], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGuardado {out} ({len(top)} documentos)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--query", default="Santander")
    p.add_argument("--context", default=None, help='p.ej. "Banco Santander, entidad financiera"')
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--pre-limit", type=int, default=10)
    p.add_argument("--media-dir", default="/tmp/ramdisk")
    p.add_argument("--min-score", type=float, default=0.5)
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--light-models", action="store_true")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(main(p.parse_args()))"""Prueba de extremo a extremo de la ingesta (requiere red).

    PYTHONPATH=. python tests/test_ingesta_rerank.py --query Santander --n 5
    PYTHONPATH=. python tests/test_ingesta_rerank.py --query Apple --context "Apple Inc., empresa tecnológica"

Genera tests/mocks/sample_top_n_docs.json (Contrato A) para la Integrante 2.
Opciones útiles:
    --no-rerank      salta el reranker (no descarga modelos)
    --no-download    no descarga multimedia
    --light-models   usa modelos pequeños para probar en CPU
    --min-score      relevancia mínima para entrar en el Top (por defecto 0.5)
"""
import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # antes de importar src, para que se lean todas las variables

from src.connectors.media_downloader import MediaDownloader
from src.connectors.orchestrator import FederatedSearchOrchestrator
from src.reranker.semantic_ranker import SemanticReranker

LIGHT_EMBED = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LIGHT_CROSS = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


async def main(args):
    t0 = time.perf_counter()
    result = await FederatedSearchOrchestrator(per_source_timeout=60).gather_all_sources(
        args.query, pre_limit=args.pre_limit
    )
    t_search = time.perf_counter() - t0

    print(f"\n== Búsqueda federada: {len(result.documents)} candidatos en {t_search:.1f}s ==")
    for r in result.reports:
        estado = f"{r.n_docs} docs" if r.ok else f"FALLO: {r.error}"
        print(f"  {r.source:<17} {r.seconds:5.1f}s  {estado}")

    docs = result.documents
    if args.no_rerank:
        top = docs[: args.n]
    else:
        kw = {"embed_model": LIGHT_EMBED, "cross_model": LIGHT_CROSS} if args.light_models else {}
        top = SemanticReranker(**kw).rerank(
            args.query, docs, args.n, query_context=args.context, min_score=args.min_score
        )

    if not args.no_download:
        top = await MediaDownloader(base_dir=args.media_dir).download_all(top)

    print(f"\n== Top {len(top)} ==")
    for d in top:
        score = f"{d.relevance_score:.2f}" if d.relevance_score is not None else "  - "
        print(f"  [{score}] {d.source.value:<17} {d.title[:70]}  modalidades={d.available_modalities}")

    out = Path("tests/mocks/sample_top_n_docs.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([d.model_dump() for d in top], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGuardado {out} ({len(top)} documentos)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--query", default="Santander")
    p.add_argument("--context", default=None, help='p.ej. "Banco Santander, entidad financiera"')
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--pre-limit", type=int, default=10)
    p.add_argument("--media-dir", default="/tmp/ramdisk")
    p.add_argument("--min-score", type=float, default=0.5)
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--light-models", action="store_true")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(main(p.parse_args()))