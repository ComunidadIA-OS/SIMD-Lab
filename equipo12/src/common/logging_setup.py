"""Registro (logging) de la auditoría: consola + fichero logs/auditoria.log.

En JupyterLab, Streamlit corre en segundo plano y no se ve su consola: para seguir
una auditoría en directo, abre una terminal y ejecuta
    tail -f /workspace/equipo12/logs/auditoria.log
En Windows:
    Get-Content logs\\auditoria.log -Wait -Tail 30
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

LOG_FILE = os.getenv("LOG_FILE", "logs/auditoria.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
FORMATO = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

# Librerías que escriben mucho y no aportan para seguir la auditoría
RUIDOSAS = ("urllib3", "httpx", "httpcore", "PIL", "matplotlib", "codecarbon", "filelock",
            "huggingface_hub", "sentence_transformers", "transformers", "asyncio", "aiohttp",
            "numba", "tensorflow", "h5py", "fsspec", "watchdog", "streamlit")

_configurado = False


def setup_logging() -> None:
    """Configura el registro una sola vez por proceso (Streamlit re-ejecuta el script en cada clic)."""
    global _configurado
    if _configurado:
        return
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    formato = logging.Formatter(FORMATO, datefmt="%H:%M:%S")

    raiz = logging.getLogger()
    raiz.setLevel(LOG_LEVEL)
    for handler in (logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")):
        handler.setFormatter(formato)
        raiz.addHandler(handler)

    for nombre in RUIDOSAS:
        logging.getLogger(nombre).setLevel(logging.WARNING)
    _configurado = True
    logging.getLogger(__name__).info("Registro activo en %s (nivel %s)", Path(LOG_FILE).resolve(), LOG_LEVEL)
