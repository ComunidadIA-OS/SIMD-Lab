"""
Elimina la multimedia descargada de un documento en cuanto se ha analizado.
El objetivo es no conservar fotos, audios ni vídeos de personas, solo quedan las métricas y los resultados del análisis.
"""
from __future__ import annotations

import gc
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


class EphemeralPrivacyScrubber:
    """Borra la multimedia de un documento, solo dentro de su carpeta MEDIA_DIR/<doc_id>."""

    def __init__(self, base_dir: str = None):
        self.base = Path(base_dir or os.getenv("MEDIA_DIR", "/tmp/ramdisk")).resolve()

    def purge_biometric_tensors(self, doc) -> dict:
        """Borra audios, vídeos, imágenes y fotogramas de un UnifiedMediaDocument."""
        carpeta = self.base / doc.doc_id   
        carpeta_valida = carpeta.parent == self.base and doc.doc_id not in ("", ".", "..")
        borrados, no_borrados = 0, []   # cuántos archivos se han borrado y cuáles se han dejado

        # Las rutas que trae el documento
        for ruta in [*doc.audio_paths, *doc.video_paths, *doc.image_paths, *doc.frame_dirs]:
            p = Path(ruta)
            if not (p.exists() or p.is_symlink()):
                continue                         
            if carpeta_valida and self._esta_dentro(p, carpeta):
                borrados += self._borrar(p)
            else:
                no_borrados.append(str(p))       

        # La carpeta del documento entera 
        if carpeta_valida and (carpeta.exists() or carpeta.is_symlink()):
            borrados += self._borrar(carpeta)

        doc.audio_paths, doc.video_paths, doc.image_paths, doc.frame_dirs = [], [], [], []
        gc.collect()                 
        self._vaciar_memoria_gpu()   
        return {
            "doc_id": doc.doc_id,
            "ficheros_borrados": borrados,
            "no_borrados": no_borrados,
            "completo": not no_borrados,    
            "fecha_utc": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _esta_dentro(p: Path, carpeta: Path) -> bool:
        if carpeta.is_symlink():
            return False  
        real = p.parent.resolve() / p.name
        return real != carpeta and real.is_relative_to(carpeta)

    @staticmethod
    def _borrar(p: Path) -> int:
        """Borra un fichero, un enlace o una carpeta. Devuelve cuántos había."""
        if p.is_symlink() or p.is_file():
            p.unlink()   
            return 1
        n = sum(len(ficheros) for _, _, ficheros in os.walk(p, followlinks=False))
        shutil.rmtree(p)   
        return n

    @staticmethod
    def _vaciar_memoria_gpu():
        """Si hay tarjeta gráfica, vacía la memoria que guardaba PyTorch."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass   
