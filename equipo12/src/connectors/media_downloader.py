"""Descarga de multimedia al volumen efímero, solo para los documentos ya filtrados.

- Imágenes: descarga directa con tope de tamaño.
- Audio/vídeo (solo Internet Archive): ffmpeg lee el stream remoto y se queda con
  los primeros `max_seconds`, sin descargar el fichero completo:
    * audio .wav 16 kHz mono
    * clip .mp4 a 480p
    * fotogramas a 1 fps
YouTube no se descarga (decisión del equipo por Términos de Servicio).
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import List

import aiohttp

from src.common.schemas import MediaSource, UnifiedMediaDocument
from src.connectors.base import DEFAULT_USER_AGENT

logger = logging.getLogger(__name__)

DEFAULT_MEDIA_DIR = os.getenv("MEDIA_DIR", "/tmp/ramdisk")
STREAM_SOURCES = {MediaSource.ARCHIVE}


class MediaDownloader:
    def __init__(
        self,
        base_dir: str = DEFAULT_MEDIA_DIR,
        max_image_bytes: int = 10 * 1024 * 1024,
        max_seconds: int = 300,
        fps: int = 1,
        max_images_per_doc: int = 5,
        concurrency: int = 3,
    ) -> None:
        self.base = Path(base_dir)
        self.max_image_bytes = max_image_bytes
        self.max_seconds = max_seconds
        self.fps = fps
        self.max_images_per_doc = max_images_per_doc
        self._sem = asyncio.Semaphore(concurrency)
        self.ffmpeg = shutil.which("ffmpeg")

    def doc_dir(self, doc: UnifiedMediaDocument) -> Path:
        d = self.base / doc.doc_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    async def _download_image(self, session: aiohttp.ClientSession, url: str, dest: Path) -> bool:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as r:
                r.raise_for_status()
                size = 0
                with open(dest, "wb") as f:
                    async for chunk in r.content.iter_chunked(64 * 1024):
                        size += len(chunk)
                        if size > self.max_image_bytes:
                            raise ValueError("imagen demasiado grande")
                        f.write(chunk)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Imagen no descargada (%s): %s", url, exc)
            dest.unlink(missing_ok=True)
            return False

    async def _ffmpeg(self, *args: str) -> bool:
        if not self.ffmpeg:
            logger.warning("ffmpeg no está instalado: se omite audio/vídeo.")
            return False
        proc = await asyncio.create_subprocess_exec(
            self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *args,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        if proc.returncode != 0:
            logger.warning("ffmpeg falló: %s", err.decode(errors="ignore")[:300])
            return False
        return True

    async def _process_stream(self, doc: UnifiedMediaDocument, url: str, folder: Path, idx: int, is_video: bool) -> None:
        t = str(self.max_seconds)
        wav = folder / f"audio_{idx}.wav"
        if await self._ffmpeg("-t", t, "-i", url, "-vn", "-ac", "1", "-ar", "16000", str(wav)):
            doc.audio_paths.append(str(wav))
        if not is_video:
            return
        clip = folder / f"video_{idx}.mp4"
        if await self._ffmpeg("-t", t, "-i", url, "-vf", "scale=-2:480", "-c:v", "libx264",
                              "-preset", "veryfast", "-c:a", "aac", str(clip)):
            doc.video_paths.append(str(clip))
            frames = folder / f"frames_{idx}"
            frames.mkdir(exist_ok=True)
            if await self._ffmpeg("-i", str(clip), "-vf", f"fps={self.fps}", str(frames / "frame_%04d.jpg")):
                doc.frame_dirs.append(str(frames))

    async def download_document(self, session: aiohttp.ClientSession, doc: UnifiedMediaDocument) -> UnifiedMediaDocument:
        async with self._sem:
            folder = self.doc_dir(doc)
            for i, url in enumerate(doc.image_urls[: self.max_images_per_doc]):
                ext = Path(url.split("?")[0]).suffix.lower() or ".jpg"
                dest = folder / f"image_{i}{ext if len(ext) <= 5 else '.jpg'}"
                if await self._download_image(session, url, dest):
                    doc.image_paths.append(str(dest))
            if doc.source in STREAM_SOURCES:
                for i, url in enumerate(doc.video_urls[:1]):
                    await self._process_stream(doc, url, folder, i, is_video=True)
                if not doc.video_urls:
                    for i, url in enumerate(doc.audio_urls[:1]):
                        await self._process_stream(doc, url, folder, i, is_video=False)
            return doc

    async def download_all(self, docs: List[UnifiedMediaDocument]) -> List[UnifiedMediaDocument]:
        self.base.mkdir(parents=True, exist_ok=True)
        async with aiohttp.ClientSession(headers={"User-Agent": DEFAULT_USER_AGENT}) as session:
            return list(await asyncio.gather(*(self.download_document(session, d) for d in docs)))

    def purge(self, doc_id: str) -> None:
        """Borra la carpeta de un documento (la Integrante 2 lo llama tras analizarlo)."""
        shutil.rmtree(self.base / doc_id, ignore_errors=True)
