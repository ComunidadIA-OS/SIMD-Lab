"""Clase base común a todos los conectores de ingesta."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import aiohttp
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.common.schemas import MediaSource, UnifiedMediaDocument

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = os.getenv(
    "WIKIMEDIA_USER_AGENT",
    "SIMD-GenderAuditor/0.1 (UCLM I3A research project; contacto@example.org)",
)


class RetryableHTTPError(Exception):
    """Error HTTP transitorio (429 o 5xx) que merece reintento."""


class BaseMediaConnector(ABC):
    """Interfaz común: buscar en una fuente y devolver documentos normalizados.

    Cada conector solo busca y normaliza metadatos (URLs remotas).
    La descarga de multimedia la hace MediaDownloader después del reranking.
    """

    source: MediaSource
    #: Peticiones simultáneas máximas contra esta fuente
    max_concurrency: int = 4
    #: Pausa mínima entre peticiones (segundos), para respetar la cuota
    min_interval: float = 0.0

    def __init__(
        self,
        session: Optional[aiohttp.ClientSession] = None,
        timeout_s: float = 20.0,
        max_retries: int = 3,
    ) -> None:
        self._external_session = session
        self._session: Optional[aiohttp.ClientSession] = session
        self.timeout = aiohttp.ClientTimeout(total=timeout_s)
        self.max_retries = max_retries
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    # ---------- ciclo de vida de la sesión ----------
    async def __aenter__(self) -> "BaseMediaConnector":
        if self._session is None:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": DEFAULT_USER_AGENT}, timeout=self.timeout
            )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        if self._session is not None and self._external_session is None:
            await self._session.close()
            self._session = None

    # ---------- utilidades ----------
    @staticmethod
    def make_doc_id(source: MediaSource, url: str) -> str:
        return hashlib.sha1(f"{source.value}|{url}".encode("utf-8")).hexdigest()[:16]

    async def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        async with self._lock:
            loop = asyncio.get_running_loop()
            wait = self._last_call + self.min_interval - loop.time()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = loop.time()

    async def get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """GET con reintentos exponenciales ante 429/5xx y errores de red."""
        if self._session is None:
            raise RuntimeError("Usa el conector dentro de 'async with' o pásale una sesión.")
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=max(1, self.min_interval), max=30),
            retry=retry_if_exception_type(
                (RetryableHTTPError, aiohttp.ClientConnectionError, asyncio.TimeoutError)
            ),
            reraise=True,
        ):
            with attempt:
                async with self._semaphore:
                    await self._throttle()
                    async with self._session.get(url, params=params, timeout=self.timeout) as resp:
                        if resp.status == 429 or resp.status >= 500:
                            raise RetryableHTTPError(f"{self.source.value}: HTTP {resp.status}")
                        resp.raise_for_status()
                        try:
                            return await resp.json(content_type=None)
                        except ValueError:
                            text = await resp.text()
                            if "limit requests" in text.lower():
                                await asyncio.sleep(max(self.min_interval, 5))
                                raise RetryableHTTPError(f"{self.source.value}: límite de peticiones")
                            raise

    # ---------- contrato ----------
    @abstractmethod
    async def fetch_top_candidates(self, query: str, limit: int) -> List[UnifiedMediaDocument]:
        """Devuelve hasta `limit` documentos candidatos normalizados."""
