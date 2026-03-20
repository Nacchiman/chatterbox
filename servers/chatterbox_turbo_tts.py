"""Chatterbox Turbo TTS — LiveKit Agent プラグイン。

tts_server.py (FastAPI) に HTTP ストリーミングリクエストを送り、
受信した PCM チャンクを LiveKit Agent の TTS パイプラインに流し込む。

Usage (LiveKit Agent 内):
    from servers.chatterbox_turbo_tts import ChatterboxTTS

    tts = ChatterboxTTS(
        base_url="http://localhost:8880",
        voice_id="default",
    )

    # VoicePipelineAgent などに渡す
    agent = VoicePipelineAgent(tts=tts, ...)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx
from livekit.agents import APIStatusError, tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

logger = logging.getLogger(__name__)

# サーバーのデフォルト設定
_DEFAULT_BASE_URL = "http://localhost:8880"
_DEFAULT_SAMPLE_RATE = 24000
_DEFAULT_NUM_CHANNELS = 1


@dataclass
class _TTSOptions:
    base_url: str
    voice_id: str
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    chunk_duration_sec: float
    crossfade_duration_sec: float
    mel_overlap: int
    http_timeout: float


class ChatterboxTTS(tts.TTS):
    """Chatterbox Turbo TTS サーバーに接続する LiveKit Agent TTS プラグイン。

    テキスト全体を一括送信し、サーバーからストリーミングで返される
    PCM int16 バイト列を SynthesizedAudio として逐次出力する。
    """

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        voice_id: str = "default",
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        chunk_duration_sec: float = 1.0,
        crossfade_duration_sec: float = 0.04,
        mel_overlap: int = 10,
        http_timeout: float = 120.0,
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
        num_channels: int = _DEFAULT_NUM_CHANNELS,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        self._opts = _TTSOptions(
            base_url=base_url.rstrip("/"),
            voice_id=voice_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            chunk_duration_sec=chunk_duration_sec,
            crossfade_duration_sec=crossfade_duration_sec,
            mel_overlap=mel_overlap,
            http_timeout=http_timeout,
        )
        self._http_client: httpx.AsyncClient | None = None

    @property
    def model(self) -> str:
        return "chatterbox-turbo"

    @property
    def provider(self) -> str:
        return "chatterbox"

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                base_url=self._opts.base_url,
                timeout=httpx.Timeout(self._opts.http_timeout, connect=10.0),
            )
        return self._http_client

    def update_options(
        self,
        *,
        voice_id: str | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        chunk_duration_sec: float | None = None,
        crossfade_duration_sec: float | None = None,
        mel_overlap: int | None = None,
    ) -> None:
        """実行時にオプションを変更する。"""
        if voice_id is not None:
            self._opts.voice_id = voice_id
        if temperature is not None:
            self._opts.temperature = temperature
        if top_k is not None:
            self._opts.top_k = top_k
        if top_p is not None:
            self._opts.top_p = top_p
        if repetition_penalty is not None:
            self._opts.repetition_penalty = repetition_penalty
        if chunk_duration_sec is not None:
            self._opts.chunk_duration_sec = chunk_duration_sec
        if crossfade_duration_sec is not None:
            self._opts.crossfade_duration_sec = crossfade_duration_sec
        if mel_overlap is not None:
            self._opts.mel_overlap = mel_overlap

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return _ChatterboxChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            opts=self._opts,
        )

    async def aclose(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None


class _ChatterboxChunkedStream(tts.ChunkedStream):
    """TTS サーバーからの HTTP ストリーミングレスポンスを ChunkedStream に変換する。"""

    def __init__(
        self,
        *,
        tts: ChatterboxTTS,
        input_text: str,
        conn_options: APIConnectOptions,
        opts: _TTSOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """TTS サーバーにリクエストし、ストリーミングレスポンスを AudioEmitter に流す。"""
        parent_tts: ChatterboxTTS = self._tts  # type: ignore[assignment]
        client = parent_tts._ensure_client()

        payload = {
            "text": self._input_text,
            "voice_id": self._opts.voice_id,
            "temperature": self._opts.temperature,
            "top_k": self._opts.top_k,
            "top_p": self._opts.top_p,
            "repetition_penalty": self._opts.repetition_penalty,
            "chunk_duration_sec": self._opts.chunk_duration_sec,
            "crossfade_duration_sec": self._opts.crossfade_duration_sec,
            "mel_overlap": self._opts.mel_overlap,
        }

        request_id = "unknown"
        sample_rate = _DEFAULT_SAMPLE_RATE
        num_channels = _DEFAULT_NUM_CHANNELS
        emitted_audio = False
        initialized = False

        max_retry = max(0, int(self._conn_options.max_retry))
        attempts = max_retry + 1
        retry_interval = max(0.0, float(self._conn_options.retry_interval))
        request_timeout = (
            float(self._conn_options.timeout)
            if float(self._conn_options.timeout) > 0
            else float(self._opts.http_timeout)
        )
        timeout = httpx.Timeout(request_timeout, connect=min(10.0, request_timeout))

        def _parse_int_header(headers: httpx.Headers, name: str, fallback: int) -> int:
            raw = headers.get(name)
            try:
                return int(raw) if raw is not None else fallback
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid %s header from TTS server: %r; fallback=%s",
                    name,
                    raw,
                    fallback,
                )
                return fallback

        try:
            for attempt in range(attempts):
                try:
                    async with client.stream("POST", "/v1/tts/stream", json=payload, timeout=timeout) as resp:
                        if resp.status_code != 200:
                            body = await resp.aread()
                            raise APIStatusError(
                                message=f"TTS server error: {resp.status_code} {body.decode(errors='replace')}",
                                status_code=resp.status_code,
                                request_id=request_id,
                                body=body.decode(errors="replace"),
                            )

                        # レスポンスヘッダーからメタデータ取得
                        request_id = resp.headers.get("x-request-id", "unknown")
                        sample_rate = _parse_int_header(resp.headers, "x-sample-rate", _DEFAULT_SAMPLE_RATE)
                        num_channels = _parse_int_header(resp.headers, "x-channels", _DEFAULT_NUM_CHANNELS)

                        if not initialized:
                            output_emitter.initialize(
                                request_id=request_id,
                                sample_rate=sample_rate,
                                num_channels=num_channels,
                                mime_type="audio/pcm",
                            )
                            initialized = True

                        # ストリーミングで PCM バイト列を受信し、AudioEmitter に push
                        async for chunk in resp.aiter_bytes(chunk_size=4800):
                            # chunk_size=4800 → 2400 samples (int16) ≒ 0.1s at 24kHz
                            if chunk:
                                output_emitter.push(chunk)
                                emitted_audio = True

                    break
                except APIStatusError as exc:
                    retryable = 500 <= exc.status_code < 600
                    if retryable and not emitted_audio and attempt < attempts - 1:
                        logger.warning(
                            "Retrying TTS request after API error: status=%s attempt=%s/%s",
                            exc.status_code,
                            attempt + 1,
                            attempts,
                        )
                        await asyncio.sleep(retry_interval)
                        continue
                    raise
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    if not emitted_audio and attempt < attempts - 1:
                        logger.warning(
                            "Retrying TTS request after transport error: %s attempt=%s/%s",
                            type(exc).__name__,
                            attempt + 1,
                            attempts,
                        )
                        await asyncio.sleep(retry_interval)
                        continue
                    raise RuntimeError(
                        f"TTS server connection failed after {attempt + 1} attempt(s): {exc}"
                    ) from exc
        finally:
            if initialized:
                output_emitter.flush()
