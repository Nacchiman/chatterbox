"""Chatterbox Turbo ストリーミング TTS サーバー。

Chatterbox Turbo モデルを GPU 上でロードし、
HTTP ストリーミングで PCM 音声チャンクを逐次送出する FastAPI サーバー。

Usage:
    cd servers/
    uvicorn tts_server:app --host 0.0.0.0 --port 8880
    # または
    python tts_server.py
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8880

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TTSRequest(BaseModel):
    """POST /v1/tts/stream のリクエストボディ。"""

    text: str = Field(..., min_length=1, description="合成するテキスト")
    voice_id: str = Field("default", description="使用する voice_id (default = 組み込み音声)")
    temperature: float = Field(0.8, ge=0.01, le=5.0)
    top_k: int = Field(1000, ge=1)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.2, ge=1.0, le=3.0)
    chunk_duration_sec: float = Field(1.0, ge=0.25, le=5.0)
    crossfade_duration_sec: float = Field(
        0.04, ge=0.0, le=1.0,
        description="チャンク境界のクロスフェード秒数（プツプツ音緩和、0で無効）",
    )
    mel_overlap: int = Field(
        10, ge=0,
        description="HiFiGAN 用 mel フレーム overlap 数（チャンク境界の連続性向上）",
    )


class VoiceInfo(BaseModel):
    voice_id: str
    name: str
    registered_at: float


class VoiceListResponse(BaseModel):
    voices: list[VoiceInfo]


# ---------------------------------------------------------------------------
# Voice registry (voice_id -> pre-computed conditionals)
# ---------------------------------------------------------------------------


@dataclass
class VoiceEntry:
    voice_id: str
    name: str
    registered_at: float
    audio_path: str | None = None  # None for "default"


class VoiceRegistry:
    """登録済みボイスの条件付けを管理する。"""

    def __init__(self) -> None:
        self._entries: dict[str, VoiceEntry] = {}
        self._conds_cache: dict[str, object] = {}  # voice_id -> Conditionals
        self._upload_dir = Path("voice_uploads")
        self._upload_dir.mkdir(exist_ok=True)

    def register_default(self) -> None:
        self._entries["default"] = VoiceEntry(
            voice_id="default",
            name="Default",
            registered_at=time.time(),
            audio_path=None,
        )

    def add(self, voice_id: str, name: str, audio_path: str) -> VoiceEntry:
        entry = VoiceEntry(
            voice_id=voice_id,
            name=name,
            registered_at=time.time(),
            audio_path=audio_path,
        )
        self._entries[voice_id] = entry
        # 既存のキャッシュを無効化
        self._conds_cache.pop(voice_id, None)
        return entry

    def get(self, voice_id: str) -> VoiceEntry | None:
        return self._entries.get(voice_id)

    def list_all(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(
                voice_id=e.voice_id,
                name=e.name,
                registered_at=e.registered_at,
            )
            for e in self._entries.values()
        ]

    def set_conds(self, voice_id: str, conds: object) -> None:
        self._conds_cache[voice_id] = conds

    def get_conds(self, voice_id: str) -> object | None:
        return self._conds_cache.get(voice_id)


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_model = None
_voice_registry = VoiceRegistry()
_inference_lock = asyncio.Lock()


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    device = _detect_device()
    logger.info("Loading Chatterbox Turbo on %s ...", device)
    _model = ChatterboxTurboTTS.from_pretrained(device=device)
    logger.info("Model loaded successfully (sample_rate=%d)", _model.sr)

    # デフォルトボイスを登録
    _voice_registry.register_default()
    if _model.conds is not None:
        _voice_registry.set_conds("default", _model.conds)

    yield

    logger.info("Shutting down TTS server")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Chatterbox Turbo TTS Server",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/v1/voice/list", response_model=VoiceListResponse)
async def list_voices():
    return VoiceListResponse(voices=_voice_registry.list_all())


@app.post("/v1/voice/register")
async def register_voice(
    file: UploadFile = File(...),
    name: str = Form(""),
    voice_id: str = Form(""),
):
    """参照音声ファイルをアップロードし、voice_id を発行して条件付けを事前計算する。"""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not voice_id:
        voice_id = str(uuid.uuid4())[:8]
    if not name:
        name = voice_id

    # ファイル保存
    upload_path = _voice_registry._upload_dir / f"{voice_id}.wav"
    content = await file.read()
    with open(upload_path, "wb") as f:
        f.write(content)

    # 条件付けを事前計算 (GPU 排他制御下)
    async with _inference_lock:
        await asyncio.to_thread(_prepare_voice, voice_id, str(upload_path))

    entry = _voice_registry.add(voice_id, name, str(upload_path))
    return JSONResponse(
        content={
            "voice_id": entry.voice_id,
            "name": entry.name,
            "registered_at": entry.registered_at,
        }
    )


def _prepare_voice(voice_id: str, audio_path: str) -> None:
    """ブロッキングで条件付けを計算し、レジストリにキャッシュする。"""
    assert _model is not None

    _model.prepare_conditionals(audio_path)
    # prepare_conditionals は self.conds を書き換えるので、それをコピーして保存
    conds_copy = copy.deepcopy(_model.conds)
    _voice_registry.set_conds(voice_id, conds_copy)


@app.post("/v1/tts/stream")
async def tts_stream(req: TTSRequest):
    """テキスト一括入力 → PCM ストリーミング出力。

    レスポンスは application/octet-stream で、
    生の PCM int16 little-endian (24kHz, mono) バイト列を連続送出する。
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    voice_entry = _voice_registry.get(req.voice_id)
    if voice_entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"voice_id '{req.voice_id}' not found. Register it first via /v1/voice/register",
        )

    sample_rate = _model.sr  # 24000

    return StreamingResponse(
        _stream_pcm(req, voice_entry),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(sample_rate),
            "X-Sample-Width": "2",  # int16 = 2 bytes
            "X-Channels": "1",
            "X-Request-Id": str(uuid.uuid4()),
            "Cache-Control": "no-cache",
        },
    )


async def _stream_pcm(
    req: TTSRequest, voice_entry: VoiceEntry
) -> AsyncGenerator[bytes, None]:
    """generate_stream() をラップし、PCM バイト列を非同期に yield する。"""
    assert _model is not None

    async with _inference_lock:
        # ブロッキングジェネレータを非同期で回す
        gen_iter = iter(
            await asyncio.to_thread(
                _start_generate_stream,
                req,
                voice_entry,
            )
        )

        while True:
            try:
                chunk = await asyncio.to_thread(_next_chunk, gen_iter)
            except _StopIterationSentinel:
                break

            sr, wav_np = chunk
            # float32 → int16 PCM
            pcm_int16 = _float32_to_int16_bytes(wav_np)
            yield pcm_int16


class _StopIterationSentinel(Exception):
    pass


def _next_chunk(gen_iter):
    """StopIteration を例外として外に出すためのラッパー。"""
    try:
        return next(gen_iter)
    except StopIteration:
        raise _StopIterationSentinel()


def _start_generate_stream(req: TTSRequest, voice_entry: VoiceEntry):
    """ブロッキングでジェネレータを構築して返す。

    voice_entry に応じて conds を切り替えてから generate_stream を呼ぶ。
    """
    assert _model is not None

    # Voice 条件付けの復元
    cached_conds = _voice_registry.get_conds(voice_entry.voice_id)
    if cached_conds is not None:
        _model.conds = cached_conds
        audio_prompt_path = None
    elif voice_entry.audio_path is not None:
        audio_prompt_path = voice_entry.audio_path
    else:
        raise RuntimeError(f"No conditionals for voice_id={voice_entry.voice_id}")

    return _model.generate_stream(
        text=req.text,
        audio_prompt_path=audio_prompt_path,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        chunk_duration_sec=req.chunk_duration_sec,
        crossfade_duration_sec=req.crossfade_duration_sec,
        mel_overlap=req.mel_overlap,
    )


def _float32_to_int16_bytes(wav_np: np.ndarray) -> bytes:
    """float32 numpy 配列を int16 little-endian PCM バイト列に変換する。"""
    # クリッピング
    wav_clipped = np.clip(wav_np, -1.0, 1.0)
    wav_int16 = (wav_clipped * 32767).astype(np.int16)
    return wav_int16.tobytes()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "tts_server:app",
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        log_level="info",
    )
