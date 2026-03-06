"""Chatterbox Turbo ストリーミング生成テスト（Voice Cloning なし）。

デフォルトボイスでストリーミング生成を行い、
チャンクごとの生成状況をログ出力したうえで wav ファイルに保存する。
"""

import time
import numpy as np
import torch
import torchaudio as ta

from chatterbox.tts_turbo import ChatterboxTurboTTS

# ---- デバイス自動検出 ----
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# ---- モデルロード ----
model = ChatterboxTurboTTS.from_pretrained(device=device)

# ---- テスト用テキスト ----
text = (
    "Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. "
    "It's the SkyNet T-800 series and it's got basically everything. "
    "Including AI integration with ChatGPT and all that jazz. "
    "Would you like me to get some prices for you?"
)

# ---- ストリーミング生成 (Voice Cloning なし) ----
print(f"\n{'='*60}")
print("[TEST] Streaming WITHOUT voice cloning (default voice)")
print(f"{'='*60}")

chunks: list[np.ndarray] = []
sample_rate = None
t_start = time.perf_counter()
t_first_chunk = None

for i, (sr, wav_chunk) in enumerate(
    model.generate_stream(text, chunk_duration_sec=1.0)
):
    elapsed = time.perf_counter() - t_start
    if t_first_chunk is None:
        t_first_chunk = elapsed
    chunk_dur = len(wav_chunk) / sr
    print(f"  chunk {i:3d} | {chunk_dur:.3f}s audio | elapsed {elapsed:.3f}s")
    chunks.append(wav_chunk)
    sample_rate = sr

t_total = time.perf_counter() - t_start

if not chunks:
    print("  ⚠ No chunks generated!")
else:
    full_wav = np.concatenate(chunks)
    total_audio_sec = len(full_wav) / sample_rate
    wav_tensor = torch.from_numpy(full_wav).unsqueeze(0)
    output_path = "test-turbo-stream-default.wav"
    ta.save(output_path, wav_tensor, sample_rate)

    print(f"\n  ✅ Saved: {output_path}")
    print(f"     chunks     : {len(chunks)}")
    print(f"     audio      : {total_audio_sec:.2f}s")
    print(f"     TTFB       : {t_first_chunk:.3f}s")
    print(f"     total time : {t_total:.3f}s")
    print(f"     RTF        : {t_total / total_audio_sec:.3f}")
