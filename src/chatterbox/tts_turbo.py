import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Tuple

import librosa
import numpy as np
import torch
import perth
import pyloudnorm as ln

from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from .models.t3 import T3
from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond
from .models.t3.modules.t3_config import T3Config
from .models.s3gen.const import S3GEN_SIL
import logging
logger = logging.getLogger(__name__)

# Streaming constants
_TOKENS_PER_SEC = 25  # S3 tokenizer: 25 tokens/sec
_MEL_PER_TOKEN = 2    # token_mel_ratio in CausalMaskedDiffWithXvec
_SAMPLES_PER_MEL = 480  # mel hop_size at 24kHz

REPO_ID = "ResembleAI/chatterbox-turbo"


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("…", ", "),
        (":", ","),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxTurboTTS:
    ENC_COND_LEN = 15 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTurboTTS':
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(
            load_file(ckpt_dir / "ve.safetensors")
        )
        ve.to(device).eval()

        # Turbo specific hp
        hp = T3Config(text_tokens_dict_size=50276)
        hp.llama_config_name = "GPT2_medium"
        hp.speech_tokens_dict_size = 6563
        hp.input_pos_emb = None
        hp.speech_cond_prompt_len = 375
        hp.use_perceiver_resampler = False
        hp.emotion_adv = False

        t3 = T3(hp)
        t3_state = load_file(ckpt_dir / "t3_turbo_v1.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        del t3.tfmr.wte
        t3.to(device).eval()

        s3gen = S3Gen(meanflow=True)
        weights = load_file(ckpt_dir / "s3gen_meanflow.safetensors")
        s3gen.load_state_dict(
            weights, strict=True
        )
        s3gen.to(device).eval()

        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if len(tokenizer) != 50276:
            print(f"WARNING: Tokenizer len {len(tokenizer)} != 50276")

        conds = None
        builtin_voice = ckpt_dir / "conds.pt"
        if builtin_voice.exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTurboTTS':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        local_path = snapshot_download(
            repo_id=REPO_ID,
            token=os.getenv("HF_TOKEN") or True,
            # Optional: Filter to download only what you need
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]
        )

        return cls.from_local(local_path, device)

    def norm_loudness(self, wav, sr, target_lufs=-27):
        try:
            meter = ln.Meter(sr)
            loudness = meter.integrated_loudness(wav)
            gain_db = target_lufs - loudness
            gain_linear = 10.0 ** (gain_db / 20.0)
            if math.isfinite(gain_linear) and gain_linear > 0.0:
                wav = wav * gain_linear
        except Exception as e:
            print(f"Warning: Error in norm_loudness, skipping: {e}")

        return wav

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5, norm_loudness=True):
        ## Load and norm reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        assert len(s3gen_ref_wav) / _sr > 5.0, "Audio prompt must be longer than 5 seconds!"

        if norm_loudness:
            s3gen_ref_wav = self.norm_loudness(s3gen_ref_wav, _sr)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        repetition_penalty=1.2,
        min_p=0.00,
        top_p=0.95,
        audio_prompt_path=None,
        exaggeration=0.0,
        cfg_weight=0.0,
        temperature=0.8,
        top_k=1000,
        norm_loudness=True,
        apply_watermark=True,
    ):
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration, norm_loudness=norm_loudness)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        if cfg_weight > 0.0 or exaggeration > 0.0 or min_p > 0.0:
            logger.warning("CFG, min_p and exaggeration are not supported by Turbo version and will be ignored.")

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_tokens = text_tokens.input_ids.to(self.device)

        speech_tokens = self.t3.inference_turbo(
            t3_cond=self.conds.t3,
            text_tokens=text_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Remove OOV tokens and add silence to end
        speech_tokens = speech_tokens[speech_tokens < 6561]
        speech_tokens = speech_tokens.to(self.device)
        silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]).long().to(self.device)
        speech_tokens = torch.cat([speech_tokens, silence])

        wav, _ = self.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=self.conds.gen,
            n_cfm_timesteps=2,
        )
        wav = wav.squeeze(0).detach().cpu().numpy()
        if apply_watermark:
            wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(wav).unsqueeze(0)

    def generate_stream(
        self,
        text,
        repetition_penalty=1.2,
        min_p=0.00,
        top_p=0.95,
        audio_prompt_path=None,
        exaggeration=0.0,
        cfg_weight=0.0,
        temperature=0.8,
        top_k=1000,
        norm_loudness=True,
        chunk_duration_sec=1.0,
        crossfade_duration_sec=0.04,
        mel_overlap=10,
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """ストリーミング音声生成。チャンクごとに (sample_rate, wav_numpy) を yield する。

        T3 のトークン生成を 1 トークンずつ受け取り、chunk_duration_sec 秒分蓄積するたびに
        S3Gen (flow + HiFiGAN) で音声を増分デコードして yield する。

        Args:
            text: 合成テキスト
            chunk_duration_sec: 1 チャンクあたりの目標秒数 (デフォルト 1.0)
            crossfade_duration_sec: チャンク境界のクロスフェード秒数 (デフォルト 0.04 = 40ms)
            mel_overlap: HiFiGAN の畳み込み受容野を補うための mel フレーム overlap 数 (デフォルト 10)
            その他: generate() と同一パラメータ

        Yields:
            (sample_rate, wav_chunk): int と 1D numpy array のタプル。
                ストリーミング中はウォーターマークなし。
        """
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration, norm_loudness=norm_loudness)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        if cfg_weight > 0.0 or exaggeration > 0.0 or min_p > 0.0:
            logger.warning("CFG, min_p and exaggeration are not supported by Turbo version and will be ignored.")

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_tokens = text_tokens.input_ids.to(self.device)

        # Streaming parameters
        chunk_tokens = max(1, round(chunk_duration_sec * _TOKENS_PER_SEC))
        crossfade_samples = int(crossfade_duration_sec * S3GEN_SR)

        # Streaming state
        accumulated_tokens = []      # list of (1,1) tensors
        prev_mel_end = 0             # 前回 flow で確定した mel フレーム数
        cache_source = None          # HiFiGAN f0 音源キャッシュ
        prev_wav_tail = None         # crossfade 用の前チャンク末尾 wav (1D tensor)
        is_first_chunk = True

        def _decode_chunk(speech_tokens_1d: torch.Tensor, finalize: bool):
            """累積トークンから新規 wav チャンクをデコードする内部関数。"""
            nonlocal prev_mel_end, cache_source, prev_wav_tail, is_first_chunk

            # OOV 除去 + 最終チャンク時にサイレンス追加
            tokens_clean = speech_tokens_1d[speech_tokens_1d < 6561]
            if finalize:
                silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL], dtype=torch.long, device=self.device)
                tokens_clean = torch.cat([tokens_clean, silence])

            if tokens_clean.numel() == 0:
                return None

            tokens_2d = tokens_clean.unsqueeze(0)  # (1, T)

            # 1. Flow: 累積全トークンで mel 生成
            mel = self.s3gen.flow_inference_streaming(
                speech_tokens=tokens_2d,
                ref_dict=self.conds.gen,
                finalize=finalize,
                n_cfm_timesteps=2,
            )
            # mel shape: (1, 80, mel_len)
            total_mel_len = mel.shape[2]

            if total_mel_len <= prev_mel_end:
                return None

            # 2. 新規 mel 抽出 (+ overlap でコンテキスト確保)
            if prev_mel_end == 0:
                # 最初のチャンク: overlap なし
                new_mel = mel
                overlap_mel_frames = 0
            else:
                overlap_start = max(0, prev_mel_end - mel_overlap)
                new_mel = mel[:, :, overlap_start:]
                overlap_mel_frames = prev_mel_end - overlap_start

            # 3. HiFiGAN: 新規 mel + overlap のみデコード
            wav_chunk, source = self.s3gen.hift_inference_streaming(
                speech_feat=new_mel,
                cache_source=cache_source,
            )
            # wav_chunk shape: (1, wav_len)

            # cache_source を更新: 次チャンクの先頭で f0 連続性を保証
            overlap_source_samples = mel_overlap * _SAMPLES_PER_MEL
            # mel_overlap=0 のとき overlap_source_samples=0 になり, Python の -0==0 により
            # source[:, :, -0:] が全テンソルを返してしまうため, 明示的に空キャッシュを設定する
            if overlap_source_samples > 0:
                cache_source = source[:, :, -overlap_source_samples:] if source.shape[2] > overlap_source_samples else source
            else:
                cache_source = torch.zeros(1, 1, 0, device=source.device, dtype=source.dtype)

            wav_1d = wav_chunk.squeeze(0).detach().cpu()  # (wav_len,)

            # 最初のチャンクで trim_fade を適用 (リファレンスクリップの spillover 低減)
            if is_first_chunk:
                trim_fade = self.s3gen.trim_fade.cpu()
                fade_len = min(len(trim_fade), wav_1d.shape[0])
                wav_1d[:fade_len] *= trim_fade[:fade_len]
                is_first_chunk = False

            # 4. Crossfade
            # overlap 区間は HiFiGAN にコンテキストを与えるためのもの。
            # overlap の末尾と prev_wav_tail の末尾を crossfade し、
            # overlap を除いた新規部分のみを出力する。
            if prev_wav_tail is not None and overlap_mel_frames > 0:
                overlap_wav_samples = overlap_mel_frames * _SAMPLES_PER_MEL
                actual_overlap = min(overlap_wav_samples, wav_1d.shape[0])
                xfade_len = min(crossfade_samples, actual_overlap, prev_wav_tail.shape[0])

                if xfade_len > 0 and actual_overlap > 0:
                    fade_out = torch.linspace(1.0, 0.0, xfade_len)
                    fade_in = torch.linspace(0.0, 1.0, xfade_len)
                    # overlap 区間の末尾 xfade_len サンプルと prev_wav_tail の末尾をブレンド
                    blend_start = actual_overlap - xfade_len
                    blended = (prev_wav_tail[-xfade_len:] * fade_out
                               + wav_1d[blend_start:actual_overlap] * fade_in)
                    # [crossfade 済み区間] + [新規コンテンツ]
                    wav_1d = torch.cat([blended, wav_1d[actual_overlap:]])
                else:
                    # crossfade 不可能な場合、overlap 部分を単純にスキップ
                    wav_1d = wav_1d[actual_overlap:]

            # 次回の crossfade 用に末尾を保持
            prev_wav_tail = (wav_1d[-crossfade_samples:].clone()
                             if wav_1d.shape[0] > crossfade_samples
                             else wav_1d.clone())

            prev_mel_end = total_mel_len

            return wav_1d.numpy()

        # ---- メインループ: T3 からトークンを逐次受け取る ----
        token_gen = self.t3.inference_turbo_stream(
            t3_cond=self.conds.t3,
            text_tokens=text_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        for token in token_gen:
            accumulated_tokens.append(token)

            if len(accumulated_tokens) % chunk_tokens == 0:
                speech_tokens_1d = torch.cat(accumulated_tokens, dim=1).squeeze(0)
                wav_np = _decode_chunk(speech_tokens_1d, finalize=False)
                if wav_np is not None and len(wav_np) > 0:
                    yield (self.sr, wav_np)

        # ---- 最終チャンク: 残りトークンを finalize=True で処理 ----
        if accumulated_tokens:
            speech_tokens_1d = torch.cat(accumulated_tokens, dim=1).squeeze(0)
            wav_np = _decode_chunk(speech_tokens_1d, finalize=True)
            if wav_np is not None and len(wav_np) > 0:
                yield (self.sr, wav_np)
