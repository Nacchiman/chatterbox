# Chatterbox Turbo TTS Server

Chatterbox Turbo モデルによるストリーミング音声合成 (TTS) サーバーと、
LiveKit Agent 向けの TTS プラグインです。

## アーキテクチャ

```
LiveKit Agent                    TTS Server (GPU)
+-----------------------+        +----------------------------+
| VoicePipelineAgent    |        | FastAPI (tts_server.py)    |
|   |                   |        |   |                        |
|   +-- ChatterboxTTS --+--HTTP--+-> /v1/tts/stream           |
|      (プラグイン)      |  POST  |   |                        |
|                       |        |   +-- ChatterboxTurboTTS   |
|   PCM chunks (stream) |<-------+       generate_stream()    |
+-----------------------+        +----------------------------+
```

- **TTS Server** (`tts_server.py`): Chatterbox Turbo モデルを GPU 上でロードし、HTTP ストリーミングで PCM 音声チャンクを逐次送出
- **LiveKit Plugin** (`chatterbox_turbo_tts.py`): TTS サーバーに接続し、`livekit.agents.tts.TTS` インターフェースとして動作

---

## ファイル構成

```
servers/
├── README.md                     # このドキュメント
├── tts_server.py                 # FastAPI TTS サーバー本体
└── chatterbox_turbo_tts.py       # LiveKit Agent 用 TTS プラグイン
```

---

## セットアップ

### 前提条件

- Python 3.10+
- Chatterbox TTS パッケージがインストール済み (`pip install -e .` をプロジェクトルートで実行)
- NVIDIA GPU (推奨) / Apple Silicon MPS / CPU

### TTS サーバーの依存パッケージ

```bash
pip install fastapi uvicorn python-multipart
```

### LiveKit プラグインの依存パッケージ

```bash
pip install livekit-agents httpx
```

---

## TTS サーバーの起動

```bash
cd servers/
python tts_server.py
```

または uvicorn で直接起動:

```bash
cd servers/
uvicorn tts_server:app --host 0.0.0.0 --port 8880
```

起動するとモデルが自動ロードされ (初回は HuggingFace からダウンロード)、
ポート **8880** で API が利用可能になります。

### 環境変数

| 変数名 | 説明 | デフォルト |
|--------|------|-----------|
| `HF_TOKEN` | HuggingFace トークン (モデルダウンロード用) | なし |

---

## API リファレンス

サーバー起動後、Swagger UI が `http://localhost:8880/docs` で利用できます。

### `GET /health`

ヘルスチェック。

**レスポンス例:**

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### `POST /v1/tts/stream`

テキストを一括入力し、音声を PCM ストリーミングで返します。

**リクエストボディ (JSON):**

| フィールド | 型 | 必須 | デフォルト | 説明 |
|-----------|-----|------|-----------|------|
| `text` | string | Yes | - | 合成するテキスト |
| `voice_id` | string | No | `"default"` | 使用するボイス ID |
| `temperature` | float | No | `0.8` | サンプリング温度 (0.01 - 5.0) |
| `top_k` | int | No | `1000` | Top-K サンプリング (>= 1) |
| `top_p` | float | No | `0.95` | Top-P (nucleus) サンプリング (0.0 - 1.0) |
| `repetition_penalty` | float | No | `1.2` | 繰り返しペナルティ (1.0 - 3.0) |
| `chunk_duration_sec` | float | No | `1.0` | 1 チャンクあたりの目標秒数 (0.25 - 5.0) |

**リクエスト例:**

```bash
curl -X POST http://localhost:8880/v1/tts/stream \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test of the Chatterbox Turbo streaming TTS server.",
    "voice_id": "default",
    "temperature": 0.8
  }' \
  --output output.pcm
```

**レスポンス:**

- Content-Type: `application/octet-stream`
- ボディ: 生の PCM int16 little-endian バイト列を連続送出

**レスポンスヘッダー:**

| ヘッダー | 説明 | 値の例 |
|---------|------|--------|
| `X-Sample-Rate` | サンプルレート (Hz) | `24000` |
| `X-Sample-Width` | サンプル幅 (bytes) | `2` |
| `X-Channels` | チャンネル数 | `1` |
| `X-Request-Id` | リクエスト ID (UUID) | `a1b2c3d4-...` |

**PCM → WAV 変換 (ffmpeg):**

```bash
ffmpeg -f s16le -ar 24000 -ac 1 -i output.pcm output.wav
```

**Python でのストリーミング受信:**

```python
import httpx

with httpx.stream("POST", "http://localhost:8880/v1/tts/stream",
                   json={"text": "Hello world."}) as resp:
    sample_rate = int(resp.headers["x-sample-rate"])
    for chunk in resp.iter_bytes(chunk_size=4800):
        # chunk は int16 little-endian PCM バイト列
        # オーディオ再生やバッファに追加する
        process_audio(chunk, sample_rate)
```

---

### `POST /v1/voice/register`

参照音声をアップロードし、Voice Cloning 用のボイスを登録します。
内部で `prepare_conditionals()` を呼び、条件付けを事前計算してキャッシュします。

**リクエスト:** `multipart/form-data`

| フィールド | 型 | 必須 | 説明 |
|-----------|-----|------|------|
| `file` | file | Yes | 参照音声ファイル (WAV, 5 秒以上) |
| `name` | string | No | ボイスの表示名 |
| `voice_id` | string | No | 任意の voice_id (省略時は自動生成) |

**リクエスト例:**

```bash
curl -X POST http://localhost:8880/v1/voice/register \
  -F "file=@my_reference.wav" \
  -F "name=MyVoice" \
  -F "voice_id=alice"
```

**レスポンス例:**

```json
{
  "voice_id": "alice",
  "name": "MyVoice",
  "registered_at": 1741305600.0
}
```

> **注意:** 参照音声は 5 秒以上である必要があります。

---

### `GET /v1/voice/list`

登録済みボイスの一覧を取得します。

**レスポンス例:**

```json
{
  "voices": [
    {
      "voice_id": "default",
      "name": "Default",
      "registered_at": 1741305600.0
    },
    {
      "voice_id": "alice",
      "name": "MyVoice",
      "registered_at": 1741305660.0
    }
  ]
}
```

---

## LiveKit Agent での利用

### 基本的な使い方

```python
from servers.chatterbox_turbo_tts import ChatterboxTTS

# TTS プラグインを作成
tts = ChatterboxTTS(
    base_url="http://localhost:8880",  # TTS サーバーのアドレス
    voice_id="default",
)
```

### VoicePipelineAgent との統合

```python
from livekit.agents.pipeline import VoicePipelineAgent
from servers.chatterbox_turbo_tts import ChatterboxTTS

tts = ChatterboxTTS(
    base_url="http://localhost:8880",
    voice_id="alice",              # 事前に /v1/voice/register で登録した voice_id
    temperature=0.8,
    chunk_duration_sec=1.0,
)

agent = VoicePipelineAgent(
    tts=tts,
    # ... 他の設定
)
```

### ChatterboxTTS コンストラクタ引数

| 引数 | 型 | デフォルト | 説明 |
|------|-----|-----------|------|
| `base_url` | str | `"http://localhost:8880"` | TTS サーバーの URL |
| `voice_id` | str | `"default"` | 使用するボイス ID |
| `temperature` | float | `0.8` | サンプリング温度 |
| `top_k` | int | `1000` | Top-K サンプリング |
| `top_p` | float | `0.95` | Top-P サンプリング |
| `repetition_penalty` | float | `1.2` | 繰り返しペナルティ |
| `chunk_duration_sec` | float | `1.0` | サーバー側の 1 チャンクあたりの目標秒数 |
| `http_timeout` | float | `120.0` | HTTP タイムアウト (秒) |
| `sample_rate` | int | `24000` | サンプルレート |
| `num_channels` | int | `1` | チャンネル数 |

### 実行時のオプション変更

```python
tts.update_options(
    voice_id="bob",
    temperature=1.0,
)
```

---

## 設計上の注意点

### GPU 排他制御

TTS サーバーは `asyncio.Lock` を使い、GPU 推論リクエストを直列化しています。
複数リクエストが同時に来た場合、先着順で処理されます。

### Voice キャッシュ

`/v1/voice/register` で登録されたボイスの条件付け (Conditionals) は
メモリ上にキャッシュされ、同じ `voice_id` での合成時に再計算をスキップします。
サーバーを再起動するとキャッシュはクリアされるため、再登録が必要です。

### 音声フォーマット

| 項目 | 値 |
|------|-----|
| フォーマット | PCM int16 little-endian (raw) |
| サンプルレート | 24,000 Hz |
| チャンネル数 | 1 (モノラル) |
| ビット深度 | 16 bit |

### 今後の拡張予定

- テキストの逐次入力対応 (`SynthesizeStream` / `stream()` メソッドの実装)
- Voice キャッシュの永続化
- 複数 GPU / ワーカー対応
