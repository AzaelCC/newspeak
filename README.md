# Newspeak

On-device, real-time multimodal AI. Have natural voice and vision conversations with an AI that runs entirely on your machine.

Newspeak uses [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) for understanding speech and vision, and [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) for text-to-speech. You talk, show your camera, and it talks back, all locally.

https://github.com/user-attachments/assets/cb0ffb2e-f84f-48e7-872c-c5f7b5c6d51f

> **Research preview.** This is an early experiment. Expect rough edges and bugs.

# Why?

I'm [self-hosting a totally free voice AI](https://www.fikrikarim.com/bule-ai-initial-release/) on my home server to help people learn speaking English. It has hundreds of monthly active users, and I've been thinking about how to keep it free while making it sustainable.

The obvious answer: run everything on-device, eliminating any server cost. Six months ago I needed an RTX 5090 to run just the voice models in real-time.

Google just released a super capable small model that I can run on my M3 Pro in real-time, with vision too! Sure you can't do agentic coding with this, but it is a game-changer for people learning a new language. Imagine a few years from now that people can run this locally on their phones. They can point their camera at objects and talk about them. And this model is multi-lingual, so people can always fallback to their native language if they want. This is essentially what OpenAI demoed a few years ago.

## How it works

```
Browser (mic + camera)
    |
    |  WebSocket (audio PCM + JPEG frames)
    v
FastAPI server
    |-- Direct mode: sends audio + image to Gemma via OpenAI-compatible chat
    |-- WhisperX mode: transcribes audio locally, then sends text + image to Gemma
    |-- Kokoro TTS (MLX on Mac, ONNX on Linux) -> speaks back
    |
    |  WebSocket (streamed audio chunks)
    v
Browser (playback + transcript)
```

- **Voice Activity Detection** in the browser ([Silero VAD](https://github.com/ricky0123/vad)). Hands-free, no push-to-talk.
- **Barge-in.** Interrupt the AI mid-sentence by speaking.
- **Sentence-level TTS streaming.** Audio starts playing before the full response is generated.
- **Selectable audio pipeline.** Keep direct multimodal audio, or switch to local WhisperX transcription before chat inference.

## Requirements

- Python 3.12+
- macOS with Apple Silicon, or Linux/Windows with supported TTS and inference backends
- A running OpenAI-compatible chat server, configured via `SERVER__LLAMA_SERVER_URL`
- ~3 GB free RAM for the chat model, plus any WhisperX model memory when enabled
- The `whisper` extra excludes macOS `mlx-audio` because WhisperX 3.8.5 and `mlx-audio` currently require incompatible `huggingface-hub` versions.

## Quick start

```bash
git clone https://github.com/fikrikarim/newspeak.git
cd newspeak

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync --group dev
uv run newspeak-server
```

Open [http://localhost:8000](http://localhost:8000), grant camera and microphone access, and start talking.

Models are downloaded automatically on first run.

To use the WhisperX pipeline:

```bash
uv sync --extra whisper
```

Then set `AUDIO__PIPELINE=whisperx` in your `.env` (copy `.env.example` to `.env` to get started) or as an environment variable. Optionally set `WHISPERX__DEVICE` to `cpu`, `cuda`, `gpu`, or `auto` (default).

## Configuration

Configuration is set via environment variables or a `.env` file at the repo root. Copy `.env.example` to `.env` and uncomment the lines you want to change. All settings have working defaults; you only need to set what differs.

Nested keys use double-underscore as separator (`SECTION__KEY`). Environment variables take precedence over `.env` values.

| Variable                    | Default                 | Description                             |
| --------------------------- | ----------------------- | --------------------------------------- |
| `SERVER__PORT`              | `8000`                  | FastAPI server port                     |
| `SERVER__LLAMA_SERVER_URL`  | `http://localhost:8080` | OpenAI-compatible llama-server URL      |
| `SERVER__LLAMA_MODEL`       | `gemma-4-E2B-it`        | Model name or alias for chat calls      |
| `SERVER__LOG_LEVEL`         | `INFO`                  | Python logging level                    |
| `AUDIO__PIPELINE`           | `direct`                | `direct` or `whisperx`                  |
| `WHISPERX__MODEL`           | `small`                 | WhisperX ASR model                      |
| `WHISPERX__DEVICE`          | `auto`                  | `cpu`, `cuda`, `gpu`, or `auto`         |
| `WHISPERX__COMPUTE_TYPE`    | `auto`                  | `int8`, `float16`, `float32`, or `auto` |
| `WHISPERX__BATCH_SIZE`      | `8`                     | WhisperX transcription batch size       |
| `WHISPERX__LANGUAGE`        | empty                   | Empty = auto-detect language            |
| `WHISPERX__DOWNLOAD_ROOT`   | empty                   | Optional model download/cache directory |

`WHISPERX__DEVICE=cpu` always forces CPU, even when CUDA is installed. `gpu` and `cuda` require CUDA and fail at startup if unavailable. `auto` uses CUDA when available, CPU otherwise.

## Performance (Apple M3 Pro)

| Stage                            | Time          |
| -------------------------------- | ------------- |
| Speech + vision understanding    | ~1.8-2.2s     |
| Response generation (~25 tokens) | ~0.3s         |
| Text-to-speech (1-3 sentences)   | ~0.3-0.7s     |
| **Total end-to-end**             | **~2.5-3.0s** |

Decode speed: ~83 tokens/sec on GPU (Apple M3 Pro).

## Project structure

```
pyproject.toml             # Dependencies, scripts, Ruff, and dev tooling
src/
|-- server.py              # Compatibility entry point
|-- tts.py                 # Compatibility TTS wrapper
`-- newspeak/
    |-- app.py             # FastAPI app factory
    |-- config.py          # Pydantic settings
    |-- api/               # HTTP and WebSocket routes
    |-- adapters/          # OpenAI-compatible LLM, WhisperX, and TTS backends
    |-- services/          # Conversation, prompts, audio, and TTS streaming
    `-- web/static/        # Browser HTML, CSS, and JS
benchmarks/
|-- bench.py               # End-to-end WebSocket benchmark
`-- benchmark_tts.py       # TTS backend comparison
tests/                     # Unit tests for core behavior
```

`.env.example` documents all available settings. Local `.env` is ignored by git.
See [ARCHITECTURE.md](ARCHITECTURE.md) for the layered design, patterns used, and anti-patterns fixed in the refactor.

## Acknowledgments

- [Gemma 4](https://ai.google.dev/gemma) by Google DeepMind
- [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) by Google AI Edge
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) TTS by Hexgrad
- [Silero VAD](https://github.com/snakers4/silero-vad) for browser voice activity detection
- [WhisperX](https://github.com/m-bain/whisperX) for local ASR when enabled

## License

[Apache 2.0](LICENSE)
