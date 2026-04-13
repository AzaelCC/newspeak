"""Parlor — on-device, real-time multimodal AI (voice + vision)."""

import asyncio
import base64
import io
import json
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import wave

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

import tts

ROOT_DIR = Path(__file__).parent.parent


class ServerConfig(BaseModel):
    port: int = 8000
    llama_server_url: str = "http://localhost:8080"
    llama_model: str = "gemma-4-E2B-it"


class AudioConfig(BaseModel):
    pipeline: Literal["direct", "whisperx"] = "direct"


class WhisperXConfig(BaseModel):
    model: str = "small"
    device: Literal["auto", "cpu", "gpu", "cuda"] = "auto"
    compute_type: Literal["auto", "int8", "float16", "float32"] = "auto"
    batch_size: int = 8
    language: str = ""
    download_root: str = ""


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=str(ROOT_DIR / ".env"),
        extra="ignore",
    )
    server: ServerConfig = Field(default_factory=ServerConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    whisperx: WhisperXConfig = Field(default_factory=WhisperXConfig)


CONFIG = AppConfig()
LLAMA_SERVER_URL = CONFIG.server.llama_server_url
LLAMA_MODEL = CONFIG.server.llama_model

CHAT_SYSTEM_PROMPT = (
    "You are a friendly, conversational AI assistant. The user is talking to you "
    "through a microphone and showing you their camera. Keep responses conversational "
    "and concise."
)

DIRECT_AUDIO_SYSTEM_PROMPT = (
    "You are a friendly, conversational AI assistant. The user is talking to you "
    "through a microphone and showing you their camera. "
    "You MUST always use the respond_to_user tool to reply. "
    "First transcribe exactly what the user said, then write your response."
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "respond_to_user",
        "description": "Transcribe and respond to a raw voice message.",
        "parameters": {
            "type": "object",
            "properties": {
                "transcription": {
                    "type": "string",
                    "description": "Exact transcription of what the user said.",
                },
                "response": {
                    "type": "string",
                    "description": "Your conversational response. Keep it to 1-4 short sentences.",
                },
            },
            "required": ["transcription", "response"],
        },
    },
}]

SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

llm = AsyncOpenAI(base_url=f"{LLAMA_SERVER_URL}/v1", api_key="not-needed")
tts_backend = None
whisperx_transcriber = None


class WhisperXTranscriber:
    def __init__(self, config: WhisperXConfig):
        self.config = config
        self.device = ""
        self.compute_type = ""
        self.model = None

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
        except ImportError:
            return False
        return bool(torch.cuda.is_available())

    def _resolve_runtime(self) -> tuple[str, str]:
        requested_device = self.config.device
        cuda_available = self._cuda_available()

        if requested_device == "cpu":
            device = "cpu"
        elif requested_device in {"gpu", "cuda"}:
            if not cuda_available:
                raise RuntimeError(
                    "WhisperX config requested CUDA/GPU, but CUDA is not available. "
                    "Set whisperx.device = 'cpu' or install a CUDA-enabled PyTorch runtime."
                )
            device = "cuda"
        elif cuda_available:
            device = "cuda"
        else:
            device = "cpu"

        if self.config.compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        else:
            compute_type = self.config.compute_type

        return device, compute_type

    def load(self):
        try:
            import whisperx
        except ImportError as exc:
            raise RuntimeError(
                "audio.pipeline is 'whisperx', but WhisperX is not installed. "
                "Install it with `uv sync --extra whisper`."
            ) from exc

        self.device, self.compute_type = self._resolve_runtime()
        kwargs = {"compute_type": self.compute_type}
        if self.config.language:
            kwargs["language"] = self.config.language
        if self.config.download_root:
            kwargs["download_root"] = self.config.download_root

        print(
            "Loading WhisperX "
            f"model={self.config.model!r} device={self.device!r} "
            f"compute_type={self.compute_type!r}..."
        )
        self.model = whisperx.load_model(self.config.model, self.device, **kwargs)
        print("WhisperX loaded.")

    def transcribe(self, audio_b64: str) -> str:
        if self.model is None:
            self.load()

        audio = decode_wav_base64(audio_b64)
        result = self.model.transcribe(audio, batch_size=self.config.batch_size)
        segments = result.get("segments", [])
        text = " ".join(segment.get("text", "").strip() for segment in segments)
        return " ".join(text.split())


def decode_wav_base64(audio_b64: str) -> np.ndarray:
    audio_bytes = base64.b64decode(audio_b64)
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV audio, got {sample_width * 8}-bit")
    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz WAV audio, got {sample_rate} Hz")

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples


def load_models():
    global tts_backend, whisperx_transcriber
    if CONFIG.audio.pipeline == "whisperx":
        whisperx_transcriber = WhisperXTranscriber(CONFIG.whisperx)
        whisperx_transcriber.load()
    tts_backend = tts.load()


@asynccontextmanager
async def lifespan(app):
    await asyncio.get_event_loop().run_in_executor(None, load_models)
    yield


app = FastAPI(lifespan=lifespan)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming TTS."""
    parts = SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


def build_history_user_message(
    content: list[dict],
    *,
    has_audio: bool,
    transcription: str | None,
) -> dict:
    """Store audio turns as text once ASR has produced a transcript."""
    if not has_audio:
        return {"role": "user", "content": content}

    history_content = [
        item for item in content
        if item.get("type") not in {"input_audio", "text"}
    ]
    transcript_text = transcription or "[empty transcription]"
    history_content.append({"type": "text", "text": f"The user said: {transcript_text}"})
    return {"role": "user", "content": history_content}


@app.get("/")
async def root():
    return HTMLResponse(content=(Path(__file__).parent / "index.html").read_text())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    messages = []

    interrupted = asyncio.Event()
    msg_queue = asyncio.Queue()

    async def receiver():
        """Receive messages from WebSocket and route them."""
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "interrupt":
                    interrupted.set()
                    print("Client interrupted")
                else:
                    await msg_queue.put(msg)
        except WebSocketDisconnect:
            await msg_queue.put(None)

    recv_task = asyncio.create_task(receiver())

    try:
        while True:
            msg = await msg_queue.get()
            if msg is None:
                break

            interrupted.clear()

            content = []
            transcription = None
            asr_time = None
            has_audio = bool(msg.get("audio"))
            audio_pipeline = CONFIG.audio.pipeline if has_audio else None

            if has_audio and CONFIG.audio.pipeline == "whisperx":
                if whisperx_transcriber is None:
                    raise RuntimeError("WhisperX pipeline is enabled, but the transcriber is not loaded")
                asr_start = time.time()
                transcription = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: whisperx_transcriber.transcribe(msg["audio"])
                )
                asr_time = time.time() - asr_start
                print(f"WhisperX ({asr_time:.2f}s) heard: {transcription!r}")
            elif has_audio:
                # Browser already sends base64 WAV — pass directly
                content.append({
                    "type": "input_audio",
                    "input_audio": {"data": msg["audio"], "format": "wav"},
                })
            if msg.get("image"):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{msg['image']}"},
                })

            if has_audio and msg.get("image"):
                if CONFIG.audio.pipeline == "whisperx":
                    prompt = (
                        "The user just said this while showing their camera: "
                        f"{transcription or '[empty transcription]'!r}. "
                        "Respond to what they said, referencing what you see if relevant."
                    )
                else:
                    prompt = "The user just spoke to you (audio) while showing their camera (image). Respond to what they said, referencing what you see if relevant."
                content.append({"type": "text", "text": prompt})
            elif has_audio:
                if CONFIG.audio.pipeline == "whisperx":
                    prompt = (
                        "The user just said this: "
                        f"{transcription or '[empty transcription]'!r}. Respond to what they said."
                    )
                else:
                    prompt = "The user just spoke to you. Respond to what they said."
                content.append({"type": "text", "text": prompt})
            elif msg.get("image"):
                content.append({"type": "text", "text": "The user is showing you their camera. Describe what you see."})
            else:
                content.append({"type": "text", "text": msg.get("text", "Hello!")})

            # LLM inference
            t0 = time.time()
            user_message = {"role": "user", "content": content}
            use_transcription_tool = has_audio and CONFIG.audio.pipeline == "direct"
            system_prompt = (
                DIRECT_AUDIO_SYSTEM_PROMPT
                if use_transcription_tool
                else CHAT_SYSTEM_PROMPT
            )
            completion_kwargs = {
                "model": LLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    *messages,
                    user_message,
                ],
            }
            if use_transcription_tool:
                completion_kwargs.update({
                    "tools": TOOLS,
                    "tool_choice": "required",
                })
            completion = await llm.chat.completions.create(**completion_kwargs)
            llm_time = time.time() - t0

            choice = completion.choices[0].message

            # Direct raw-audio turns use the LLM tool as an ASR adapter. WhisperX,
            # text-only, and image-only turns return normal assistant content.
            if use_transcription_tool and choice.tool_calls:
                args = json.loads(choice.tool_calls[0].function.arguments)
                tool_transcription = args.get("transcription", "").strip()
                if transcription is None:
                    transcription = tool_transcription
                text_response = args.get("response", "").strip()
                print(f"LLM ({llm_time:.2f}s) [tool] heard: {transcription!r} → {text_response}")
            else:
                text_response = (choice.content or "").strip()
                print(f"LLM ({llm_time:.2f}s) [chat]: {text_response}")

            messages.append(build_history_user_message(
                content,
                has_audio=has_audio,
                transcription=transcription,
            ))
            messages.append({"role": "assistant", "content": text_response})

            if interrupted.is_set():
                print("Interrupted after LLM, skipping response")
                continue

            reply = {"type": "text", "text": text_response, "llm_time": round(llm_time, 2)}
            if audio_pipeline:
                reply["audio_pipeline"] = audio_pipeline
            if asr_time is not None:
                reply["asr_time"] = round(asr_time, 2)
            if transcription:
                reply["transcription"] = transcription
            await ws.send_text(json.dumps(reply))

            if interrupted.is_set():
                print("Interrupted before TTS, skipping audio")
                continue

            # Streaming TTS: split into sentences and send chunks progressively
            sentences = split_sentences(text_response)
            if not sentences:
                sentences = [text_response]

            tts_start = time.time()

            # Signal start of audio stream
            await ws.send_text(json.dumps({
                "type": "audio_start",
                "sample_rate": tts_backend.sample_rate,
                "sentence_count": len(sentences),
            }))

            for i, sentence in enumerate(sentences):
                if interrupted.is_set():
                    print(f"Interrupted during TTS (sentence {i+1}/{len(sentences)})")
                    break

                # Generate audio for this sentence
                pcm = await asyncio.get_event_loop().run_in_executor(
                    None, lambda s=sentence: tts_backend.generate(s)
                )

                if interrupted.is_set():
                    break

                # Convert to 16-bit PCM and send as base64
                pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
                await ws.send_text(json.dumps({
                    "type": "audio_chunk",
                    "audio": base64.b64encode(pcm_int16.tobytes()).decode(),
                    "index": i,
                }))

            tts_time = time.time() - tts_start
            print(f"TTS ({tts_time:.2f}s): {len(sentences)} sentences")

            if not interrupted.is_set():
                await ws.send_text(json.dumps({
                    "type": "audio_end",
                    "tts_time": round(tts_time, 2),
                }))

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        recv_task.cancel()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=CONFIG.server.port)
