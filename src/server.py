"""Parlor — on-device, real-time multimodal AI (voice + vision)."""

import asyncio
import base64
import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from openai import AsyncOpenAI

import tts

load_dotenv(Path(__file__).parent.parent / ".env")

LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8080")
LLAMA_MODEL = os.environ.get("LLAMA_MODEL", "gemma-4-E2B-it")

SYSTEM_PROMPT = (
    "You are a friendly, conversational AI assistant. The user is talking to you "
    "through a microphone and showing you their camera. "
    "You MUST always use the respond_to_user tool to reply. "
    "First transcribe exactly what the user said, then write your response."
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "respond_to_user",
        "description": "Respond to the user's voice message.",
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


def load_models():
    global tts_backend
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


@app.get("/")
async def root():
    return HTMLResponse(content=(Path(__file__).parent / "index.html").read_text())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

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
            if msg.get("audio"):
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

            if msg.get("audio") and msg.get("image"):
                content.append({"type": "text", "text": "The user just spoke to you (audio) while showing their camera (image). Respond to what they said, referencing what you see if relevant."})
            elif msg.get("audio"):
                content.append({"type": "text", "text": "The user just spoke to you. Respond to what they said."})
            elif msg.get("image"):
                content.append({"type": "text", "text": "The user is showing you their camera. Describe what you see."})
            else:
                content.append({"type": "text", "text": msg.get("text", "Hello!")})

            # LLM inference
            t0 = time.time()
            messages.append({"role": "user", "content": content})
            completion = await llm.chat.completions.create(
                model=LLAMA_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="required",
            )
            llm_time = time.time() - t0

            choice = completion.choices[0].message
            messages.append(choice.model_dump(exclude_unset=True))

            # Extract response from tool call or fallback to raw text
            if choice.tool_calls:
                args = json.loads(choice.tool_calls[0].function.arguments)
                transcription = args.get("transcription", "").strip()
                text_response = args.get("response", "").strip()
                messages.append({
                    "role": "tool",
                    "tool_call_id": choice.tool_calls[0].id,
                    "content": "OK",
                })
                print(f"LLM ({llm_time:.2f}s) [tool] heard: {transcription!r} → {text_response}")
            else:
                transcription = None
                text_response = (choice.content or "").strip()
                print(f"LLM ({llm_time:.2f}s) [no tool]: {text_response}")

            if interrupted.is_set():
                print("Interrupted after LLM, skipping response")
                continue

            reply = {"type": "text", "text": text_response, "llm_time": round(llm_time, 2)}
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
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
