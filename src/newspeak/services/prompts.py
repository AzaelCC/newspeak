from typing import Any

from newspeak.schemas import ClientMessage

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

TRANSCRIPTION_TOOLS: list[dict[str, Any]] = [
    {
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
                        "description": (
                            "Your conversational response. Keep it to 1-4 short sentences."
                        ),
                    },
                },
                "required": ["transcription", "response"],
            },
        },
    }
]


def system_prompt_for(
    use_transcription_tool: bool,
    roleplay_system: str | None = None,
) -> str:
    if use_transcription_tool:
        base = DIRECT_AUDIO_SYSTEM_PROMPT
        if roleplay_system:
            # Prepend mode persona before the tool-use instruction
            base = f"{roleplay_system}\n\n{DIRECT_AUDIO_SYSTEM_PROMPT}"
        return base
    return roleplay_system or CHAT_SYSTEM_PROMPT


def build_user_content(
    message: ClientMessage,
    *,
    audio_pipeline: str | None,
    transcription: str | None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    has_audio = bool(message.audio)
    has_image = bool(message.image)

    if has_audio and audio_pipeline == "direct":
        content.append(
            {
                "type": "input_audio",
                "input_audio": {"data": message.audio, "format": "wav"},
            }
        )

    if has_image:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{message.image}"},
            }
        )

    if has_audio and has_image:
        if audio_pipeline == "whisperx":
            prompt = (
                "The user just said this while showing their camera: "
                f"{transcription or '[empty transcription]'!r}. "
                "Respond to what they said, referencing what you see if relevant."
            )
        else:
            prompt = (
                "The user just spoke to you (audio) while showing their camera (image). "
                "Respond to what they said, referencing what you see if relevant."
            )
    elif has_audio:
        if audio_pipeline == "whisperx":
            prompt = (
                "The user just said this: "
                f"{transcription or '[empty transcription]'!r}. Respond to what they said."
            )
        else:
            prompt = "The user just spoke to you. Respond to what they said."
    elif has_image:
        prompt = "The user is showing you their camera. Describe what you see."
    else:
        prompt = message.text or "Hello!"

    content.append({"type": "text", "text": prompt})
    return content


def build_history_user_message(
    content: list[dict[str, Any]],
    *,
    has_audio: bool,
    transcription: str | None,
) -> dict[str, Any]:
    if not has_audio:
        return {"role": "user", "content": content}

    history_content = [item for item in content if item.get("type") not in {"input_audio", "text"}]
    transcript_text = transcription or "[empty transcription]"
    history_content.append({"type": "text", "text": f"The user said: {transcript_text}"})
    return {"role": "user", "content": history_content}
