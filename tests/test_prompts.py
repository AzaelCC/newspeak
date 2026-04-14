from newspeak.schemas import ClientMessage
from newspeak.services.prompts import build_history_user_message, build_user_content
from newspeak.services.text import sentences_or_original, split_sentences


def test_split_sentences_and_empty_fallback() -> None:
    assert split_sentences("Hello. How are you? Fine!") == [
        "Hello.",
        "How are you?",
        "Fine!",
    ]
    assert sentences_or_original("   ") == ["   "]


def test_text_only_content() -> None:
    content = build_user_content(
        ClientMessage(text="Tell me a fact."),
        audio_pipeline=None,
        transcription=None,
    )

    assert content == [{"type": "text", "text": "Tell me a fact."}]


def test_image_only_content() -> None:
    content = build_user_content(
        ClientMessage(image="jpg"),
        audio_pipeline=None,
        transcription=None,
    )

    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"] == "data:image/jpeg;base64,jpg"
    assert content[1]["text"] == "The user is showing you their camera. Describe what you see."


def test_direct_audio_content_keeps_input_audio() -> None:
    content = build_user_content(
        ClientMessage(audio="wav"),
        audio_pipeline="direct",
        transcription=None,
    )

    assert content[0] == {
        "type": "input_audio",
        "input_audio": {"data": "wav", "format": "wav"},
    }
    assert content[1]["text"] == "The user just spoke to you. Respond to what they said."


def test_whisperx_audio_content_uses_transcription() -> None:
    content = build_user_content(
        ClientMessage(audio="wav"),
        audio_pipeline="whisperx",
        transcription="hello there",
    )

    assert all(item["type"] != "input_audio" for item in content)
    assert "hello there" in content[0]["text"]


def test_audio_plus_image_content() -> None:
    content = build_user_content(
        ClientMessage(audio="wav", image="jpg"),
        audio_pipeline="whisperx",
        transcription="look at this",
    )

    assert content[0]["type"] == "image_url"
    assert "look at this" in content[1]["text"]
    assert "referencing what you see" in content[1]["text"]


def test_audio_history_drops_raw_audio_and_preserves_image() -> None:
    content = [
        {"type": "input_audio", "input_audio": {"data": "wav", "format": "wav"}},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,jpg"}},
        {"type": "text", "text": "prompt"},
    ]

    history = build_history_user_message(content, has_audio=True, transcription="hello")

    assert history["role"] == "user"
    assert history["content"] == [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,jpg"}},
        {"type": "text", "text": "The user said: hello"},
    ]
