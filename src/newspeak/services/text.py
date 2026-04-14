import re

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    parts = SENTENCE_SPLIT_RE.split(text.strip())
    return [sentence.strip() for sentence in parts if sentence.strip()]


def sentences_or_original(text: str) -> list[str]:
    sentences = split_sentences(text)
    return sentences if sentences else [text]
