import os
import shutil
from dataclasses import dataclass

DEFAULT_SYSTEM_PROMPT = (
    "You are Mistral Assistant, a voice-first assistant. Reply in the same language "
    "the user speaks; you understand English, French, Italian and Spanish. "
    "Keep answers short, conversational, and easy to listen to."
)
DEFAULT_STT_MODEL = "voxtral-mini-latest"
DEFAULT_CHAT_MODEL = "mistral-large-latest"
DEFAULT_WAKE_WORD = "salut mistral ça va"

BUILTIN_WAKE_WORDS = {
    "alexa",
    "hey mycroft",
    "hey_mycroft",
    "hey jarvis",
    "hey_jarvis",
    "hey rhasspy",
    "hey_rhasspy",
    "timers",
    "weather",
}


@dataclass
class Config:
    api_key: str | None
    backend: str
    model: str
    system_prompt: str
    stt: str
    stt_model: str
    stt_language: str | None
    whisper_model: str
    tts: str
    tts_voice: str
    tts_lang: str
    wake_word: str
    wake_mode: str
    wakeword_model: str | None
    wake_threshold: float
    sample_rate: int
    chunk_samples: int
    silence_threshold: float
    silence_seconds: float
    max_utterance_seconds: float
    vibe_timeout: float
    output_device: str | None
    web_search: bool

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        api_key = overrides.get("api_key", os.getenv("MISTRAL_API_KEY"))
        values = dict(
            api_key=api_key,
            backend=overrides.get("backend", os.getenv("ASSISTANT_BACKEND", "auto")),
            model=overrides.get("model", os.getenv("ASSISTANT_MODEL", DEFAULT_CHAT_MODEL)),
            system_prompt=overrides.get(
                "system_prompt",
                os.getenv("ASSISTANT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
            ),
            stt=overrides.get("stt", os.getenv("ASSISTANT_STT", "auto")),
            stt_model=overrides.get("stt_model", os.getenv("ASSISTANT_STT_MODEL", DEFAULT_STT_MODEL)),
            stt_language=overrides.get("stt_language", os.getenv("ASSISTANT_STT_LANGUAGE") or None),
            whisper_model=overrides.get("whisper_model", os.getenv("ASSISTANT_WHISPER_MODEL", "base")),
            tts=overrides.get("tts", os.getenv("ASSISTANT_TTS", "auto")),
            tts_voice=overrides.get("tts_voice", os.getenv("ASSISTANT_TTS_VOICE", "af_bella")),
            tts_lang=overrides.get("tts_lang", os.getenv("ASSISTANT_TTS_LANG", "auto")),
            wake_word=overrides.get("wake_word", os.getenv("ASSISTANT_WAKE_WORD", DEFAULT_WAKE_WORD)),
            wake_mode=overrides.get("wake_mode", os.getenv("ASSISTANT_WAKE_MODE", "auto")),
            wakeword_model=overrides.get(
                "wakeword_model", os.getenv("ASSISTANT_WAKEWORD_MODEL") or None
            ),
            wake_threshold=float(overrides.get("wake_threshold", os.getenv("ASSISTANT_WAKE_THRESHOLD", "0.5"))),
            sample_rate=int(overrides.get("sample_rate", os.getenv("ASSISTANT_SAMPLE_RATE", "16000"))),
            chunk_samples=int(overrides.get("chunk_samples", os.getenv("ASSISTANT_CHUNK_SAMPLES", "1280"))),
            silence_threshold=float(
                overrides.get("silence_threshold", os.getenv("ASSISTANT_SILENCE_THRESHOLD", "300"))
            ),
            silence_seconds=float(overrides.get("silence_seconds", os.getenv("ASSISTANT_SILENCE_SECONDS", "1.5"))),
            max_utterance_seconds=float(
                overrides.get("max_utterance_seconds", os.getenv("ASSISTANT_MAX_UTTERANCE_SECONDS", "30"))
            ),
            vibe_timeout=float(overrides.get("vibe_timeout", os.getenv("ASSISTANT_VIBE_TIMEOUT", "180"))),
            output_device=overrides.get("output_device", os.getenv("ASSISTANT_OUTPUT_DEVICE") or None),
            web_search=overrides.get(
                "web_search",
                os.getenv("ASSISTANT_WEB_SEARCH", "true").lower() not in ("0", "false", "no"),
            ),
        )
        values["backend"] = resolve_backend(values["backend"], api_key)
        values["stt"] = resolve_stt(values["stt"], api_key)
        values["wake_mode"] = resolve_wake_mode(
            values["wake_mode"], values["wakeword_model"], values["wake_word"]
        )
        return cls(**values)


def resolve_backend(requested: str, api_key: str | None) -> str:
    if requested != "auto":
        return requested
    if shutil.which("vibe"):
        return "vibe"
    if api_key:
        return "api"
    return "echo"


def resolve_stt(requested: str, api_key: str | None) -> str:
    if requested != "auto":
        return requested
    return "mistral" if api_key else "whisper"


def resolve_wake_mode(requested: str, wakeword_model: str | None, wake_word: str) -> str:
    if requested != "auto":
        return requested
    if wakeword_model:
        return "model"
    if wake_word.strip().lower() in BUILTIN_WAKE_WORDS:
        return "model"
    return "phrase"
