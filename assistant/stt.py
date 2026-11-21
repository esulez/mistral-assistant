import os

from .config import Config

_mistral_client = None
_whisper_model = None


def _client(api_key: str):
    global _mistral_client
    if _mistral_client is None:
        from mistralai import Mistral

        _mistral_client = Mistral(api_key=api_key)
    return _mistral_client


def _whisper(cfg: Config):
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel(cfg.whisper_model, device="cpu", compute_type="int8")
    return _whisper_model


def transcribe_file(path: str, cfg: Config) -> str:
    if cfg.stt == "mistral":
        if not cfg.api_key:
            raise RuntimeError("MISTRAL_API_KEY is required for stt=mistral")
        with open(path, "rb") as handle:
            content = handle.read()
        response = _client(cfg.api_key).audio.transcriptions.complete(
            model=cfg.stt_model,
            file={"file_name": os.path.basename(path), "content": content},
        )
        return (response.text or "").strip()
    if cfg.stt == "whisper":
        segments, _info = _whisper(cfg).transcribe(path)
        return " ".join(segment.text for segment in segments).strip()
    raise ValueError(f"unknown stt backend: {cfg.stt}")
