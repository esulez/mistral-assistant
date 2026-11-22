import io
import os
import shutil
import subprocess
import tempfile

from .config import Config
from .lang import DEFAULT_VOICES, LANG_CODES, detect_language

_kokoro_pipelines: dict[str, object] = {}


def kokoro_params(text: str, cfg: Config) -> tuple[str, str]:
    if cfg.tts_lang and cfg.tts_lang != "auto":
        return cfg.tts_lang, cfg.tts_voice
    lang = detect_language(text)
    lang_code = LANG_CODES.get(lang, "a")
    voice = cfg.tts_voice if cfg.tts_voice.startswith(lang_code) else DEFAULT_VOICES.get(lang, "af_bella")
    return lang_code, voice


def available_engines() -> list[str]:
    engines = []
    try:
        import kokoro  # noqa: F401

        engines.append("kokoro")
    except ImportError:
        pass
    if shutil.which("piper"):
        engines.append("piper")
    for binary in ("espeak-ng", "espeak"):
        if shutil.which(binary):
            engines.append("espeak")
            break
    engines.append("none")
    return engines


def resolve_tts(requested: str) -> str:
    if requested != "auto":
        return requested
    engines = available_engines()
    return engines[0] if engines else "none"


def _kokoro_synthesize(text: str, cfg: Config) -> tuple[bytes, int]:
    from kokoro import KPipeline

    import numpy as np
    import soundfile as sf
    import torch

    lang_code, voice = kokoro_params(text, cfg)
    pipeline = _kokoro_pipelines.get(lang_code)
    if pipeline is None:
        pipeline = KPipeline(lang_code=lang_code)
        _kokoro_pipelines[lang_code] = pipeline
    arrays = [np.asarray(audio, dtype=np.float32) for _gs, _ps, audio in pipeline(text, voice=voice)]
    if not arrays:
        raise RuntimeError("kokoro produced no audio")
    samples = torch.as_tensor(np.concatenate(arrays))
    buffer = io.BytesIO()
    sf.write(buffer, samples.cpu().numpy(), 24000, format="WAV", subtype="PCM_16")
    return buffer.getvalue(), 24000


def _piper_synthesize(text: str) -> tuple[bytes, int]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name
    try:
        subprocess.run(
            ["piper", "--output_file", out_path],
            input=text.encode(),
            check=True,
            capture_output=True,
        )
        with open(out_path, "rb") as handle:
            return handle.read(), 22050
    finally:
        os.unlink(out_path)


def _espeak_synthesize(text: str) -> tuple[bytes, int]:
    binary = shutil.which("espeak-ng") or shutil.which("espeak")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name
    try:
        subprocess.run([binary, "-w", out_path, text], check=True, capture_output=True)
        with open(out_path, "rb") as handle:
            return handle.read(), 22050
    finally:
        os.unlink(out_path)


def synthesize(text: str, cfg: Config) -> tuple[bytes, int] | None:
    engine = resolve_tts(cfg.tts)
    if engine == "kokoro":
        return _kokoro_synthesize(text, cfg)
    if engine == "piper":
        return _piper_synthesize(text)
    if engine == "espeak":
        return _espeak_synthesize(text)
    return None


def speak(text: str, cfg: Config) -> None:
    from .audio import play_wav_bytes

    result = synthesize(text, cfg)
    if result is None:
        print(f"(no tts engine) {text}")
        return
    data, _sample_rate = result
    play_wav_bytes(data, device=cfg.output_device)
