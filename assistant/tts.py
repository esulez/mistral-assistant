import io
import os
import shutil
import subprocess
import tempfile

from .config import Config

_kokoro_pipeline = None


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
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        from kokoro import KPipeline

        _kokoro_pipeline = KPipeline(lang_code=cfg.tts_lang)
    import numpy as np
    import soundfile as sf
    import torch

    chunks = []
    for _gs, _ps, audio in _kokoro_pipeline(text, voice=cfg.tts_voice):
        chunks.append(audio)
    if not chunks:
        raise RuntimeError("kokoro produced no audio")
    samples = torch.cat(chunks).cpu().numpy().astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, samples, 24000, format="WAV")
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
    play_wav_bytes(data)
