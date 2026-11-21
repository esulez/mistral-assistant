import argparse
import os
import sys
import tempfile

from .brain import Conversation, create_backend
from .config import Config

EXIT_PHRASES = {"goodbye", "stop listening", "exit", "quit", "stop the assistant"}


def is_exit_phrase(text: str) -> bool:
    return text.strip().lower().strip(".!?") in EXIT_PHRASES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mistral-assistant",
        description="Voice-first front-end for Mistral Vibe: say the wake word, ask, hear the answer.",
    )
    parser.add_argument("--text", action="store_true", help="text mode: type instead of speaking")
    parser.add_argument("--once", action="store_true", help="handle a single query, then exit")
    parser.add_argument("--backend", choices=["auto", "vibe", "api", "echo"], default=None)
    parser.add_argument("--stt", choices=["auto", "mistral", "whisper"], default=None)
    parser.add_argument("--tts", choices=["auto", "kokoro", "piper", "espeak", "none"], default=None)
    parser.add_argument("--wake-word", default=None)
    return parser


def run_text_mode(cfg: Config, once: bool) -> int:
    conversation = Conversation(create_backend(cfg), cfg.system_prompt)
    print(f"text mode | backend={cfg.backend} stt={cfg.stt} tts={cfg.tts} (type 'goodbye' to quit)")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if is_exit_phrase(line):
            print("assistant> Goodbye!")
            break
        reply = conversation.ask(line)
        print(f"assistant> {reply}")
        if once:
            break
    return 0


def run_voice_mode(cfg: Config, once: bool) -> int:
    from .audio import Microphone, beep, record_utterance, save_wav
    from .stt import transcribe_file
    from .tts import speak
    from .wakeword import create_listener

    listener = create_listener(cfg)
    conversation = Conversation(create_backend(cfg), cfg.system_prompt)
    print(
        f"voice mode | backend={cfg.backend} stt={cfg.stt} tts={cfg.tts} "
        f"wake word='{cfg.wake_word}' (Ctrl+C to quit)"
    )
    with Microphone(cfg.sample_rate, cfg.chunk_samples) as mic:
        while True:
            frame = mic.read()
            if not listener.feed(frame):
                continue
            print("wake word detected")
            beep(cfg.sample_rate)
            pcm = record_utterance(
                mic,
                silence_threshold=cfg.silence_threshold,
                silence_seconds=cfg.silence_seconds,
                max_utterance_seconds=cfg.max_utterance_seconds,
            )
            if pcm.size == 0:
                continue
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name
            try:
                save_wav(wav_path, pcm, cfg.sample_rate)
                text = transcribe_file(wav_path, cfg)
            finally:
                os.unlink(wav_path)
            if not text:
                continue
            print(f"you> {text}")
            if is_exit_phrase(text):
                speak("Goodbye!", cfg)
                break
            reply = conversation.ask(text)
            print(f"assistant> {reply}")
            speak(reply, cfg)
            if once:
                break
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = {}
    for name in ("backend", "stt", "tts", "wake_word"):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    cfg = Config.from_env(**overrides)
    if args.text:
        return run_text_mode(cfg, args.once)
    try:
        import sounddevice  # noqa: F401
    except OSError as exc:
        print(f"audio unavailable ({exc}); use --text mode", file=sys.stderr)
        return 2
    return run_voice_mode(cfg, args.once)
