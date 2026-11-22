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
    parser.add_argument("--tts-voice", default=None, help="e.g. af_bella, af_sarah, af_heart (Kokoro US female)")
    parser.add_argument("--wake-word", default=None)
    parser.add_argument("--wake-mode", choices=["auto", "model", "phrase"], default=None)
    parser.add_argument("--output-device", default=None, help="speaker device index or name (see --list-devices)")
    parser.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    return parser


def run_text_mode(cfg: Config, once: bool) -> int:
    conversation = Conversation(create_backend(cfg), cfg.system_prompt)
    print(f"text mode | backend={cfg.backend} stt={cfg.stt} tts={cfg.tts} (type 'goodbye' to quit)", flush=True)
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if is_exit_phrase(line):
            print("assistant> Goodbye!", flush=True)
            break
        try:
            reply = conversation.ask(line)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr, flush=True)
            continue
        print(f"assistant> {reply}", flush=True)
        if once:
            break
    return 0


def _transcribe_pcm(pcm, cfg: Config) -> str:
    from .audio import save_wav
    from .stt import transcribe_file

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        save_wav(wav_path, pcm, cfg.sample_rate)
        return transcribe_file(wav_path, cfg)
    finally:
        os.unlink(wav_path)


def _handle_query(text: str, conversation: Conversation, cfg: Config) -> bool:
    from .tts import speak

    print(f"you> {text}")
    if is_exit_phrase(text):
        speak("Goodbye!", cfg)
        return True
    try:
        reply = conversation.ask(text)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        speak("Sorry, something went wrong.", cfg)
        return False
    print(f"assistant> {reply}")
    speak(reply, cfg)
    return False


def _record(mic, cfg: Config):
    from .audio import record_utterance

    return record_utterance(
        mic,
        silence_threshold=cfg.silence_threshold,
        silence_seconds=cfg.silence_seconds,
        max_utterance_seconds=cfg.max_utterance_seconds,
    )


def _run_phrase_mode(cfg: Config, once: bool, conversation: Conversation) -> int:
    from .audio import Microphone, beep
    from .wakeword import extract_query

    print(
        f"voice mode | backend={cfg.backend} stt={cfg.stt} tts={cfg.tts} "
        f"wake mode=phrase phrase='{cfg.wake_word}' (Ctrl+C to quit)"
    )
    with Microphone(cfg.sample_rate, cfg.chunk_samples) as mic:
        print("listening...", flush=True)
        while True:
            try:
                pcm = _record(mic, cfg)
                if pcm.size == 0:
                    continue
                query = extract_query(_transcribe_pcm(pcm, cfg), cfg.wake_word)
                if query is None:
                    continue
                print("wake phrase detected", flush=True)
                beep(cfg.sample_rate, device=cfg.output_device)
                if not query:
                    pcm = _record(mic, cfg)
                    if pcm.size == 0:
                        continue
                    query = _transcribe_pcm(pcm, cfg)
                    if not query:
                        continue
                if _handle_query(query, conversation, cfg):
                    break
                if once:
                    break
                print("listening...", flush=True)
            except Exception as exc:
                print(f"error: {exc}", file=sys.stderr, flush=True)
                print("listening...", flush=True)
    return 0


def run_voice_mode(cfg: Config, once: bool) -> int:
    from .audio import Microphone, beep
    from .tts import resolve_tts
    from .wakeword import create_listener

    cfg.tts = resolve_tts(cfg.tts)
    conversation = Conversation(create_backend(cfg), cfg.system_prompt)
    try:
        import sounddevice as sd

        device = cfg.output_device or sd.default.device[1]
        print(f"audio out: {sd.query_devices(device)['name']}", flush=True)
    except Exception:
        pass
    if cfg.wake_mode == "phrase":
        return _run_phrase_mode(cfg, once, conversation)
    listener = create_listener(cfg)
    print(
        f"voice mode | backend={cfg.backend} stt={cfg.stt} tts={cfg.tts} "
        f"wake mode=model word='{cfg.wake_word}' (Ctrl+C to quit)"
    )
    with Microphone(cfg.sample_rate, cfg.chunk_samples) as mic:
        print("listening...", flush=True)
        while True:
            try:
                frame = mic.read()
                if not listener.feed(frame):
                    continue
                print("wake word detected", flush=True)
                beep(cfg.sample_rate, device=cfg.output_device)
                pcm = _record(mic, cfg)
                if pcm.size == 0:
                    continue
                text = _transcribe_pcm(pcm, cfg)
                if not text:
                    continue
                if _handle_query(text, conversation, cfg):
                    break
                if once:
                    break
                print("listening...", flush=True)
            except Exception as exc:
                print(f"error: {exc}", file=sys.stderr, flush=True)
                print("listening...", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = {}
    for name in ("backend", "stt", "tts", "tts_voice", "wake_word", "wake_mode", "output_device"):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    cfg = Config.from_env(**overrides)
    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return 0
    try:
        if args.text:
            return run_text_mode(cfg, args.once)
        try:
            import sounddevice  # noqa: F401
        except OSError as exc:
            print(f"audio unavailable ({exc}); use --text mode", file=sys.stderr)
            return 2
        return run_voice_mode(cfg, args.once)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
