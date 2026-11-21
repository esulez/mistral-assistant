# mistral-assistant

A voice-first front-end for Mistral Vibe. Say a wake word (like Alexa), ask a
question out loud, and hear the answer — no typing required. The assistant can
drive the `vibe` CLI directly (so it uses your Vibe setup: connectors, skills,
knowledge bases) or talk to the Mistral API.

## How it works

```
microphone -> wake word -> record utterance -> speech-to-text
           -> brain (vibe CLI or Mistral API) -> text-to-speech -> speaker
```

- **Wake word**: default phrase is **"salut mistral ça va"**. Custom phrases are
  spotted via speech-to-text (accent- and punctuation-insensitive), so any phrase
  works. Built-in openwakeword models (e.g. `alexa`, `hey jarvis`) or a custom
  `.onnx` model are used instead when the wake word is one of those (local,
  lower latency). Falls back to an energy-based listener if the package or a
  model is unavailable.
- **Speech-to-text**: Mistral Voxtral transcription API (`stt=mistral`, needs
  `MISTRAL_API_KEY`) or local `faster-whisper` (`stt=whisper`).
- **Brain**: `backend=vibe` shells out to `vibe -p "..."` (continues the current
  Vibe session with `-c`, so conversations carry over); `backend=api` uses the
  Mistral chat completions API; `backend=echo` is a no-key test backend.
- **Text-to-speech**: local Kokoro (`tts=kokoro`), or `piper`, or `espeak` if
  installed; `tts=none` prints instead of speaking.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # core: numpy, sounddevice (mistralai optional, see below)
pip install -r requirements-voice.txt    # full voice stack (wake word, whisper, kokoro, spacy)
```

The `mistralai` SDK is optional: if it is missing or broken, the assistant falls
back to direct REST calls (`assistant/rest.py`) against `https://api.mistral.ai/v1`,
so `backend=api` and `stt=mistral` still work. Check an SDK install with
`python -c "import mistralai; print(mistralai.__file__)"` — if it prints `None`,
the install is broken (or a stray `mistralai/` folder in the project root shadows
it); either way the REST fallback keeps things running.

On Linux you may need PortAudio for the microphone:

```bash
sudo apt install libportaudio2        # Debian/Ubuntu
sudo pacman -S portaudio alsa-lib     # Arch
sudo dnf install portaudio             # Fedora
```

Set your key (for `stt=mistral` and/or `backend=api`):

```bash
export MISTRAL_API_KEY=...
```

## Run

Voice mode (needs a microphone and speaker):

```bash
python -m assistant                      # say "salut mistral ça va", then talk
python -m assistant --wake-word "hey jarvis"   # built-in model mode
python -m assistant --wake-word "hey computer"  # custom phrase mode (via STT)
python -m assistant --backend api --once # single question via the Mistral API
```

Text mode (no audio hardware, e.g. over SSH or for testing):

```bash
python -m assistant --text               # type instead of speaking
printf 'hello\ngoodbye\n' | python -m assistant --text --backend echo
```

Say "goodbye" (or press Ctrl+C) to end the session.

## Configuration

All settings can be overridden with environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MISTRAL_API_KEY` | — | Mistral API key (STT via Voxtral and/or `backend=api`) |
| `ASSISTANT_BACKEND` | `auto` | `vibe`, `api`, `echo`, or `auto` (vibe CLI if on PATH, else API key, else echo) |
| `ASSISTANT_MODEL` | `mistral-large-latest` | chat model for `backend=api` |
| `ASSISTANT_SYSTEM_PROMPT` | short voice-assistant prompt | system prompt for `backend=api` |
| `ASSISTANT_STT` | `auto` | `mistral` (Voxtral API) or `whisper` (local faster-whisper) |
| `ASSISTANT_STT_MODEL` | `voxtral-mini-latest` | transcription model for `stt=mistral` |
| `ASSISTANT_WHISPER_MODEL` | `base` | faster-whisper model for `stt=whisper` |
| `ASSISTANT_TTS` | `auto` | `kokoro`, `piper`, `espeak`, `none`, or `auto` (first available) |
| `ASSISTANT_TTS_VOICE` | `af_heart` | Kokoro voice |
| `ASSISTANT_WAKE_WORD` | `salut mistral ça va` | wake phrase; built-in model name or custom phrase |
| `ASSISTANT_WAKE_MODE` | `auto` | `model` (openwakeword/custom onnx), `phrase` (STT spotting), or `auto` (built-in name or custom model → `model`, otherwise `phrase`) |
| `ASSISTANT_WAKEWORD_MODEL` | — | path to a custom `.onnx` wake-word model |
| `ASSISTANT_WAKE_THRESHOLD` | `0.5` | detection score threshold |
| `ASSISTANT_SAMPLE_RATE` | `16000` | microphone sample rate (Hz) |
| `ASSISTANT_SILENCE_SECONDS` | `1.5` | silence that ends an utterance |
| `ASSISTANT_MAX_UTTERANCE_SECONDS` | `30` | longest single utterance |
| `ASSISTANT_VIBE_TIMEOUT` | `180` | seconds to wait for a `vibe` reply |

## Development

```bash
python -m compileall assistant tests
python -m unittest discover -s tests -v
```

## Layout

```
assistant/
  cli.py       entry point: voice loop and text mode
  config.py    environment-driven configuration
  audio.py     microphone capture, utterance recording, playback
  wakeword.py  wake-word listeners + phrase spotting (normalize/extract_query)
  stt.py       Voxtral API / faster-whisper transcription
  brain.py     backends (vibe CLI, Mistral API/SDK, echo) + conversation state
  rest.py      dependency-free Mistral REST fallback (chat + Voxtral transcription)
  tts.py       Kokoro / piper / espeak speech synthesis
tests/
  test_assistant.py
```
