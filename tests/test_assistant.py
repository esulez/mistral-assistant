import io
import json
import os
import sys
import types
import unittest
import wave
from unittest import mock

import tempfile

import numpy as np

from assistant.audio import wav_bytes
from assistant.brain import Conversation, EchoBackend, create_backend
from assistant.cli import is_exit_phrase
from assistant.config import Config, resolve_backend, resolve_stt, resolve_wake_mode
from assistant.lang import DEFAULT_VOICES, detect_language
from assistant.tts import kokoro_params, resolve_tts
from assistant.wakeword import extract_query, normalize_text


class ConfigTest(unittest.TestCase):
    def test_defaults_with_no_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("shutil.which", return_value=None):
                cfg = Config.from_env()
        self.assertIsNone(cfg.api_key)
        self.assertEqual(cfg.backend, "echo")
        self.assertEqual(cfg.stt, "whisper")
        self.assertEqual(cfg.wake_word, "salut mistral ça va")
        self.assertEqual(cfg.wake_mode, "phrase")
        self.assertEqual(cfg.sample_rate, 16000)

    def test_env_overrides(self):
        env = {"MISTRAL_API_KEY": "test-key", "ASSISTANT_BACKEND": "api", "ASSISTANT_WAKE_WORD": "hey jarvis"}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = Config.from_env()
        self.assertEqual(cfg.backend, "api")
        self.assertEqual(cfg.stt, "mistral")
        self.assertEqual(cfg.wake_word, "hey jarvis")

    def test_resolve_backend_prefers_vibe_binary(self):
        with mock.patch("shutil.which", return_value="/usr/local/bin/vibe"):
            self.assertEqual(resolve_backend("auto", "key"), "vibe")
        with mock.patch("shutil.which", return_value=None):
            self.assertEqual(resolve_backend("auto", "key"), "api")
            self.assertEqual(resolve_backend("auto", None), "echo")

    def test_resolve_stt(self):
        self.assertEqual(resolve_stt("auto", "key"), "mistral")
        self.assertEqual(resolve_stt("auto", None), "whisper")
        self.assertEqual(resolve_stt("whisper", "key"), "whisper")


class ConversationTest(unittest.TestCase):
    def test_echo_conversation_tracks_history(self):
        conv = Conversation(EchoBackend(), "system")
        reply = conv.ask("hello there")
        self.assertEqual(reply, "Echo: hello there")
        self.assertEqual(
            conv.history,
            [
                {"role": "user", "content": "hello there"},
                {"role": "assistant", "content": "Echo: hello there"},
            ],
        )
        conv.reset()
        self.assertEqual(conv.history, [])

    def test_create_backend_echo(self):
        cfg = Config.from_env(backend="echo")
        self.assertIsInstance(create_backend(cfg), EchoBackend)


class CliHelpersTest(unittest.TestCase):
    def test_exit_phrases(self):
        for phrase in ("goodbye", "Goodbye!", " stop listening ", "EXIT."):
            self.assertTrue(is_exit_phrase(phrase), phrase)
        for phrase in ("goodbye everyone", "hello", ""):
            self.assertFalse(is_exit_phrase(phrase), phrase)

    def test_handle_query_survives_backend_error(self):
        from assistant.cli import _handle_query

        class Boom:
            def ask(self, text):
                raise RuntimeError("api down")

        cfg = Config.from_env(tts="none")
        self.assertFalse(_handle_query("hello", Boom(), cfg))

    def test_text_mode_survives_backend_error(self):
        from assistant import cli

        class Boom:
            def ask(self, text):
                raise RuntimeError("down")

        cfg = Config.from_env(backend="echo", tts="none")
        with mock.patch.object(cli, "create_backend", return_value=Boom()):
            with mock.patch("builtins.input", side_effect=["hello", "goodbye"]):
                rc = cli.run_text_mode(cfg, once=False)
        self.assertEqual(rc, 0)

    def test_vibe_backend_uses_devnull_stdin(self):
        import subprocess as sp

        from assistant.brain import VibeBackend

        cfg = Config.from_env(backend="vibe")
        backend = VibeBackend(cfg)
        with mock.patch("assistant.brain.subprocess.run") as run:
            run.return_value = mock.Mock(returncode=0, stdout="ok", stderr="")
            out = backend.reply([{"role": "user", "content": "hi"}], "sys")
        self.assertEqual(out, "ok")
        self.assertIs(run.call_args.kwargs.get("stdin"), sp.DEVNULL)


class AudioTest(unittest.TestCase):
    def test_wav_bytes_roundtrip(self):
        pcm = (np.sin(np.arange(1600) / 10) * 10000).astype(np.int16)
        data = wav_bytes(pcm, 16000)
        with wave.open(io.BytesIO(data), "rb") as wav:
            self.assertEqual(wav.getframerate(), 16000)
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getnframes(), 1600)


class WakePhraseTest(unittest.TestCase):
    def test_normalize_strips_accents_and_punctuation(self):
        self.assertEqual(normalize_text("Salut Mistral, ça va ?"), "salut mistral ca va")

    def test_extract_query_returns_none_without_phrase(self):
        self.assertIsNone(extract_query("bonjour", "salut mistral ça va"))

    def test_extract_query_phrase_only(self):
        self.assertEqual(extract_query("Salut Mistral, ça va ?", "salut mistral ça va"), "")

    def test_extract_query_with_question(self):
        self.assertEqual(
            extract_query("salut mistral ça va, quelle heure est-il ?", "salut mistral ça va"),
            "quelle heure est il",
        )

    def test_resolve_wake_mode(self):
        self.assertEqual(resolve_wake_mode("auto", None, "alexa"), "model")
        self.assertEqual(resolve_wake_mode("auto", None, "hey jarvis"), "model")
        self.assertEqual(resolve_wake_mode("auto", None, "salut mistral ça va"), "phrase")
        self.assertEqual(resolve_wake_mode("auto", "/tmp/custom.onnx", "anything"), "model")
        self.assertEqual(resolve_wake_mode("phrase", None, "alexa"), "phrase")
        self.assertEqual(resolve_wake_mode("model", None, "salut mistral ça va"), "model")

    def test_extract_query_fuzzy_matches_stt_mishearing(self):
        self.assertEqual(
            extract_query("salut mistral sa va, quelle heure est-il ?", "salut mistral ça va"),
            "quelle heure est il",
        )

    def test_extract_query_multiple_phrases(self):
        wake = "salut mistral ça va|hey mistral"
        self.assertEqual(extract_query("hey mistral, what time is it", wake), "what time is it")
        self.assertEqual(extract_query("Salut Mistral, ça va ?", wake), "")
        self.assertIsNone(extract_query("bonjour", wake))


class LanguageTest(unittest.TestCase):
    def test_detect_language(self):
        self.assertEqual(detect_language("Hello, how are you today?"), "en")
        self.assertEqual(detect_language("Bonjour, comment ça va aujourd'hui ?"), "fr")
        self.assertEqual(detect_language("Ciao, come stai? Grazie mille"), "it")
        self.assertEqual(detect_language("¿Hola, cómo estás? Gracias"), "es")
        self.assertEqual(detect_language("The weather is nice today"), "en")

    def test_kokoro_params_autodetects_per_language(self):
        cfg = Config.from_env(tts="kokoro")
        self.assertEqual(kokoro_params("Hello there", cfg), ("a", DEFAULT_VOICES["en"]))
        self.assertEqual(kokoro_params("Bonjour, ça va ?", cfg), ("f", DEFAULT_VOICES["fr"]))
        self.assertEqual(kokoro_params("Ciao, come stai?", cfg), ("i", DEFAULT_VOICES["it"]))
        self.assertEqual(kokoro_params("¿Hola, cómo estás?", cfg), ("e", DEFAULT_VOICES["es"]))

    def test_kokoro_params_explicit_lang_and_voice(self):
        cfg = Config.from_env(tts="kokoro", tts_lang="f", tts_voice="ff_siwis")
        self.assertEqual(kokoro_params("Hello", cfg), ("f", "ff_siwis"))

    def test_kokoro_params_keeps_matching_voice(self):
        cfg = Config.from_env(tts="kokoro", tts_voice="af_sarah")
        self.assertEqual(kokoro_params("Hello", cfg), ("a", "af_sarah"))
        self.assertEqual(kokoro_params("Bonjour", cfg)[0], "f")


class MistralFallbackTest(unittest.TestCase):
    def _broken_sdk(self):
        return mock.patch.dict(sys.modules, {"mistralai": types.ModuleType("mistralai")})

    def test_api_backend_falls_back_to_rest_when_sdk_broken(self):
        cfg = Config.from_env(backend="api", api_key="k")
        with self._broken_sdk():
            backend = create_backend(cfg)
        self.assertIsNone(backend.client)
        with mock.patch("assistant.rest.chat_complete", return_value="pong") as rest_call:
            reply = backend.reply([{"role": "user", "content": "hi"}], "sys")
        self.assertEqual(reply, "pong")
        rest_call.assert_called_once()
        args = rest_call.call_args.args
        self.assertEqual(args[0], "k")
        self.assertEqual(args[2], [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}])

    def test_main_text_mode_works_with_broken_sdk_via_rest(self):
        from assistant.cli import main

        with mock.patch.dict(os.environ, {"MISTRAL_API_KEY": "k"}, clear=True):
            with self._broken_sdk():
                with mock.patch("assistant.rest.chat_complete", return_value="pong"):
                    with mock.patch("builtins.input", return_value="hello"):
                        rc = main(["--text", "--backend", "api", "--once"])
        self.assertEqual(rc, 0)

    def test_stt_mistral_falls_back_to_rest_when_sdk_broken(self):
        from assistant import stt

        cfg = Config.from_env(stt="mistral", api_key="k")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"RIFF....")
            path = tmp.name
        try:
            with self._broken_sdk():
                with mock.patch("assistant.rest.transcribe", return_value="salut") as rest_call:
                    text = stt.transcribe_file(path, cfg)
            self.assertEqual(text, "salut")
            rest_call.assert_called_once()
        finally:
            os.unlink(path)


class RestTest(unittest.TestCase):
    def _fake_urlopen(self, captured, response_payload):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(response_payload).encode()

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["auth"] = request.headers.get("Authorization")
            captured["content_type"] = request.headers.get("Content-type")
            captured["data"] = request.data
            return FakeResponse()

        return mock.patch("urllib.request.urlopen", side_effect=fake_urlopen)

    def test_chat_complete_posts_and_parses(self):
        from assistant import rest

        captured = {}
        payload = {"choices": [{"message": {"content": " hi "}}]}
        with self._fake_urlopen(captured, payload):
            out = rest.chat_complete("k", "m", [{"role": "user", "content": "q"}])
        self.assertEqual(out, "hi")
        self.assertEqual(captured["url"], "https://api.mistral.ai/v1/chat/completions")
        self.assertEqual(captured["auth"], "Bearer k")
        body = json.loads(captured["data"].decode())
        self.assertEqual(body, {"model": "m", "messages": [{"role": "user", "content": "q"}]})

    def test_chat_complete_joins_list_content_parts(self):
        from assistant import rest

        captured = {}
        payload = {
            "choices": [
                {"message": {"content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": " there"}]}}
            ]
        }
        with self._fake_urlopen(captured, payload):
            out = rest.chat_complete("k", "m", [])
        self.assertEqual(out, "Hello there")

    def test_chat_complete_raises_on_missing_choices(self):
        from assistant import rest

        with self._fake_urlopen({}, {"error": "bad"}):
            with self.assertRaises(RuntimeError):
                rest.chat_complete("k", "m", [])

    def test_extract_text_variants(self):
        from assistant.rest import extract_text

        self.assertEqual(extract_text(" hi "), "hi")
        self.assertEqual(extract_text([{"type": "text", "text": "a"}, {"text": "b"}]), "ab")
        self.assertEqual(extract_text(None), "")
        self.assertEqual(extract_text([]), "")

    def test_transcribe_sends_multipart(self):
        from assistant import rest

        captured = {}
        with self._fake_urlopen(captured, {"text": " salut mistral "}):
            out = rest.transcribe("k", "voxtral-mini-latest", b"RIFFDATA", "utt.wav")
        self.assertEqual(out, "salut mistral")
        self.assertEqual(captured["url"], "https://api.mistral.ai/v1/audio/transcriptions")
        self.assertIn("multipart/form-data", captured["content_type"])
        self.assertIn(b'name="model"', captured["data"])
        self.assertIn(b"voxtral-mini-latest", captured["data"])
        self.assertIn(b'filename="utt.wav"', captured["data"])
        self.assertIn(b"RIFFDATA", captured["data"])


class SttDependencyTest(unittest.TestCase):
    def test_whisper_missing_dependency_has_actionable_error(self):
        from assistant import stt

        stt._whisper_model = None
        self.addCleanup(setattr, stt, "_whisper_model", None)
        cfg = Config.from_env(stt="whisper", api_key=None)
        with mock.patch.dict(sys.modules, {"faster_whisper": None}):
            with self.assertRaises(RuntimeError) as ctx:
                stt._whisper(cfg)
        self.assertIn("requirements-voice.txt", str(ctx.exception))


class KokoroSynthesizeTest(unittest.TestCase):
    def test_accepts_numpy_chunks_from_pipeline(self):
        from assistant import tts as tts_mod

        fake_kokoro = types.ModuleType("kokoro")

        class FakePipeline:
            def __init__(self, lang_code):
                self.lang_code = lang_code

            def __call__(self, text, voice):
                yield "gs", "ps", np.array([0.1, 0.2], dtype=np.float32)
                yield "gs", "ps", np.array([0.3], dtype=np.float32)

        fake_kokoro.KPipeline = FakePipeline
        fake_torch = types.ModuleType("torch")
        captured = {}

        class FakeTensor:
            def __init__(self, arr):
                self.arr = arr

            def cpu(self):
                return self

            def numpy(self):
                return self.arr

        fake_torch.as_tensor = lambda arr: FakeTensor(arr)
        fake_sf = types.ModuleType("soundfile")

        def fake_write(buf, samples, rate, format, subtype=None):
            captured["samples"] = samples
            captured["rate"] = rate
            captured["subtype"] = subtype
            buf.write(b"WAV")

        fake_sf.write = fake_write
        modules = {"kokoro": fake_kokoro, "torch": fake_torch, "soundfile": fake_sf}
        tts_mod._kokoro_pipelines.clear()
        self.addCleanup(tts_mod._kokoro_pipelines.clear)
        with mock.patch.dict(sys.modules, modules):
            data, rate = tts_mod._kokoro_synthesize("hello", Config.from_env(tts="kokoro"))
        self.assertEqual(rate, 24000)
        self.assertEqual(data, b"WAV")
        self.assertEqual(captured["subtype"], "PCM_16")
        self.assertEqual(list(captured["samples"]), [0.1, 0.2, 0.3])


class TtsTest(unittest.TestCase):
    def test_resolve_tts_explicit(self):
        self.assertEqual(resolve_tts("none"), "none")
        self.assertEqual(resolve_tts("kokoro"), "kokoro")

    def test_resolve_tts_auto_ends_with_none(self):
        engines = resolve_tts("auto")
        self.assertIn(engines, ("kokoro", "piper", "espeak", "none"))


if __name__ == "__main__":
    unittest.main()
