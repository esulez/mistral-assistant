import io
import os
import sys
import types
import unittest
import wave
from unittest import mock

import numpy as np

from assistant.audio import wav_bytes
from assistant.brain import Conversation, EchoBackend, create_backend
from assistant.cli import is_exit_phrase
from assistant.config import Config, resolve_backend, resolve_stt, resolve_wake_mode
from assistant.tts import resolve_tts
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


class MistralImportGuardTest(unittest.TestCase):
    def _broken_sdk(self):
        return mock.patch.dict(sys.modules, {"mistralai": types.ModuleType("mistralai")})

    def test_api_backend_raises_actionable_error_when_sdk_broken(self):
        cfg = Config.from_env(backend="api", api_key="k")
        with self._broken_sdk(), self.assertRaises(RuntimeError) as ctx:
            create_backend(cfg)
        self.assertIn("force-reinstall", str(ctx.exception))

    def test_main_returns_2_with_clear_message_when_sdk_broken(self):
        from assistant.cli import main

        with mock.patch.dict(os.environ, {"MISTRAL_API_KEY": "k"}, clear=True):
            with self._broken_sdk():
                rc = main(["--text", "--backend", "api", "--once"])
        self.assertEqual(rc, 2)


class TtsTest(unittest.TestCase):
    def test_resolve_tts_explicit(self):
        self.assertEqual(resolve_tts("none"), "none")
        self.assertEqual(resolve_tts("kokoro"), "kokoro")

    def test_resolve_tts_auto_ends_with_none(self):
        engines = resolve_tts("auto")
        self.assertIn(engines, ("kokoro", "piper", "espeak", "none"))


if __name__ == "__main__":
    unittest.main()
