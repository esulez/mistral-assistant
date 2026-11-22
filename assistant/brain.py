import shutil
import subprocess
from datetime import datetime

from .config import Config
from .rest import extract_text


def import_mistral():
    try:
        from mistralai import Mistral
    except ImportError as exc:
        raise RuntimeError(
            "the mistralai SDK is missing or broken (no 'Mistral' class). Fix with: "
            "pip install --force-reinstall --no-cache-dir 'mistralai>=1.0' -- and if a "
            "'mistralai/' folder sits in the project root, remove it, it shadows the "
            "installed package"
        ) from exc
    return Mistral


class EchoBackend:
    def reply(self, history: list[dict], system_prompt: str) -> str:
        return f"Echo: {history[-1]['content']}"


class ApiBackend:
    def __init__(self, cfg: Config):
        if not cfg.api_key:
            raise RuntimeError("MISTRAL_API_KEY is required for backend=api")
        self.cfg = cfg
        try:
            self.client = import_mistral()(api_key=cfg.api_key)
        except RuntimeError:
            self.client = None

    def reply(self, history: list[dict], system_prompt: str) -> str:
        user_text = history[-1]["content"]
        extra = f"\nCurrent date and time: {datetime.now().strftime('%Y-%m-%d %H:%M (%A)')}"
        if self.cfg.web_search:
            from . import web

            if web.needs_web(user_text):
                results = web.search(user_text)
                if results:
                    extra += "\nWeb search results (use them if relevant):\n" + "\n".join(
                        f"- {r['title']} — {r['snippet']} ({r['url']})" for r in results
                    )
        messages = [{"role": "system", "content": system_prompt + extra}] + history
        if self.client is not None:
            response = self.client.chat.complete(model=self.cfg.model, messages=messages)
            return extract_text(response.choices[0].message.content)
        from . import rest

        return rest.chat_complete(
            self.cfg.api_key, self.cfg.model, messages, timeout=self.cfg.vibe_timeout
        )


class VibeBackend:
    def __init__(self, cfg: Config):
        if not shutil.which("vibe"):
            raise RuntimeError("the `vibe` CLI is not on PATH; run backend=api or install Vibe")
        self.cfg = cfg

    def reply(self, history: list[dict], system_prompt: str) -> str:
        prompt = history[-1]["content"]
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"
        base_cmd = ["vibe", "-p", prompt, "--output", "text", "--max-turns", "10"]
        attempts = [["vibe", "-c", "-p", prompt, "--output", "text", "--max-turns", "10"], base_cmd]
        last_error = ""
        for cmd in attempts:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    stdin=subprocess.DEVNULL,
                    timeout=self.cfg.vibe_timeout,
                )
            except subprocess.TimeoutExpired:
                last_error = "vibe timed out"
                continue
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            last_error = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"vibe backend failed: {last_error}")


def create_backend(cfg: Config):
    if cfg.backend == "echo":
        return EchoBackend()
    if cfg.backend == "api":
        return ApiBackend(cfg)
    if cfg.backend == "vibe":
        return VibeBackend(cfg)
    raise ValueError(f"unknown backend: {cfg.backend}")


class Conversation:
    def __init__(self, backend, system_prompt: str):
        self.backend = backend
        self.system_prompt = system_prompt
        self.history: list[dict] = []

    def ask(self, text: str) -> str:
        self.history.append({"role": "user", "content": text})
        reply = self.backend.reply(self.history, self.system_prompt)
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self) -> None:
        self.history.clear()
