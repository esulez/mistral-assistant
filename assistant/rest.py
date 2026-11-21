import json
import urllib.request
import uuid

API_BASE = "https://api.mistral.ai/v1"


def _post_json(path: str, payload: dict, api_key: str, timeout: float) -> dict:
    request = urllib.request.Request(
        f"{API_BASE}{path}",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def chat_complete(api_key: str, model: str, messages: list[dict], timeout: float = 180) -> str:
    data = _post_json(
        "/chat/completions",
        {"model": model, "messages": messages},
        api_key,
        timeout,
    )
    return data["choices"][0]["message"]["content"].strip()


def _post_multipart(path: str, fields: dict, files: dict, api_key: str, timeout: float) -> dict:
    boundary = uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        body += (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
        ).encode()
    for name, (filename, content, content_type) in files.items():
        body += (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
            f"filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n"
        ).encode()
        body += content + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        f"{API_BASE}{path}",
        data=bytes(body),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def transcribe(api_key: str, model: str, wav_bytes: bytes, filename: str = "utterance.wav",
               timeout: float = 300) -> str:
    data = _post_multipart(
        "/audio/transcriptions",
        {"model": model},
        {"file": (filename, wav_bytes, "audio/wav")},
        api_key,
        timeout,
    )
    return (data.get("text") or "").strip()
