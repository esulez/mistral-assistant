import json
import urllib.request
import uuid

API_BASE = "https://api.mistral.ai/v1"
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"


def _post_json(path: str, payload: dict, api_key: str, timeout: float, api_base: str = API_BASE) -> dict:
    request = urllib.request.Request(
        f"{api_base}{path}",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def extract_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text is not None:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content or "").strip()


def chat_complete(
    api_key: str, model: str, messages: list[dict], timeout: float = 180, api_base: str = API_BASE
) -> str:
    data = _post_json(
        "/chat/completions",
        {"model": model, "messages": messages},
        api_key,
        timeout,
        api_base,
    )
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"unexpected response from Mistral API: {data}")
    message = choices[0].get("message") or {}
    return extract_text(message.get("content", ""))


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
               language: str | None = None, timeout: float = 300) -> str:
    fields = {"model": model}
    if language:
        fields["language"] = language
    data = _post_multipart(
        "/audio/transcriptions",
        fields,
        {"file": (filename, wav_bytes, "audio/wav")},
        api_key,
        timeout,
    )
    return (data.get("text") or "").strip()
