import io
import wave

import numpy as np


def wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.astype(np.int16).tobytes())
    return buffer.getvalue()


def save_wav(path: str, pcm: np.ndarray, sample_rate: int) -> None:
    with open(path, "wb") as handle:
        handle.write(wav_bytes(pcm, sample_rate))


def play_wav_bytes(data: bytes) -> None:
    import sounddevice as sd

    with wave.open(io.BytesIO(data), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())
    if width == 2:
        dtype = np.int16
        scale = 32768.0
    elif width == 4:
        dtype = np.int32
        scale = 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {width}")
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float32) / scale
    if channels > 1:
        samples = samples.reshape(-1, channels)
    sd.play(samples, sample_rate)
    sd.wait()


def beep(sample_rate: int = 16000, frequency: float = 880.0, duration: float = 0.15) -> None:
    import sounddevice as sd

    t = np.arange(int(sample_rate * duration)) / sample_rate
    tone = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    sd.play(tone, sample_rate)
    sd.wait()


def frame_rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))


class Microphone:
    def __init__(self, sample_rate: int, chunk_samples: int):
        import sounddevice as sd

        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=chunk_samples,
        )

    def __enter__(self) -> "Microphone":
        self.stream.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stream.stop()
        self.stream.close()

    def read(self) -> np.ndarray:
        data, _overflowed = self.stream.read(self.chunk_samples)
        return data.reshape(-1).copy()


def record_utterance(mic: Microphone, silence_threshold: float, silence_seconds: float,
                     max_utterance_seconds: float) -> np.ndarray:
    frames: list[np.ndarray] = []
    silent_frames = 0
    max_frames = int(max_utterance_seconds * mic.sample_rate / mic.chunk_samples)
    silence_frames = int(silence_seconds * mic.sample_rate / mic.chunk_samples)
    speech_started = False
    while len(frames) < max_frames:
        frame = mic.read()
        energy = frame_rms(frame)
        if not speech_started:
            if energy >= silence_threshold:
                speech_started = True
                frames.append(frame)
            continue
        frames.append(frame)
        if energy < silence_threshold:
            silent_frames += 1
            if silent_frames >= silence_frames:
                break
        else:
            silent_frames = 0
    if not frames:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(frames)
