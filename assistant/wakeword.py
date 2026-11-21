import re
import unicodedata

import numpy as np

from .audio import frame_rms
from .config import Config


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", stripped).strip()


def extract_query(text: str, wake_phrase: str) -> str | None:
    norm_phrase = normalize_text(wake_phrase)
    norm_text = normalize_text(text)
    if not norm_phrase or norm_phrase not in norm_text:
        return None
    return norm_text.split(norm_phrase, 1)[1].strip()


class OpenWakeWordListener:
    def __init__(self, cfg: Config):
        from openwakeword.model import Model
        from openwakeword.utils import download_models

        download_models()
        self.model = Model(inference_framework="onnx")
        self.threshold = cfg.wake_threshold
        self.target_key = None
        if cfg.wakeword_model:
            self.model = Model(wakeword_models=[cfg.wakeword_model], inference_framework="onnx")
            self.target_key = next(iter(self.model.models))
        else:
            for name in self.model.models:
                if cfg.wake_word.lower() in name.lower():
                    self.target_key = name
                    break
            if self.target_key is None and self.model.models:
                self.target_key = next(iter(self.model.models))
        if self.target_key is None:
            raise RuntimeError("no wake word models available")

    def feed(self, frame: np.ndarray) -> bool:
        if len(frame) < 1280:
            frame = np.pad(frame, (0, 1280 - len(frame)))
        scores = self.model.predict(frame[:1280].astype(np.float32) / 32768.0)
        return float(scores.get(self.target_key, 0.0)) >= self.threshold


class EnergyListener:
    def __init__(self, cfg: Config):
        self.threshold = cfg.silence_threshold
        self.cooldown = int(cfg.sample_rate * 1.0 / cfg.chunk_samples)
        self._quiet = 0

    def feed(self, frame: np.ndarray) -> bool:
        if frame_rms(frame) >= self.threshold:
            if self._quiet >= self.cooldown:
                self._quiet = 0
                return True
            self._quiet = 0
            return False
        self._quiet += 1
        return False


def create_listener(cfg: Config):
    try:
        return OpenWakeWordListener(cfg)
    except Exception:
        return EnergyListener(cfg)
