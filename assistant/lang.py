import re

LANG_CODES = {"en": "a", "fr": "f", "it": "i", "es": "e"}

DEFAULT_VOICES = {
    "en": "af_bella",
    "fr": "ff_siwis",
    "it": "if_sara",
    "es": "ef_dora",
}

_WORDS = {
    "fr": {"le", "la", "les", "des", "du", "est", "vous", "merci", "salut", "bonjour", "oui",
           "non", "et", "un", "une", "pas", "pour", "avec", "que", "dans", "au", "aux", "bien",
           "voici", "voilà", "comment", "quelle", "heure", "aujourd'hui", "monsieur", "madame"},
    "it": {"il", "lo", "la", "le", "di", "che", "non", "per", "sono", "sei", "un", "una", "e",
           "in", "con", "su", "grazie", "ciao", "no", "questo", "questa", "come", "dove", "bene"},
    "es": {"el", "la", "los", "las", "de", "que", "no", "un", "una", "es", "en", "con", "por",
           "para", "gracias", "hola", "si", "esto", "esta", "como", "donde", "bien", "qué"},
    "en": {"the", "a", "an", "is", "are", "you", "to", "and", "of", "in", "that", "it", "what",
           "how", "please", "hello", "hi", "yes", "no", "sure", "today", "weather"},
}

_ACCENTS = {
    "fr": set("àâäçéèêëîïôöùûüÿœæ"),
    "it": set("àèéìòù"),
    "es": set("áéíóúñü¿¡"),
    "en": set(),
}

_TOKEN_RE = re.compile(r"[a-zàâäçéèêëîïôöùûüÿœæñáéíóú¿¡]+")


def detect_language(text: str) -> str:
    lowered = text.lower()
    tokens = _TOKEN_RE.findall(lowered)
    best, best_score = "en", 0
    for lang, words in _WORDS.items():
        score = sum(1 for token in tokens if token in words)
        score += 2 * sum(1 for ch in lowered if ch in _ACCENTS[lang])
        if score > best_score:
            best, best_score = lang, score
    return best
