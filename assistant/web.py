import html
import re
import urllib.parse
import urllib.request

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
SEARCH_URL = "https://lite.duckduckgo.com/lite/?q={query}"

WEB_KEYWORDS = (
    "weather", "forecast", "météo", "meteo", "tempo", "prévision", "prevision",
    "previsioni", "previsión", "wetter", "vorhersage",
    "news", "actualité", "actualite", "notizie", "noticias",
    "today", "aujourd'hui", "oggi", "hoy", "heute",
    "now", "maintenant", "adesso", "ahora",
    "current", "actuel", "attuelle", "attuale", "actual", "aktuell",
    "time", "heure", "ora", "hora", "uhrzeit", "date",
    "price", "prix", "prezzo", "precio", "preis",
    "exchange rate", "taux de change", "cambio", "wechselkurs",
    "score", "résultat", "resultat", "risultato", "resultado", "ergebnis",
    "search", "look up", "lookup",
)


def needs_web(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(r"\b" + re.escape(keyword) + r"\b", lowered) for keyword in WEB_KEYWORDS)


def _strip_tags(fragment: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def _decode_url(href: str) -> str:
    parsed = urllib.parse.urlparse(href)
    if parsed.query:
        uddg = urllib.parse.parse_qs(parsed.query).get("uddg")
        if uddg:
            return uddg[0]
    if href.startswith("//"):
        return "https:" + href
    return href


def search(query: str, max_results: int = 5, timeout: float = 15) -> list[dict]:
    url = SEARCH_URL.format(query=urllib.parse.quote(query))
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        page = response.read().decode("utf-8", "ignore")
    snippets = re.findall(r"class=[\"']result-snippet[\"'][^>]*>(.*?)</td>", page, re.S)
    results = []
    for attrs, title in re.findall(r"<a\b([^>]*)>(.*?)</a>", page, re.S):
        if "result-link" not in attrs:
            continue
        href_match = re.search(r"href=[\"']([^\"']+)[\"']", attrs)
        if not href_match:
            continue
        snippet = _strip_tags(snippets[len(results)]) if len(results) < len(snippets) else ""
        results.append({
            "title": _strip_tags(title),
            "url": _decode_url(href_match.group(1)),
            "snippet": snippet,
        })
        if len(results) >= max_results:
            break
    return results
