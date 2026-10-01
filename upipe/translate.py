import os
import re
import threading
import time

import requests
from langdetect import detect_langs, DetectorFactory, LangDetectException
from exorde_data import Translation, Language, Translated, Item

from evaluate_token_count import evaluate_token_count, MAX_MODEL_TOKENS

DetectorFactory.seed = 0

LANG_CONFIDENCE_THRESHOLD = float(os.getenv("LANG_CONFIDENCE_THRESHOLD", "0.90"))
TRANSLATOR_URL = os.getenv("TRANSLATOR_URL", "http://translator:8003").rstrip("/")
TRANSLATOR_TIMEOUT = float(os.getenv("TRANSLATOR_TIMEOUT", "90"))

LANG_TO_NLLB = {
    "af": "afr_Latn", "ar": "arb_Arab", "bg": "bul_Cyrl", "bn": "ben_Beng", "ca": "cat_Latn",
    "cs": "ces_Latn", "cy": "cym_Latn", "da": "dan_Latn", "de": "deu_Latn", "el": "ell_Grek",
    "es": "spa_Latn", "et": "est_Latn", "fa": "pes_Arab", "fi": "fin_Latn", "fr": "fra_Latn",
    "gu": "guj_Gujr", "he": "heb_Hebr", "hi": "hin_Deva", "hr": "hrv_Latn", "hu": "hun_Latn",
    "id": "ind_Latn", "it": "ita_Latn", "ja": "jpn_Jpan", "kn": "kan_Knda", "ko": "kor_Hang",
    "lt": "lit_Latn", "lv": "lvs_Latn", "mk": "mkd_Cyrl", "ml": "mal_Mlym", "mr": "mar_Deva",
    "ne": "npi_Deva", "nl": "nld_Latn", "no": "nob_Latn", "pa": "pan_Guru", "pl": "pol_Latn",
    "pt": "por_Latn", "ro": "ron_Latn", "ru": "rus_Cyrl", "sk": "slk_Latn", "sl": "slv_Latn",
    "so": "som_Latn", "sq": "als_Latn", "sv": "swe_Latn", "sw": "swh_Latn", "ta": "tam_Taml",
    "te": "tel_Telu", "th": "tha_Thai", "tl": "tgl_Latn", "tr": "tur_Latn", "uk": "ukr_Cyrl",
    "ur": "urd_Arab", "vi": "vie_Latn", "zh-cn": "zho_Hans", "zh-tw": "zho_Hant",
}
ISO_OVERRIDE = {"zh-cn": "zh", "zh-tw": "zh"}

_SENT_RE = re.compile(r"[^.!?…。！？\n]+[.!?…。！？]*\s*")
_local = threading.local()


class NonEnglishError(ValueError):
    pass


class TranslatorError(Exception):
    pass


def _session() -> requests.Session:
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        _local.s = s
    return s


def _call_translator(text: str, src: str):
    last = "?"
    for attempt in range(3):
        try:
            r = _session().post(
                f"{TRANSLATOR_URL}/translate",
                json={"items": [{"text": text, "src": src}]},
                timeout=TRANSLATOR_TIMEOUT,
            )
            if r.status_code == 200:
                return r.json()["translations"][0]
            last = f"HTTP {r.status_code}"
            if r.status_code != 503:
                break
        except (requests.Timeout, requests.ConnectionError) as e:
            last = repr(e)
        except (ValueError, KeyError, IndexError) as e:
            last = f"bad response: {e!r}"
            break
        if attempt < 2:
            time.sleep(1 + attempt)
    raise TranslatorError(f"translator недоступен/ошибка: {last}")


def _fit_tokens(text: str) -> str:
    if evaluate_token_count(text) <= MAX_MODEL_TOKENS:
        return text
    cur = ""
    for s in _SENT_RE.findall(text):
        cand = cur + s
        if evaluate_token_count(cand) > MAX_MODEL_TOKENS:
            break
        cur = cand
    if cur.strip():
        return cur.strip()
    words = text.split()
    lo, hi = 1, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if evaluate_token_count(" ".join(words[:mid])) <= MAX_MODEL_TOKENS:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo])


def translate(item: Item, installed_languages, low_memory: bool = False) -> Translation:
    content = str(item.content)

    if not content or not any(c.isalpha() for c in content):
        return Translation(language=Language(""), translation=Translated(""))

    try:
        candidates = detect_langs(content)
    except LangDetectException:
        return Translation(language=Language(""), translation=Translated(""))

    if not candidates:
        return Translation(language=Language(""), translation=Translated(""))

    top = candidates[0]
    if top.lang == "en":
        return Translation(language=Language("en"), translation=Translated(content))

    if top.prob < LANG_CONFIDENCE_THRESHOLD or top.lang not in LANG_TO_NLLB:
        raise NonEnglishError(f"язык не поддержан/неуверенно: {top.lang} (p={top.prob:.2f})")

    res = _call_translator(content, LANG_TO_NLLB[top.lang])
    text = (res or {}).get("text", "").strip() if res else ""
    if not text:
        raise ValueError("No content to work with")

    text = _fit_tokens(text)
    return Translation(
        language=Language(ISO_OVERRIDE.get(top.lang, top.lang)),
        translation=Translated(text),
    )