"""Tiny bilingual helper: tr("中文", "English") picks one based on TRAINKIT_LANG, config or system locale."""

from __future__ import annotations

import locale
import os
from functools import lru_cache


@lru_cache(maxsize=1)
def lang() -> str:
    env = os.environ.get("TRAINKIT_LANG", "").lower()
    if env in ("zh", "en"):
        return env
    try:
        from .config import load_config

        cfg_lang = load_config().get("lang", "")
        if cfg_lang in ("zh", "en"):
            return cfg_lang
    except Exception:
        pass
    loc = ""
    try:
        loc = locale.getlocale()[0] or ""
    except Exception:
        pass
    loc = loc or os.environ.get("LANG", "")
    return "zh" if loc.lower().startswith(("zh", "chinese")) else "en"


def tr(zh: str, en: str) -> str:
    return zh if lang() == "zh" else en
