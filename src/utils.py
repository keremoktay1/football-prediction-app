"""
utils.py — Paylaşılan yardımcı fonksiyonlar.
"""
from __future__ import annotations

import pandas as pd

_MONTHS_TR = {
    "Jan": "Oca", "Feb": "Şub", "Mar": "Mar", "Apr": "Nis",
    "May": "May", "Jun": "Haz", "Jul": "Tem", "Aug": "Ağu",
    "Sep": "Eyl", "Oct": "Eki", "Nov": "Kas", "Dec": "Ara",
}

_TBD_PREFIXES = (
    "Winner", "Runner", "Best", "Loser",
    "UEFA Playoff", "FIFA Playoff",
)


def fmt_date(dt) -> str:
    """Tarihi Türkçe ay adıyla formatlar: '14 Haz 18:00'"""
    try:
        s = pd.to_datetime(dt).strftime("%d %b %H:%M")
        for en, tr in _MONTHS_TR.items():
            s = s.replace(en, tr)
        return s
    except Exception:
        return str(dt)


def team_display(name: str) -> str:
    """TBD takım adlarını [badge] formatında gösterir."""
    if str(name).startswith(_TBD_PREFIXES):
        return f"`[{name}]`"
    return str(name)


def clamp_pct(p: float) -> float:
    """Olasılığı %1–%99 aralığına kısıtlar (görüntüleme için)."""
    try:
        return max(0.01, min(0.99, float(p)))
    except (TypeError, ValueError):
        return 0.33
