"""
fetch_odds.py

The-Odds-API entegrasyonu (ücretsiz: 500 istek/ay).
API key .streamlit/secrets.toml'da ODDS_API_KEY olarak saklanır.

Kullanım:
  python scripts/fetch_odds.py

Çıktı:
  data/processed/odds_cache.csv
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

import pandas as pd

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(APP_DIR, "data", "processed")
RAW_DIR       = os.path.join(APP_DIR, "data", "raw")

SPORT    = "soccer_fifa_world_cup"
BASE_URL = "https://api.the-odds-api.com/v4/sports"


def get_api_key() -> str:
    """API key'i env değişkeninden veya Streamlit secrets'tan çeker."""
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        try:
            import streamlit as st
            key = str(st.secrets.get("ODDS_API_KEY", "")).strip()
        except Exception:
            pass
    return key


def fetch_and_save() -> pd.DataFrame | None:
    """Odds verisini API'den çeker, GROUP_FIXTURES.CSV ile eşleştirir ve kaydeder."""
    try:
        import requests
    except ImportError:
        print("[ERROR] 'requests' paketi bulunamadı. pip install requests>=2.31")
        return None

    api_key = get_api_key()
    if not api_key:
        print("[ERROR] ODDS_API_KEY bulunamadı.")
        print("  .streamlit/secrets.toml dosyasına ODDS_API_KEY = '...' ekleyin.")
        return None

    url    = f"{BASE_URL}/{SPORT}/odds/"
    params = {"apiKey": api_key, "regions": "eu", "markets": "h2h"}

    print(f"API isteği gönderiliyor: {url}")
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(f"[ERROR] HTTP {resp.status_code}: {e}")
        return None
    except requests.RequestException as e:
        print(f"[ERROR] İstek hatası: {e}")
        return None

    data = resp.json()
    print(f"  {len(data)} etkinlik alındı")

    # GROUP_FIXTURES.CSV ile eşleştirme için yükle
    fixtures_path = os.path.join(RAW_DIR, "GROUP_FIXTURES.CSV")
    if not os.path.isfile(fixtures_path):
        print(f"[WARN] GROUP_FIXTURES.CSV bulunamadı: {fixtures_path}")
        fixtures = pd.DataFrame(columns=["match_id", "home_team", "away_team"])
    else:
        fixtures = pd.read_csv(fixtures_path)

    rows = []
    fetched_at = datetime.utcnow().isoformat()

    for event in data:
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        bookmakers = event.get("bookmakers", [])

        if not bookmakers:
            continue

        # İlk bookmaker'ı kullan
        bkm = bookmakers[0]
        for mkt in bkm.get("markets", []):
            if mkt["key"] != "h2h":
                continue
            outcomes = {o["name"]: float(o["price"]) for o in mkt.get("outcomes", [])}

            odds_home = outcomes.get(home_team)
            odds_draw = outcomes.get("Draw")
            odds_away = outcomes.get(away_team)

            rows.append({
                "home_team":  home_team,
                "away_team":  away_team,
                "event_date": event.get("commence_time", ""),
                "bookmaker":  bkm.get("key", ""),
                "odds_home":  odds_home,
                "odds_draw":  odds_draw,
                "odds_away":  odds_away,
                "fetched_at": fetched_at,
            })
            break  # Sadece h2h marketi, ilk bookmaker

    if not rows:
        print("[WARN] API'den geçerli odds verisi alınamadı.")
        return None

    odds_df = pd.DataFrame(rows)

    # match_id eşleştirme (home_team + away_team üzerinden)
    if not fixtures.empty and "match_id" in fixtures.columns:
        odds_df = odds_df.merge(
            fixtures[["match_id", "home_team", "away_team"]],
            on=["home_team", "away_team"],
            how="left",
        )
    else:
        odds_df["match_id"] = None

    # Implied probabilities (vig çıkarılmadan)
    for side in ["home", "draw", "away"]:
        col = f"odds_{side}"
        imp_col = f"implied_{side}"
        odds_df[imp_col] = odds_df[col].apply(
            lambda x: round(1.0 / x, 4) if (x is not None and x > 0) else None
        )

    out_path = os.path.join(PROCESSED_DIR, "odds_cache.csv")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    odds_df.to_csv(out_path, index=False)
    print(f"✅  odds_cache.csv → {len(odds_df)} maç kaydedildi")

    matched = odds_df["match_id"].notna().sum()
    print(f"  GROUP_FIXTURES eşleşmesi: {matched} / {len(odds_df)}")
    return odds_df


if __name__ == "__main__":
    result = fetch_and_save()
    if result is None:
        sys.exit(1)
