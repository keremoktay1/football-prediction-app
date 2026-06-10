"""
data_loader.py — CSV ve model dosyalarını yükler.
"""
from __future__ import annotations

import os
import sys
import pickle
import pandas as pd
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FILES, PROCESSED_DIR, ROOT_DIR

MATCH_UPDATES_PATH = os.path.join(PROCESSED_DIR, "match_updates.csv")

# Modeller için olası dizinler (en sık kullanılan önce)
_MODEL_SEARCH_DIRS = [
    os.path.join(ROOT_DIR, "models"),
    PROCESSED_DIR,
    os.path.join(ROOT_DIR, "data"),
]

# Normalize edilmiş sütun adları için olası isimler
_P_HOME_ALIASES  = {"p_home", "home_prob", "prob_home", "home_win_prob", "prob_h", "p_h"}
_P_DRAW_ALIASES  = {"p_draw", "draw_prob", "prob_draw", "draw_win_prob", "prob_d", "p_d"}
_P_AWAY_ALIASES  = {"p_away", "away_prob", "prob_away", "away_win_prob", "prob_a", "p_a"}


# ──────────────────────────────────────────────
# Tahmin dosyası
# ──────────────────────────────────────────────

def load_predictions() -> Optional[pd.DataFrame]:
    """
    predictions_latest.csv yükler.
    Olasılık sütunlarını p_home / p_draw / p_away olarak normalize eder.
    Yoksa None döner.
    """
    path = FILES["predictions"]
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        col_map = {}
        for col in df.columns:
            lower = col.lower().strip()
            if lower in _P_HOME_ALIASES:
                col_map[col] = "p_home"
            elif lower in _P_DRAW_ALIASES:
                col_map[col] = "p_draw"
            elif lower in _P_AWAY_ALIASES:
                col_map[col] = "p_away"
        if col_map:
            df = df.rename(columns=col_map)
        return df
    except Exception:
        return None


# ──────────────────────────────────────────────
# Maç güncellemeleri
# ──────────────────────────────────────────────

def load_match_updates() -> pd.DataFrame:
    """
    match_updates.csv yükler.
    Dosya yoksa boş DataFrame döner.
    """
    if not os.path.exists(MATCH_UPDATES_PATH):
        return pd.DataFrame(columns=["match_id", "home_score", "away_score", "updated_at"])
    try:
        df = pd.read_csv(MATCH_UPDATES_PATH)
        df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce").astype("Int64")
        df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce").astype("Int64")
        df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce").astype("Int64")
        return df.dropna(subset=["match_id"])
    except Exception:
        return pd.DataFrame(columns=["match_id", "home_score", "away_score", "updated_at"])


def save_match_update(match_id: int, home_score: int, away_score: int) -> None:
    """Maç sonucunu match_updates.csv'ye ekler veya günceller."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    updates = load_match_updates()
    now = datetime.now().isoformat()

    mask = updates["match_id"] == match_id
    if mask.any():
        updates.loc[mask, ["home_score", "away_score", "updated_at"]] = [
            home_score, away_score, now
        ]
    else:
        new_row = pd.DataFrame(
            [{"match_id": match_id, "home_score": home_score,
              "away_score": away_score, "updated_at": now}]
        )
        updates = pd.concat([updates, new_row], ignore_index=True)

    updates.to_csv(MATCH_UPDATES_PATH, index=False)


def delete_match_update(match_id: int) -> None:
    """Maç sonucunu siler."""
    if not os.path.exists(MATCH_UPDATES_PATH):
        return
    updates = load_match_updates()
    updates = updates[updates["match_id"] != match_id]
    updates.to_csv(MATCH_UPDATES_PATH, index=False)


# ──────────────────────────────────────────────
# Fikstür dosyaları
# ──────────────────────────────────────────────

PLAYOFF_OVERRIDES_PATH = os.path.join(PROCESSED_DIR, "playoff_overrides.csv")


def load_playoff_overrides() -> dict:
    """
    playoff_overrides.csv → {slot_name: actual_team} dict.
    Boş actual_team değerleri dahil edilmez.
    """
    if not os.path.exists(PLAYOFF_OVERRIDES_PATH):
        return {}
    try:
        df = pd.read_csv(PLAYOFF_OVERRIDES_PATH, dtype=str)
        result = {}
        for _, row in df.iterrows():
            slot = str(row.get("slot_name", "")).strip()
            team = str(row.get("actual_team", "")).strip()
            if slot and team and team.lower() not in ("", "nan", "none"):
                result[slot] = team
        return result
    except Exception:
        return {}


def save_playoff_override(slot_name: str, actual_team: str) -> None:
    """playoff_overrides.csv'de slot_name için actual_team değerini günceller."""
    SLOTS = [
        "UEFA Playoff A", "UEFA Playoff B", "UEFA Playoff C", "UEFA Playoff D",
        "FIFA Playoff 1", "FIFA Playoff 2",
    ]
    overrides = {}
    if os.path.exists(PLAYOFF_OVERRIDES_PATH):
        try:
            df = pd.read_csv(PLAYOFF_OVERRIDES_PATH, dtype=str)
            for _, row in df.iterrows():
                s = str(row.get("slot_name", "")).strip()
                t = str(row.get("actual_team", "")).strip()
                if s:
                    overrides[s] = t if t.lower() not in ("nan", "none") else ""
        except Exception:
            pass
    overrides[slot_name] = actual_team
    rows = [{"slot_name": s, "actual_team": overrides.get(s, "")} for s in SLOTS]
    pd.DataFrame(rows).to_csv(PLAYOFF_OVERRIDES_PATH, index=False)


def load_fixtures() -> Optional[pd.DataFrame]:
    """GROUP_FIXTURES.CSV yükler; playoff_overrides.csv varsa takım adlarını günceller."""
    path = FILES["group_fixtures"]
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        overrides = load_playoff_overrides()
        if overrides:
            df["home_team"] = df["home_team"].replace(overrides)
            df["away_team"] = df["away_team"].replace(overrides)
        return df
    except Exception:
        return None


def load_knockout_slots() -> Optional[pd.DataFrame]:
    """KNOCKOUT_SLOTS.CSV yükler."""
    path = FILES["knockout_slots"]
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ──────────────────────────────────────────────
# İşlenmiş feature dosyaları (notebook 02/03 çıktısı)
# ──────────────────────────────────────────────

def load_features_2026() -> Optional[pd.DataFrame]:
    """
    features_2026_fixtures.csv yükler (notebook 02 çıktısı).
    predictions_latest.csv henüz üretilmemişse özellikleri doğrudan kullanmak için.
    """
    path = os.path.join(PROCESSED_DIR, "features_2026_fixtures.csv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, parse_dates=["date_utc"])
    except Exception:
        return None


# ──────────────────────────────────────────────
# Elo dereceleri
# ──────────────────────────────────────────────

def load_elo_ratings() -> Optional[pd.DataFrame]:
    """
    Elo rating CSV'sini yükler.
    En son snapshot'ı döner (en yüksek tarihli satırlar).
    """
    path = FILES.get("elo")
    if not path or not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if "snapshot_date" in df.columns:
            df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
            df = df.dropna(subset=["snapshot_date"])
            latest = df["snapshot_date"].max()
            df = df[df["snapshot_date"] == latest]
        return df
    except Exception:
        return None


# ──────────────────────────────────────────────
# Model dosyaları
# ──────────────────────────────────────────────

def _find_model_file(filename: str) -> Optional[str]:
    """Model dosyasını olası dizinlerde arar, bulursa tam yolu döner."""
    for d in _MODEL_SEARCH_DIRS:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    return None


def load_models() -> Optional[dict]:
    """
    *.pkl / *.joblib model dosyalarını yükler.
    En az bir dosya bulunursa dict döner; yoksa None.
    """
    try:
        import importlib
        joblib_mod = importlib.import_module("joblib")
    except ImportError:
        return None

    # Dosya isimleri notebook 03_model_training.ipynb çıktısıyla birebir eşleşir
    model_candidates = {
        "lr_model":        ["lr_model.pkl",        "lr_model.joblib"],
        "preprocessor":    ["preprocessor.pkl",    "preprocessor.joblib"],
        "home_goal_model": ["home_goal_model.pkl", "home_goal_model.joblib"],
        "away_goal_model": ["away_goal_model.pkl", "away_goal_model.joblib"],
        "poisson_imputer": ["poisson_imputer.pkl", "poisson_imputer.joblib"],
        "poisson_scaler":  ["poisson_scaler.pkl",  "poisson_scaler.joblib"],
        # İleride eklenecek modeller
        "rf_model":        ["rf_model.pkl",        "rf_model.joblib"],
        "xgb_model":       ["xgb_model.pkl",       "xgb_model.joblib"],
    }

    models: dict = {}
    for name, filenames in model_candidates.items():
        for fname in filenames:
            path = _find_model_file(fname)
            if path is None:
                continue
            try:
                models[name] = joblib_mod.load(path)
                break  # İlk bulunan yüklendi
            except Exception:
                pass

    return models if models else None


# ──────────────────────────────────────────────
# Elo map yardımcısı
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# Model karşılaştırma sonuçları
# ──────────────────────────────────────────────

def load_model_comparison() -> Optional[pd.DataFrame]:
    """data/processed/model_comparison.csv yükler."""
    path = os.path.join(PROCESSED_DIR, "model_comparison.csv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def build_elo_map(elo_df: Optional[pd.DataFrame] = None) -> dict:
    """
    team_name → elo_rating sözlüğü döner.
    Bilinmeyen takımlar (playoff takımları) için varsayılan 1700 kullanılır.
    """
    DEFAULT_ELO = 1700
    if elo_df is None:
        elo_df = load_elo_ratings()
    if elo_df is None or elo_df.empty:
        return {}
    elo_map = {}
    if "country" in elo_df.columns and "rating" in elo_df.columns:
        for _, row in elo_df.iterrows():
            elo_map[str(row["country"])] = float(row["rating"])
    return elo_map


def load_per_model_predictions() -> Optional[pd.DataFrame]:
    """predictions_all_models.csv yükler (6 model × 72 maç = 432 satır)."""
    path = os.path.join(PROCESSED_DIR, "predictions_all_models.csv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None
