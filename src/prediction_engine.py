"""
prediction_engine.py — Turnuva maç tahminleri.

Kapsam:
  - Grup aşaması  : predictions_latest.csv'den önceden hesaplanmış olasılıklar
  - Eleme aşaması : Elo tabanlı (beraberlik yok — uzatma/penaltı dahil tek kazanan)

Bu modül YALNIZCA turnuva fikstüründeki maçlar için tahmin üretir.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import poisson as scipy_poisson
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FILES, BASE_GOAL_RATE

# Notebook 03 ile birebir eşleşen sütun ve sabit isimleri
FEATURE_COLS = [
    "elo_diff",
    "form_diff",
    "weighted_form_home",
    "weighted_form_away",
    "attack_home",
    "defense_home",
    "attack_away",
    "defense_away",
    "neutral",
    # ── Yeni feature'lar ──
    "points_last_5_home",
    "points_last_5_away",
    "goal_diff_last_5_home",
    "goal_diff_last_5_away",
    "goals_for_last_5_home",
    "goals_for_last_5_away",
    "goals_against_last_5_home",
    "goals_against_last_5_away",
    "win_streak_home",
    "win_streak_away",
    "loss_streak_home",
    "loss_streak_away",
    "clean_sheet_rate_home",
    "clean_sheet_rate_away",
    "failed_to_score_rate_home",
    "failed_to_score_rate_away",
    "h2h_goal_diff",
    "common_opponent_diff",
    # ── Yeni feature'lar (36 toplam) ──
    "experience_score_home",
    "experience_score_away",
    "avg_age_home",
    "avg_age_away",
    "market_value_proxy_home",
    "market_value_proxy_away",
    "top5_league_count_home",
    "top5_league_count_away",
    # ── Trend & tutarsızlık ──
    "goal_trend_home", "goal_trend_away",
    "form_consistency_home", "form_consistency_away",
    "attack_ratio_home", "attack_ratio_away",
    "elo_form_interaction",
    # ── Yeni: kadro kalite göstergeleri ──
    "goals_per90_home", "goals_per90_away",
    "assists_per90_home", "assists_per90_away",
    # ── Yeni: antrenör ──
    "coach_win_rate_home", "coach_win_rate_away",
    "coach_wc_apps_home",  "coach_wc_apps_away",
    # ── Yeni: venue & seyahat ──
    "altitude_m",
    "travel_km_diff",
    "temp_celsius",
]

POISSON_HOME_FEATS = [
    "elo_diff", "attack_home", "defense_away",
    "weighted_form_home", "goals_for_last_5_home",
    "altitude_m", "neutral", "tournament_weight",
]
POISSON_AWAY_FEATS = [
    "elo_diff", "attack_away", "defense_home",
    "weighted_form_away", "goals_for_last_5_away",
    "altitude_m", "neutral", "tournament_weight",
]
POISSON_FEATURES = POISSON_HOME_FEATS  # backward-compat alias

UNKNOWN_ELO = 1700  # Playoff takımları için varsayılan


# ──────────────────────────────────────────────
# Yardımcı: Elo
# ──────────────────────────────────────────────

def _get_elo(team: str, elo_map: Dict[str, float]) -> float:
    """Takımın Elo derecesini döner. Bulunamazsa UNKNOWN_ELO."""
    return elo_map.get(team, UNKNOWN_ELO)


def _elo_win_prob(elo_a: float, elo_b: float, home_advantage: float = 0.0) -> float:
    """P(A kazanır) — beraberlik yok (eleme formatı)."""
    return 1.0 / (1.0 + 10.0 ** (-(elo_a - elo_b + home_advantage) / 400.0))


def _elo_to_3way(elo_home: float, elo_away: float, neutral: bool = True) -> Tuple[float, float, float]:
    """
    Elo farkından 3-sonuçlu olasılık (H/D/A).
    Grup aşaması için (beraberlik mümkün).
    Davidson modeli: ~%22 beraberlik oranı.
    """
    advantage = 0.0 if neutral else 100.0
    we = _elo_win_prob(elo_home, elo_away, advantage)
    draw_frac = 0.22
    ph  = round(we * (1.0 - draw_frac), 4)
    pa  = round((1.0 - we) * (1.0 - draw_frac), 4)
    pd_ = round(1.0 - ph - pa, 4)  # garantili olarak toplam=1
    return ph, pd_, pa


# ──────────────────────────────────────────────
# Grup maçı tahmini (predictions_latest.csv)
# ──────────────────────────────────────────────

def get_group_prediction(
    match_id: int,
    predictions: Optional[pd.DataFrame],
    elo_map: Optional[Dict[str, float]] = None,
    home_team: str = "",
    away_team: str = "",
) -> dict:
    """
    Grup maçı için tahmin döner.
    1. predictions_latest.csv'de varsa → oradaki değerleri kullan
    2. Yoksa → Elo tabanlı fallback
    """
    if predictions is not None and not predictions.empty:
        row = predictions[predictions["match_id"] == match_id]
        if not row.empty:
            r = row.iloc[0]
            return {
                "p_home":        float(r.get("p_home", 1 / 3)),
                "p_draw":        float(r.get("p_draw", 1 / 3)),
                "p_away":        float(r.get("p_away", 1 / 3)),
                "lambda_home":   float(r["lambda_home"]) if "lambda_home" in r.index else None,
                "lambda_away":   float(r["lambda_away"]) if "lambda_away" in r.index else None,
                "over_2_5":      float(r["over_2_5"])    if "over_2_5" in r.index    else None,
                "btts":          float(r["btts"])        if "btts" in r.index        else None,
                "favourite":     str(r["favourite"])     if "favourite" in r.index   else None,
                "upset_risk":    str(r["upset_risk"])    if "upset_risk" in r.index  else None,
                "source":        "predictions_latest",
            }

    # Elo fallback
    elo_map = elo_map or {}
    elo_h = _get_elo(home_team, elo_map)
    elo_a = _get_elo(away_team, elo_map)
    ph, pd_, pa = _elo_to_3way(elo_h, elo_a, neutral=True)
    lh, la = _estimate_xg(elo_h, elo_a)
    return {
        "p_home":      ph,
        "p_draw":      pd_,
        "p_away":      pa,
        "lambda_home": lh,
        "lambda_away": la,
        "over_2_5":    None,
        "btts":        None,
        "favourite":   None,
        "upset_risk":  None,
        "source":      "elo_fallback",
    }


# ──────────────────────────────────────────────
# Model tahmini (LR + Poisson pipeline — notebook 03)
# ──────────────────────────────────────────────

def predict_with_model(
    features_row: pd.Series,
    models: dict,
) -> Optional[dict]:
    """
    Yüklü modeller ile bir maç tahmini yapar.
    features_row: features_2026_fixtures.csv'den bir satır (FEATURE_COLS içermeli).

    Returns None eğer model yüklenemezse veya hata olursa.
    """
    lr_model     = models.get("lr_model")
    preprocessor = models.get("preprocessor")
    home_goal    = models.get("home_goal_model")
    away_goal    = models.get("away_goal_model")
    poi_imp      = models.get("poisson_imputer")
    poi_scl      = models.get("poisson_scaler")
    poi_imp_a    = models.get("poisson_imputer_away", poi_imp)
    poi_scl_a    = models.get("poisson_scaler_away",  poi_scl)

    if lr_model is None or preprocessor is None:
        return None

    try:
        # --- LR olasılıkları ---
        # Eksik feature'lar NaN ile doldurulur (imputer median ile işler)
        feat_dict = {c: features_row.get(c, np.nan) for c in FEATURE_COLS}
        X_feat = pd.DataFrame([feat_dict])
        X_lr_t = preprocessor.transform(X_feat)
        proba = lr_model.predict_proba(X_lr_t)[0]
        classes = list(lr_model.classes_)

        p = {"p_home": 1 / 3, "p_draw": 1 / 3, "p_away": 1 / 3}
        for i, cls in enumerate(classes):
            if cls == 0:
                p["p_home"] = float(proba[i])
            elif cls == 1:
                p["p_draw"] = float(proba[i])
            elif cls == 2:
                p["p_away"] = float(proba[i])

        # --- Poisson lambda (home/away ayrı scaler) ---
        lh = la = None
        if home_goal and away_goal and poi_imp and poi_scl:
            try:
                # Home gol modeli: home saldırı + away savunma
                X_poi_h = pd.DataFrame([{c: features_row.get(c, np.nan) for c in POISSON_HOME_FEATS}])
                X_poi_h_t = poi_scl.transform(poi_imp.transform(X_poi_h))
                lh = max(0.3, float(home_goal.predict(X_poi_h_t)[0]))

                # Away gol modeli: away saldırı + home savunma + ters elo_diff
                # Away'e özel imputer/scaler kullan (fallback: home scaler)
                away_row = {c: features_row.get(c, np.nan) for c in POISSON_AWAY_FEATS}
                away_row["elo_diff"] = -(away_row.get("elo_diff") or 0)
                X_poi_a = pd.DataFrame([away_row])
                X_poi_a_t = poi_scl_a.transform(poi_imp_a.transform(X_poi_a))
                la = max(0.3, float(away_goal.predict(X_poi_a_t)[0]))
            except Exception:
                pass

        return {"source": "lr_model", "lambda_home": lh, "lambda_away": la, **p}

    except Exception:
        return None


# ──────────────────────────────────────────────
# Eleme maçı tahmini (Elo, beraberlik yok)
# ──────────────────────────────────────────────

def get_knockout_prediction(
    home_team: str,
    away_team: str,
    elo_map: Dict[str, float],
    neutral: bool = True,
) -> dict:
    """
    Eleme maçı için Elo tabanlı tahmin.
    Beraberlik yoktur — bir kazanan belirlenir (uzatma/PSO dahil).
    """
    elo_h = _get_elo(home_team, elo_map)
    elo_a = _get_elo(away_team, elo_map)
    advantage = 0.0 if neutral else 60.0  # eleme maçları genellikle nötr saha
    ph = _elo_win_prob(elo_h, elo_a, advantage)
    pa = 1.0 - ph
    lh, la = _estimate_xg(elo_h, elo_a)

    return {
        "p_home":    round(ph, 4),
        "p_draw":    None,          # eleme turunda beraberlik yok
        "p_away":    round(pa, 4),
        "elo_home":  round(elo_h),
        "elo_away":  round(elo_a),
        "lambda_home": lh,
        "lambda_away": la,
        "source":    "elo",
    }


# ──────────────────────────────────────────────
# xG tahmini
# ──────────────────────────────────────────────

def _estimate_xg(elo_home: float, elo_away: float) -> Tuple[float, float]:
    """Elo farkından basit xG tahmini."""
    diff = (elo_home - elo_away) / 400.0
    factor = 0.5 + 0.5 * np.tanh(diff)
    total = 2.7
    xg_h = round(max(0.3, min(total * factor, 4.5)), 2)
    xg_a = round(max(0.3, min(total * (1.0 - factor), 4.5)), 2)
    return xg_h, xg_a


# ──────────────────────────────────────────────
# Poisson skor tablosu
# ──────────────────────────────────────────────

def poisson_score_table(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = 7,
) -> pd.DataFrame:
    """En olası 10 skoru Poisson ile döner."""
    rows = []
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            prob = scipy_poisson.pmf(h, lambda_home) * scipy_poisson.pmf(a, lambda_away)
            rows.append({"score": f"{h}-{a}", "home_g": h, "away_g": a, "prob": prob})
    df = pd.DataFrame(rows).sort_values("prob", ascending=False).head(10).reset_index(drop=True)
    df["prob_pct"] = (df["prob"] * 100).round(1).astype(str) + "%"
    return df[["score", "prob_pct", "prob"]]


# ──────────────────────────────────────────────
# Tüm turnuva takımları
# ──────────────────────────────────────────────

def get_tournament_teams() -> List[str]:
    """GROUP_FIXTURES.CSV'deki tüm unique takım isimlerini döner (Playoff dahil)."""
    path = FILES.get("group_fixtures", "")
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        teams = set(df["home_team"].dropna().astype(str)) | set(df["away_team"].dropna().astype(str))
        return sorted(teams)
    except Exception:
        return []
