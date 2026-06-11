"""
update_tournament_predictions.py — Canlı turnuva tahmin güncelleyici.

Çalışma mantığı:
  1. match_updates.csv oku → oynanan maçları al
  2. GROUP_FIXTURES.CSV oku → maç-takım eşleşmesi
  3. Her takım için turnuva-içi istatistik hesapla (gf, ga, pts, matches)
  4. features_2026_fixtures.csv yükle
  5. Oynanmamış maçlar için attack/defense/form feature'larını blend ile güncelle
  6. Modelleri yükle, tahmin yap
  7. predictions_latest.csv'yi güncelle (oynanmış maçlara dokunma)

Blend formülü:
  blend = min(wc_matches / 3.0, 1.0)
  new_feature = old_feature * (1 - blend) + wc_feature * blend

Kullanım:
  python scripts/update_tournament_predictions.py
"""
from __future__ import annotations

import os
import sys
import json
import pickle
import joblib as joblib_mod
import warnings
import numpy as np
import pandas as pd
from typing import Dict

warnings.filterwarnings("ignore")

# ── Yol kurulumu ─────────────────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(APP_DIR, "src"))

from config import FILES, PROCESSED_DIR, BASE_GOAL_RATE

MODELS_DIR        = os.path.join(APP_DIR, "models")
FEATURES_PATH     = os.path.join(PROCESSED_DIR, "features_2026_fixtures.csv")
PREDICTIONS_PATH  = FILES["predictions"]
UPDATES_PATH      = os.path.join(PROCESSED_DIR, "match_updates.csv")
GROUP_FIXTURES_PATH = FILES["group_fixtures"]
ENSEMBLE_WEIGHTS_PATH = os.path.join(PROCESSED_DIR, "ensemble_weights.json")
ELO_CURRENT_PATH  = os.path.join(PROCESSED_DIR, "elo_current.csv")

# FIFA Dünya Kupası K-faktörü (standart aralık 20-40)
ELO_K_WC = 40

from prediction_engine import FEATURE_COLS, POISSON_FEATURES

# ── Model yükleme ─────────────────────────────────────────────────────────────

def _load_models() -> dict:
    models = {}
    for name, fname in [
        ("lr_model",        "lr_model.pkl"),
        ("preprocessor",    "preprocessor.pkl"),
        ("home_goal_model", "home_goal_model.pkl"),
        ("away_goal_model", "away_goal_model.pkl"),
        ("poisson_imputer", "poisson_imputer.pkl"),
        ("poisson_scaler",  "poisson_scaler.pkl"),
        ("rf_model",        "rf_model.pkl"),
        ("lgb_model",       "lgb_model.pkl"),
    ]:
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib_mod.load(path)
    return models


def _load_ensemble_weights() -> Dict[str, float]:
    if os.path.exists(ENSEMBLE_WEIGHTS_PATH):
        with open(ENSEMBLE_WEIGHTS_PATH) as f:
            return json.load(f)
    # Fallback: sadece LR
    return {"LR": 1.0, "Poisson": 0.0, "RF": 0.0, "LGB": 0.0}


# ── Turnuva-içi istatistik hesaplama ─────────────────────────────────────────

def _calc_wc_stats(updates: pd.DataFrame, fixtures: pd.DataFrame) -> Dict[str, dict]:
    """
    Her takım için turnuva içi istatistik hesaplar.
    Returns: {team: {wc_gf, wc_ga, wc_pts, wc_matches}}
    """
    stats: Dict[str, dict] = {}

    # fixtures'dan match_id → (home_team, away_team) haritası
    fix_map: Dict[int, tuple] = {}
    for _, row in fixtures.iterrows():
        fix_map[int(row["match_id"])] = (str(row["home_team"]), str(row["away_team"]))

    for _, upd in updates.iterrows():
        try:
            mid    = int(upd["match_id"])
            h_score = int(upd["home_score"])
            a_score = int(upd["away_score"])
        except (ValueError, TypeError):
            continue

        if mid not in fix_map:
            continue

        home_team, away_team = fix_map[mid]

        # Home takım
        if home_team not in stats:
            stats[home_team] = {"wc_gf": 0, "wc_ga": 0, "wc_pts": 0, "wc_matches": 0}
        stats[home_team]["wc_gf"]      += h_score
        stats[home_team]["wc_ga"]      += a_score
        stats[home_team]["wc_matches"] += 1
        if h_score > a_score:
            stats[home_team]["wc_pts"] += 3
        elif h_score == a_score:
            stats[home_team]["wc_pts"] += 1

        # Away takım
        if away_team not in stats:
            stats[away_team] = {"wc_gf": 0, "wc_ga": 0, "wc_pts": 0, "wc_matches": 0}
        stats[away_team]["wc_gf"]      += a_score
        stats[away_team]["wc_ga"]      += h_score
        stats[away_team]["wc_matches"] += 1
        if a_score > h_score:
            stats[away_team]["wc_pts"] += 3
        elif a_score == h_score:
            stats[away_team]["wc_pts"] += 1

    return stats


def _blend(old_val: float, wc_val: float, blend: float) -> float:
    return old_val * (1.0 - blend) + wc_val * blend


# ── WC sonuçlarına göre Elo güncelleme ────────────────────────────────────────

def _update_elo_from_wc(
    updates: pd.DataFrame,
    fixtures: pd.DataFrame,
) -> Dict[str, float]:
    """
    Oynanan WC maçlarını kronolojik sırayla işleyerek Elo'ları günceller.
    Her maçta standart Elo formülü uygulanır (K=40).

    Returns: {team: updated_elo} — tüm takımlar dahil, sadece oynayanlar değişir.
    """
    elo_path = FILES.get("elo", "")
    if not elo_path or not os.path.exists(elo_path):
        print("[WARN] Elo dosyası bulunamadı — Elo güncellemesi atlandı.")
        return {}

    elo_df = pd.read_csv(elo_path)
    if "snapshot_date" in elo_df.columns:
        elo_df["snapshot_date"] = pd.to_datetime(elo_df["snapshot_date"], errors="coerce")
        latest = elo_df["snapshot_date"].max()
        elo_df = elo_df[elo_df["snapshot_date"] == latest]

    elo_map: Dict[str, float] = {}
    if "country" in elo_df.columns and "rating" in elo_df.columns:
        for _, r in elo_df.iterrows():
            elo_map[str(r["country"])] = float(r["rating"])

    if not elo_map:
        return {}

    # Maç tarih ve takım haritaları
    date_map: Dict[int, str] = {}
    fix_map: Dict[int, tuple] = {}
    for _, row in fixtures.iterrows():
        mid = int(row["match_id"])
        date_map[mid] = str(row.get("date_utc", ""))
        fix_map[mid] = (str(row["home_team"]), str(row["away_team"]))

    # Oynanan maçları kronolojik sırala (tarihe göre)
    played: list = []
    for _, upd in updates.iterrows():
        try:
            mid = int(upd["match_id"])
            hs  = int(upd["home_score"])
            as_ = int(upd["away_score"])
            played.append((date_map.get(mid, ""), mid, hs, as_))
        except (ValueError, TypeError):
            continue
    played.sort(key=lambda x: x[0])  # ISO tarih → kronolojik

    # Elo güncelleme döngüsü
    for _, mid, hs, as_ in played:
        if mid not in fix_map:
            continue
        ht, at = fix_map[mid]
        elo_h = elo_map.get(ht, 1700)
        elo_a = elo_map.get(at, 1700)

        # Beklenen kazanma olasılığı (home perspektifinden)
        expected_h = 1.0 / (1.0 + 10.0 ** (-(elo_h - elo_a) / 400.0))

        # Gerçek sonuç: galibiyet=1, beraberlik=0.5, mağlubiyet=0
        if hs > as_:
            score_h, score_a = 1.0, 0.0
        elif hs == as_:
            score_h, score_a = 0.5, 0.5
        else:
            score_h, score_a = 0.0, 1.0

        elo_map[ht] = elo_h + ELO_K_WC * (score_h - expected_h)
        elo_map[at] = elo_a + ELO_K_WC * (score_a - (1.0 - expected_h))

    return elo_map


# ── Feature güncelleme ────────────────────────────────────────────────────────

def _update_features(
    features: pd.DataFrame,
    played_ids: set,
    wc_stats: Dict[str, dict],
    updated_elo: Dict[str, float],
) -> pd.DataFrame:
    """
    Oynanmamış maçlar için feature'ları WC verisiyle günceller.
    Oynanmış maçlara dokunmaz.

    Güncellenenler:
      - attack_home/away, defense_home/away  (WC gol istatistikleri)
      - weighted_form_home/away, form_diff   (WC puan formu)
      - elo_home, elo_away, elo_diff         (WC sonuçlarına göre K=40 Elo)
      - elo_form_interaction                 (elo_diff × weighted_form_home)
    """
    features = features.copy()

    for idx, row in features.iterrows():
        mid = int(row["match_id"])
        if mid in played_ids:
            continue  # Oynanmış maç — dokunma

        home_team = str(row["home_team"])
        away_team = str(row["away_team"])

        # ── Saldırı / Savunma / Form güncellemesi ────────────────────────────
        for side, team in [("home", home_team), ("away", away_team)]:
            if team not in wc_stats:
                continue

            s = wc_stats[team]
            wc_matches = s["wc_matches"]
            if wc_matches == 0:
                continue

            blend = min(wc_matches / 3.0, 1.0)

            wc_attack = (s["wc_gf"] / wc_matches) / BASE_GOAL_RATE
            old_attack = float(row.get(f"attack_{side}", 1.0))
            features.at[idx, f"attack_{side}"] = _blend(old_attack, wc_attack, blend)

            wc_defense = (s["wc_ga"] / wc_matches) / BASE_GOAL_RATE
            old_defense = float(row.get(f"defense_{side}", 1.0))
            features.at[idx, f"defense_{side}"] = _blend(old_defense, wc_defense, blend)

            wc_form = s["wc_pts"] / wc_matches  # 0-3
            old_form = float(row.get(f"weighted_form_{side}", 1.0))
            features.at[idx, f"weighted_form_{side}"] = _blend(old_form, wc_form, blend)

        # form_diff yeniden hesapla
        new_form_home = float(features.at[idx, "weighted_form_home"])
        new_form_away = float(features.at[idx, "weighted_form_away"])
        features.at[idx, "form_diff"] = new_form_home - new_form_away

        # ── Elo güncellemesi ──────────────────────────────────────────────────
        elo_h = updated_elo.get(home_team)
        elo_a = updated_elo.get(away_team)

        if elo_h is not None and "elo_home" in features.columns:
            features.at[idx, "elo_home"] = round(elo_h, 1)
        if elo_a is not None and "elo_away" in features.columns:
            features.at[idx, "elo_away"] = round(elo_a, 1)
        if elo_h is not None and elo_a is not None and "elo_diff" in features.columns:
            new_diff = elo_h - elo_a
            features.at[idx, "elo_diff"] = round(new_diff, 1)
            # elo_form_interaction = (elo_diff / 400) × (form_home - form_away)
            if "elo_form_interaction" in features.columns:
                features.at[idx, "elo_form_interaction"] = round(
                    (new_diff / 400.0) * (new_form_home - new_form_away), 4
                )

    return features


# ── Tahmin üretme ─────────────────────────────────────────────────────────────

def _predict_row(row: pd.Series, models: dict, weights: Dict[str, float]) -> dict:
    """
    Bir maç için ensemble tahmin üretir.
    Returns: {p_home, p_draw, p_away, lambda_home, lambda_away}
    """
    lr_model     = models.get("lr_model")
    preprocessor = models.get("preprocessor")
    rf_model     = models.get("rf_model")
    lgb_model    = models.get("lgb_model")
    home_goal    = models.get("home_goal_model")
    away_goal    = models.get("away_goal_model")
    poi_imp      = models.get("poisson_imputer")
    poi_scl      = models.get("poisson_scaler")

    feat_dict = {c: row.get(c, np.nan) for c in FEATURE_COLS}
    X = pd.DataFrame([feat_dict])

    probas = {}  # model_name → [p_home, p_draw, p_away]

    # LR
    if lr_model and preprocessor:
        try:
            X_t = preprocessor.transform(X)
            p = lr_model.predict_proba(X_t)[0]
            classes = list(lr_model.classes_)
            ph = pd_ = pa = 1 / 3
            for i, cls in enumerate(classes):
                if cls == 0:
                    ph = float(p[i])
                elif cls == 1:
                    pd_ = float(p[i])
                elif cls == 2:
                    pa = float(p[i])
            probas["LR"] = [ph, pd_, pa]
        except Exception:
            pass

    # RF
    if rf_model and preprocessor:
        try:
            X_t = preprocessor.transform(X)
            p = rf_model.predict_proba(X_t)[0]
            classes = list(rf_model.classes_)
            ph = pd_ = pa = 1 / 3
            for i, cls in enumerate(classes):
                if cls == 0:
                    ph = float(p[i])
                elif cls == 1:
                    pd_ = float(p[i])
                elif cls == 2:
                    pa = float(p[i])
            probas["RF"] = [ph, pd_, pa]
        except Exception:
            pass

    # LGB
    if lgb_model and preprocessor:
        try:
            X_t = preprocessor.transform(X)
            p = lgb_model.predict_proba(X_t)[0]
            classes = list(lgb_model.classes_)
            ph = pd_ = pa = 1 / 3
            for i, cls in enumerate(classes):
                if cls == 0:
                    ph = float(p[i])
                elif cls == 1:
                    pd_ = float(p[i])
                elif cls == 2:
                    pa = float(p[i])
            probas["LGB"] = [ph, pd_, pa]
        except Exception:
            pass

    # Poisson lambda tahmini
    lh = la = None
    if home_goal and away_goal and poi_imp and poi_scl:
        try:
            X_poi = pd.DataFrame([{c: row.get(c, np.nan) for c in POISSON_FEATURES}])
            X_poi_imp = poi_imp.transform(X_poi)
            X_poi_scl = poi_scl.transform(X_poi_imp)
            lh = max(0.3, float(home_goal.predict(X_poi_scl)[0]))
            la = max(0.3, float(away_goal.predict(X_poi_scl)[0]))
        except Exception:
            pass

    if not probas:
        return {"p_home": 1/3, "p_draw": 1/3, "p_away": 1/3, "lambda_home": lh, "lambda_away": la}

    # Ağırlıklı ensemble
    total_w = sum(weights.get(m, 0) for m in probas)
    if total_w <= 0:
        total_w = len(probas)
        avg = [sum(v[i] for v in probas.values()) / total_w for i in range(3)]
    else:
        avg = [0.0, 0.0, 0.0]
        for m, p in probas.items():
            w = weights.get(m, 0) / total_w
            for i in range(3):
                avg[i] += w * p[i]

    # Normalize (toplam 1 olsun)
    s = sum(avg)
    if s > 0:
        avg = [v / s for v in avg]

    return {
        "p_home": round(avg[0], 4),
        "p_draw": round(avg[1], 4),
        "p_away": round(avg[2], 4),
        "lambda_home": round(lh, 4) if lh else None,
        "lambda_away": round(la, 4) if la else None,
    }


# ── Ana akış ─────────────────────────────────────────────────────────────────

def main():
    # 1. Oynanan maçları yükle
    if not os.path.exists(UPDATES_PATH):
        print("[INFO] match_updates.csv bulunamadı — güncelleme yapılmadı.")
        return

    updates = pd.read_csv(UPDATES_PATH)
    updates = updates.dropna(subset=["match_id", "home_score", "away_score"])
    if updates.empty:
        print("[INFO] Hiç oynanmış maç yok — güncelleme gerekmiyor.")
        return

    played_ids = set(updates["match_id"].astype(int))
    print(f"[OK]  {len(played_ids)} oynanmış maç bulundu.")

    # 2. GROUP_FIXTURES.CSV yükle
    if not os.path.exists(GROUP_FIXTURES_PATH):
        print(f"[ERR] GROUP_FIXTURES.CSV bulunamadı: {GROUP_FIXTURES_PATH}")
        sys.exit(1)

    fixtures = pd.read_csv(GROUP_FIXTURES_PATH)
    print(f"[OK]  {len(fixtures)} grup fikstür yüklendi.")

    # 3. WC istatistikleri hesapla
    wc_stats = _calc_wc_stats(updates, fixtures)
    print(f"[OK]  {len(wc_stats)} takım için WC istatistikleri hesaplandı.")

    # 3b. Elo'ları WC sonuçlarına göre güncelle
    updated_elo = _update_elo_from_wc(updates, fixtures)
    if updated_elo:
        elo_rows = [{"team": t, "elo": round(v, 1)} for t, v in sorted(updated_elo.items())]
        pd.DataFrame(elo_rows).to_csv(ELO_CURRENT_PATH, index=False)
        print(f"[OK]  Güncel Elo {ELO_CURRENT_PATH} kaydedildi ({len(updated_elo)} takım).")
        # Değişimleri göster
        changed = [(t, v) for t, v in updated_elo.items()
                   if t in wc_stats]  # sadece WC'de oynayan takımlar
        for team, new_elo in sorted(changed, key=lambda x: -abs(x[1])):
            print(f"       {team}: {new_elo:.1f}")
    else:
        print("[WARN] Elo güncellemesi yapılamadı.")

    # 4. features_2026_fixtures.csv yükle
    if not os.path.exists(FEATURES_PATH):
        print(f"[ERR] features_2026_fixtures.csv bulunamadı: {FEATURES_PATH}")
        sys.exit(1)

    features = pd.read_csv(FEATURES_PATH)
    print(f"[OK]  {len(features)} maç için feature'lar yüklendi.")

    # 5. Feature'ları güncelle (sadece oynanmamış maçlar)
    features_updated = _update_features(features, played_ids, wc_stats, updated_elo)
    unplayed_count = len(features_updated[~features_updated["match_id"].isin(played_ids)])
    print(f"[OK]  {unplayed_count} oynanmamış maç için feature'lar güncellendi.")

    # 6. Modelleri yükle
    models = _load_models()
    weights = _load_ensemble_weights()
    print(f"[OK]  {len(models)} model yüklendi. Ensemble ağırlıkları: {weights}")

    if not models.get("lr_model"):
        print("[ERR] LR modeli yüklenemedi — tahmin yapılamıyor.")
        sys.exit(1)

    # 7. Mevcut predictions_latest.csv yükle
    if not os.path.exists(PREDICTIONS_PATH):
        print(f"[ERR] predictions_latest.csv bulunamadı: {PREDICTIONS_PATH}")
        sys.exit(1)

    predictions = pd.read_csv(PREDICTIONS_PATH)
    print(f"[OK]  {len(predictions)} mevcut tahmin yüklendi.")

    # 8. Oynanmamış maçlar için yeni tahmin üret
    updated_count = 0
    for idx, feat_row in features_updated.iterrows():
        mid = int(feat_row["match_id"])
        if mid in played_ids:
            continue  # Oynanmış maça dokunma

        new_pred = _predict_row(feat_row, models, weights)

        pred_idx = predictions.index[predictions["match_id"] == mid]
        if len(pred_idx) == 0:
            continue

        i = pred_idx[0]
        predictions.at[i, "p_home"]       = new_pred["p_home"]
        predictions.at[i, "p_draw"]       = new_pred["p_draw"]
        predictions.at[i, "p_away"]       = new_pred["p_away"]
        if new_pred["lambda_home"] is not None:
            predictions.at[i, "lambda_home"] = new_pred["lambda_home"]
        if new_pred["lambda_away"] is not None:
            predictions.at[i, "lambda_away"] = new_pred["lambda_away"]
        updated_count += 1

    # 9. Kaydet
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    print(f"[OK]  {updated_count} maç tahmini güncellendi → {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
