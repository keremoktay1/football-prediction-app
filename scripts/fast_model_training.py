"""
fast_model_training.py

Notebook 03'ün eşdeğeri — doğrudan Python scripti.

Girdi:  data/processed/features_historical.csv
        data/processed/features_2026_fixtures.csv

Çıktı:  models/lr_model.pkl
        models/preprocessor.pkl
        models/home_goal_model.pkl
        models/away_goal_model.pkl
        models/poisson_imputer.pkl
        models/poisson_scaler.pkl
        data/processed/predictions_latest.csv
"""
from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# src path
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(APP_DIR, "src"))

from config import BASE_GOAL_RATE

PROCESSED_DIR = os.path.join(APP_DIR, "data", "processed")
MODEL_DIR     = os.path.join(APP_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, brier_score_loss
from scipy.stats import poisson
import joblib


# ── Veri yükleme ─────────────────────────────────────────────────────────────
print("Veriler yükleniyor...")

hist_path   = os.path.join(PROCESSED_DIR, "features_historical.csv")
future_path = os.path.join(PROCESSED_DIR, "features_2026_fixtures.csv")

if not os.path.isfile(hist_path):
    raise FileNotFoundError(
        f"features_historical.csv bulunamadı: {hist_path}\n"
        "Önce fast_feature_engineering.py çalıştırın."
    )

hist   = pd.read_csv(hist_path,   parse_dates=["date"])
future = pd.read_csv(future_path, parse_dates=["date_utc"])

print(f"  Tarihi veri  : {hist.shape}")
print(f"  2026 fikstür : {future.shape}")
print(f"\nHedef dağılımı:")
print(hist["result"].value_counts().to_string())


# ── Time-based split ──────────────────────────────────────────────────────────
print("\nTrain/Valid/Test split...")

train = hist[hist["date"] <  "2018-01-01"].copy()
valid = hist[(hist["date"] >= "2018-01-01") & (hist["date"] < "2022-01-01")].copy()
test  = hist[hist["date"] >= "2022-01-01"].copy()

LABEL_MAP = {"H": 0, "D": 1, "A": 2}
for df in [train, valid, test]:
    df["target"] = df["result"].map(LABEL_MAP)

print(f"  Train : {len(train):>6,} maç  ({train['date'].min().date()} → {train['date'].max().date()})")
print(f"  Valid : {len(valid):>6,} maç  ({valid['date'].min().date()} → {valid['date'].max().date()})")
print(f"  Test  : {len(test):>6,} maç  ({test['date'].min().date()} → {test['date'].max().date()})")


# ── Logistic Regression ───────────────────────────────────────────────────────
print("\n[1] Logistic Regression eğitiliyor...")

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
    "tournament_weight",
]

X_train = train[FEATURE_COLS].copy()
y_train = train["target"].values
X_valid = valid[FEATURE_COLS].copy()
y_valid = valid["target"].values
X_test  = test[FEATURE_COLS].copy()
y_test  = test["target"].values

preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

X_train_t = preprocessor.fit_transform(X_train)
X_valid_t = preprocessor.transform(X_valid)
X_test_t  = preprocessor.transform(X_test)

lr_model = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=500,
    C=1.0,
    class_weight="balanced",
    random_state=42,
)
lr_model.fit(X_train_t, y_train)

# Validation
y_valid_prob = lr_model.predict_proba(X_valid_t)
y_valid_pred = lr_model.predict(X_valid_t)
val_logloss  = log_loss(y_valid, y_valid_prob)
val_acc      = (y_valid_pred == y_valid).mean()

print(f"  Valid Log Loss : {val_logloss:.4f}")
print(f"  Valid Accuracy : {val_acc*100:.1f}%")

# Test
y_test_prob  = lr_model.predict_proba(X_test_t)
y_test_pred  = lr_model.predict(X_test_t)
test_logloss = log_loss(y_test, y_test_prob)
test_acc     = (y_test_pred == y_test).mean()
print(f"  Test  Log Loss : {test_logloss:.4f}")
print(f"  Test  Accuracy : {test_acc*100:.1f}%")


# ── Poisson xG modeli ─────────────────────────────────────────────────────────
print("\n[2] Poisson xG modeli eğitiliyor...")

# Home goal model — features: [elo_diff, attack_home, defense_away, neutral, tournament_weight]
HOME_POI_FEATS = ["elo_diff", "attack_home", "defense_away", "neutral", "tournament_weight"]

imputer_p = SimpleImputer(strategy="median")
scaler_p  = StandardScaler()

X_p_train_home = train[HOME_POI_FEATS].copy()
X_p_train_home = imputer_p.fit_transform(X_p_train_home)
X_p_train_home = scaler_p.fit_transform(X_p_train_home)

home_goal_model = Ridge(alpha=1.0)
home_goal_model.fit(X_p_train_home, train["home_score"].clip(0, 8))

X_p_valid_home = scaler_p.transform(imputer_p.transform(valid[HOME_POI_FEATS]))
pred_home_val  = home_goal_model.predict(X_p_valid_home).clip(0.3, 5)
mae_home       = float(np.mean(np.abs(pred_home_val - valid["home_score"])))

# Away goal model — swap attack/defense + negate elo_diff, use .values to bypass name check
AWAY_POI_FEATS = ["elo_diff", "attack_away", "defense_home", "neutral", "tournament_weight"]

train_away_vals = train[AWAY_POI_FEATS].copy()
train_away_vals["elo_diff"] = -train_away_vals["elo_diff"]
valid_away_vals = valid[AWAY_POI_FEATS].copy()
valid_away_vals["elo_diff"] = -valid_away_vals["elo_diff"]

X_p_train_away = imputer_p.transform(train_away_vals.values)
X_p_train_away = scaler_p.transform(X_p_train_away)
X_p_valid_away = scaler_p.transform(imputer_p.transform(valid_away_vals.values))

away_goal_model = Ridge(alpha=1.0)
away_goal_model.fit(X_p_train_away, train["away_score"].clip(0, 8))

pred_away_val = away_goal_model.predict(X_p_valid_away).clip(0.3, 5)
mae_away      = float(np.mean(np.abs(pred_away_val - valid["away_score"])))

print(f"  Home goals MAE : {mae_home:.3f}")
print(f"  Away goals MAE : {mae_away:.3f}")


# ── Poisson yardımcı fonksiyon ────────────────────────────────────────────────
MAX_GOALS = 8

def poisson_match_probs(lh: float, la: float) -> dict:
    lh = max(0.1, lh)
    la = max(0.1, la)
    goals = np.arange(0, MAX_GOALS + 1)
    p_home_goals = poisson.pmf(goals, lh)
    p_away_goals = poisson.pmf(goals, la)
    sm = np.outer(p_home_goals, p_away_goals)

    p_home = np.tril(sm, -1).sum()
    p_draw = np.trace(sm)
    p_away = np.triu(sm, 1).sum()
    total  = p_home + p_draw + p_away
    p_home /= total; p_draw /= total; p_away /= total

    over_2_5 = sum(sm[h, a] for h in goals for a in goals if h + a > 2.5)
    btts     = sum(sm[h, a] for h in goals for a in goals if h > 0 and a > 0)

    all_scores = [(int(h), int(a), float(sm[h, a])) for h in goals for a in goals]
    top5 = sorted(all_scores, key=lambda x: -x[2])[:5]

    return {
        "p_home":       round(p_home, 4),
        "p_draw":       round(p_draw, 4),
        "p_away":       round(p_away, 4),
        "lambda_home":  round(lh, 3),
        "lambda_away":  round(la, 3),
        "over_2_5":     round(over_2_5, 4),
        "btts":         round(btts, 4),
        "top_scorelines": str(top5),
    }


# ── 2026 tahminleri ───────────────────────────────────────────────────────────
print("\n[3] 2026 fikstür tahminleri üretiliyor...")

# --- LR olasılıkları ---
X_future = future[FEATURE_COLS].copy()
X_future_t = preprocessor.transform(X_future)
future_probs = lr_model.predict_proba(X_future_t)
future["p_home_lr"] = future_probs[:, 0]
future["p_draw_lr"] = future_probs[:, 1]
future["p_away_lr"] = future_probs[:, 2]

# --- Poisson lambdaları ---
poisson_rows = []
for _, row in future.iterrows():
    att_h = float(row.get("attack_home", 1.0) or 1.0)
    def_a = float(row.get("defense_away", 1.0) or 1.0)
    att_a = float(row.get("attack_away", 1.0) or 1.0)
    def_h = float(row.get("defense_home", 1.0) or 1.0)
    elo_d = float(row.get("elo_diff", 0) or 0)
    neutral = int(row.get("neutral", 1))

    home_ctx    = 1.0  # WC = neutral
    elo_factor  = 1 + np.clip(elo_d / 1000, -0.3, 0.3)

    lh = BASE_GOAL_RATE * att_h * def_a * home_ctx * elo_factor
    la = BASE_GOAL_RATE * att_a * def_h / elo_factor

    lh = max(0.3, min(lh, 5.0))
    la = max(0.3, min(la, 5.0))

    probs = poisson_match_probs(lh, la)
    poisson_rows.append({"match_id": row["match_id"], **probs})

poisson_df = pd.DataFrame(poisson_rows)
future = future.merge(poisson_df, on="match_id", how="left", suffixes=("", "_poi"))

# Rename Poisson columns
for col in ["p_home", "p_draw", "p_away"]:
    if f"{col}_poi" in future.columns:
        future[f"{col}_poi"] = future[f"{col}_poi"]
    elif col in future.columns:
        future[f"{col}_poi"] = future[col]

# --- Ensemble: 50% LR + 50% Poisson ---
W_LR = 0.5
W_POI = 0.5

future["p_home"] = W_LR * future["p_home_lr"] + W_POI * future["p_home_poi"]
future["p_draw"] = W_LR * future["p_draw_lr"] + W_POI * future["p_draw_poi"]
future["p_away"] = W_LR * future["p_away_lr"] + W_POI * future["p_away_poi"]

total = future["p_home"] + future["p_draw"] + future["p_away"]
future["p_home"] = (future["p_home"] / total).round(4)
future["p_draw"] = (future["p_draw"] / total).round(4)
future["p_away"] = (future["p_away"] / total).round(4)

def get_favourite(row):
    mx = max(row["p_home"], row["p_draw"], row["p_away"])
    if mx == row["p_home"]: return row["home_team"]
    if mx == row["p_draw"]:  return "Draw"
    return row["away_team"]

def upset_label_fn(score: float) -> str:
    if score >= 0.65: return "Yüksek"
    if score >= 0.45: return "Orta"
    return "Düşük"

future["favourite"]   = future.apply(get_favourite, axis=1)
if "upset_risk" not in future.columns:
    future["upset_risk"]  = 0.5
if "upset_label" not in future.columns:
    future["upset_label"] = future["upset_risk"].apply(upset_label_fn)

print(f"  Tahminler hazır: {len(future)} maç")

# Örnek
sample = future[["group", "home_team", "away_team", "p_home", "p_draw", "p_away",
                  "lambda_home", "lambda_away", "over_2_5", "favourite"]].head(6)
print(sample.to_string(index=False))


# ── Model kaydetme ────────────────────────────────────────────────────────────
print("\n[4] Modeller kaydediliyor...")

joblib.dump(lr_model,        os.path.join(MODEL_DIR, "lr_model.pkl"))
joblib.dump(preprocessor,    os.path.join(MODEL_DIR, "preprocessor.pkl"))
joblib.dump(home_goal_model, os.path.join(MODEL_DIR, "home_goal_model.pkl"))
joblib.dump(away_goal_model, os.path.join(MODEL_DIR, "away_goal_model.pkl"))
joblib.dump(imputer_p,       os.path.join(MODEL_DIR, "poisson_imputer.pkl"))
joblib.dump(scaler_p,        os.path.join(MODEL_DIR, "poisson_scaler.pkl"))

print("  Kaydedilen modeller:")
for f in sorted(os.listdir(MODEL_DIR)):
    fpath = os.path.join(MODEL_DIR, f)
    size  = os.path.getsize(fpath) / 1024
    print(f"    {f:<35} {size:.1f} KB")


# ── Tahminleri kaydetme ───────────────────────────────────────────────────────
print("\n[5] Tahminler kaydediliyor...")

pred_cols = [
    "match_id", "group", "date_utc", "venue",
    "home_team", "away_team",
    "elo_home", "elo_away", "elo_diff",
    "p_home", "p_draw", "p_away",
    "lambda_home", "lambda_away",
    "over_2_5", "btts",
    "top_scorelines", "favourite", "upset_risk", "upset_label",
]
avail_cols = [c for c in pred_cols if c in future.columns]
predictions = future[avail_cols].copy()

pred_path = os.path.join(PROCESSED_DIR, "predictions_latest.csv")
predictions.to_csv(pred_path, index=False)

print(f"  ✅  {pred_path}")
print(f"      {len(predictions)} maç, {len(predictions.columns)} kolon")


# ── Özet ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("ÖZET")
print("=" * 50)
print(f"LR  Log Loss (Test)   : {test_logloss:.4f}")
print(f"LR  Accuracy (Test)   : {test_acc*100:.1f}%")
print(f"Poi Home MAE (Valid)  : {mae_home:.3f}")
print(f"Poi Away MAE (Valid)  : {mae_away:.3f}")
print("\nİlk 5 tahmin:")
print(predictions[["group","home_team","away_team","p_home","p_draw","p_away","favourite"]].head(5).to_string(index=False))
print("\nTüm dosyalar hazır ✅")
