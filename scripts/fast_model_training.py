"""
fast_model_training.py

Notebook 03 eşdeğeri + çoklu model karşılaştırması.

Girdi:  data/processed/features_historical.csv
        data/processed/features_2026_fixtures.csv

Çıktı:  models/lr_model.pkl
        models/preprocessor.pkl
        models/home_goal_model.pkl
        models/away_goal_model.pkl
        models/poisson_imputer.pkl
        models/poisson_scaler.pkl
        models/rf_model.pkl
        models/xgb_model.pkl
        data/processed/predictions_latest.csv
        data/processed/model_comparison.csv
"""
from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(APP_DIR, "src"))

from config import BASE_GOAL_RATE

PROCESSED_DIR = os.path.join(APP_DIR, "data", "processed")
MODEL_DIR     = os.path.join(APP_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import poisson as scipy_poisson
import joblib

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[INFO] xgboost bulunamadı, atlanıyor.")


# ── Sabitler ──────────────────────────────────────────────────────────────────
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
]
POISSON_HOME_FEATS = ["elo_diff", "attack_home", "defense_away", "neutral", "tournament_weight"]
POISSON_AWAY_FEATS = ["elo_diff", "attack_away", "defense_home", "neutral", "tournament_weight"]
LABEL_MAP = {"H": 0, "D": 1, "A": 2}
MAX_GOALS = 8


# ── Yardımcı: Poisson skor matrisi ───────────────────────────────────────────
def poisson_match_probs(lh: float, la: float) -> dict:
    lh = max(0.1, lh)
    la = max(0.1, la)
    goals = np.arange(0, MAX_GOALS + 1)
    sm = np.outer(scipy_poisson.pmf(goals, lh), scipy_poisson.pmf(goals, la))

    p_home = float(np.tril(sm, -1).sum())
    p_draw = float(np.trace(sm))
    p_away = float(np.triu(sm, 1).sum())
    total  = p_home + p_draw + p_away

    over_2_5 = float(sum(sm[h, a] for h in goals for a in goals if h + a > 2.5))
    btts     = float(sum(sm[h, a] for h in goals for a in goals if h > 0 and a > 0))
    top5 = sorted(
        [(int(h), int(a), float(sm[h, a])) for h in goals for a in goals],
        key=lambda x: -x[2]
    )[:5]

    return {
        "p_home":        round(p_home / total, 4),
        "p_draw":        round(p_draw / total, 4),
        "p_away":        round(p_away / total, 4),
        "lambda_home":   round(lh, 3),
        "lambda_away":   round(la, 3),
        "over_2_5":      round(over_2_5, 4),
        "btts":          round(btts, 4),
        "top_scorelines": str(top5),
    }


# ── Yardımcı: metrik hesapla ─────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    split_name: str, model_name: str) -> dict:
    """y_prob: (N, 3) array — sütun sırası H=0, D=1, A=2"""
    y_pred = y_prob.argmax(axis=1)
    acc    = float((y_pred == y_true).mean())
    ll     = float(log_loss(y_true, y_prob, labels=[0, 1, 2]))

    brier = 0.0
    for i in range(3):
        brier += float(brier_score_loss((y_true == i).astype(int), y_prob[:, i]))
    brier /= 3

    # Mean Calibration Error: ortalama |pred_p - actual_freq|
    bins = np.linspace(0, 1, 11)
    cal_errors = []
    for i in range(3):
        for b in range(len(bins) - 1):
            mask = (y_prob[:, i] >= bins[b]) & (y_prob[:, i] < bins[b + 1])
            if mask.sum() > 0:
                pred_mean = y_prob[:, i][mask].mean()
                actual    = (y_true[mask] == i).mean()
                cal_errors.append(abs(pred_mean - actual))
    mce = float(np.mean(cal_errors)) if cal_errors else 0.0

    return {
        "model":    model_name,
        "split":    split_name,
        "log_loss": round(ll, 4),
        "brier":    round(brier, 4),
        "accuracy": round(acc, 4),
        "mce":      round(mce, 4),
        "n":        int(len(y_true)),
    }


# ── Elo-only baseline olasılığı ───────────────────────────────────────────────
def elo_probs(elo_diff: float, neutral: int = 1) -> np.ndarray:
    """Davidson modeli: ~22% beraberlik oranı sabit."""
    advantage = 0.0 if neutral else 100.0
    we = 1.0 / (1.0 + 10.0 ** (-(elo_diff + advantage) / 400.0))
    draw_frac = 0.22
    ph = we * (1.0 - draw_frac)
    pa = (1.0 - we) * (1.0 - draw_frac)
    pd_ = draw_frac
    total = ph + pd_ + pa
    return np.array([ph / total, pd_ / total, pa / total])


def elo_prob_matrix(X: pd.DataFrame) -> np.ndarray:
    """X'deki elo_diff ve neutral sütunlarından (N,3) prob matrisi üretir."""
    elo_col     = X["elo_diff"].fillna(0).values
    neutral_col = X["neutral"].fillna(1).values.astype(int)
    return np.vstack([elo_probs(ed, n) for ed, n in zip(elo_col, neutral_col)])


# ── Veri yükleme ─────────────────────────────────────────────────────────────
print("=" * 55)
print("ÇOKLU MODEL EĞİTİMİ VE KARŞILAŞTIRMASI")
print("=" * 55)
print("\nVeriler yükleniyor...")

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


# ── Time-based split ──────────────────────────────────────────────────────────
train = hist[hist["date"] <  "2018-01-01"].copy()
valid = hist[(hist["date"] >= "2018-01-01") & (hist["date"] < "2022-01-01")].copy()
test  = hist[hist["date"] >= "2022-01-01"].copy()

for df in [train, valid, test]:
    df["target"] = df["result"].map(LABEL_MAP)

print(f"\n  Train : {len(train):>6,}  (2000 → 2017)")
print(f"  Valid : {len(valid):>6,}  (2018 → 2021)")
print(f"  Test  : {len(test):>6,}  (2022 → bugün)")

# ── 2026 WC match_updates — yüksek ağırlıklı yeniden eğitim ──────────────────
train_weights = train["tournament_weight"].fillna(1.0).values.copy().astype(float)

updates_path = os.path.join(PROCESSED_DIR, "match_updates.csv")
if os.path.isfile(updates_path):
    _updates_df = pd.read_csv(updates_path)
    if not _updates_df.empty:
        _future_feats = future[["match_id"] + FEATURE_COLS].copy()
        _wc2026 = _updates_df.merge(_future_feats, on="match_id", how="inner")
        if not _wc2026.empty:
            _wc2026 = _wc2026.copy()
            _wc2026["home_score"] = _wc2026["home_score"].astype(int)
            _wc2026["away_score"] = _wc2026["away_score"].astype(int)

            def _derive_result(r):
                if r["home_score"] > r["away_score"]: return "H"
                elif r["home_score"] == r["away_score"]: return "D"
                else: return "A"

            _wc2026["result"] = _wc2026.apply(_derive_result, axis=1)
            _wc2026["target"] = _wc2026["result"].map(LABEL_MAP)
            _wc2026["date"]   = pd.Timestamp("2026-06-01")

            train = pd.concat([train, _wc2026], ignore_index=True)
            train_weights = np.concatenate([train_weights, np.full(len(_wc2026), 20.0)])
            print(f"\n  ✅  {len(_wc2026)} adet 2026 WC maçı eğitime eklendi (ağırlık=20.0)")
        else:
            print("\n  ℹ️  match_updates.csv var ama özellik eşleşmesi bulunamadı.")
    else:
        print("\n  ℹ️  match_updates.csv boş — standart eğitim.")
else:
    print("\n  ℹ️  match_updates.csv bulunamadı — standart eğitim.")

X_train = train[FEATURE_COLS].copy()
y_train = train["target"].values
X_valid = valid[FEATURE_COLS].copy()
y_valid = valid["target"].values
X_test  = test[FEATURE_COLS].copy()
y_test  = test["target"].values


# ── Preprocessing pipeline ────────────────────────────────────────────────────
preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])
X_train_t = preprocessor.fit_transform(X_train)
X_valid_t = preprocessor.transform(X_valid)
X_test_t  = preprocessor.transform(X_test)


# ── [1] Elo-only baseline ─────────────────────────────────────────────────────
print("\n[1] Elo-only baseline...")
elo_valid_prob = elo_prob_matrix(X_valid)
elo_test_prob  = elo_prob_matrix(X_test)
print(f"  Valid LL: {log_loss(y_valid, elo_valid_prob, labels=[0,1,2]):.4f}")
print(f"  Test  LL: {log_loss(y_test,  elo_test_prob,  labels=[0,1,2]):.4f}")


# ── [2] Logistic Regression ───────────────────────────────────────────────────
print("\n[2] Logistic Regression eğitiliyor...")
lr_model = LogisticRegression(
    multi_class="multinomial", solver="lbfgs",
    max_iter=500, C=1.0, class_weight="balanced", random_state=42,
)
lr_model.fit(X_train_t, y_train, sample_weight=train_weights)
lr_valid_prob = lr_model.predict_proba(X_valid_t)
lr_test_prob  = lr_model.predict_proba(X_test_t)
print(f"  Valid LL: {log_loss(y_valid, lr_valid_prob, labels=[0,1,2]):.4f}  "
      f"Acc: {(lr_model.predict(X_valid_t)==y_valid).mean()*100:.1f}%")
print(f"  Test  LL: {log_loss(y_test, lr_test_prob, labels=[0,1,2]):.4f}  "
      f"Acc: {(lr_model.predict(X_test_t)==y_test).mean()*100:.1f}%")


# ── [3] Poisson xG modeli ─────────────────────────────────────────────────────
print("\n[3] Poisson xG modeli eğitiliyor...")
imputer_p = SimpleImputer(strategy="median")
scaler_p  = StandardScaler()

X_p_train_home = scaler_p.fit_transform(
    imputer_p.fit_transform(train[POISSON_HOME_FEATS])
)
X_p_valid_home = scaler_p.transform(imputer_p.transform(valid[POISSON_HOME_FEATS]))
X_p_test_home  = scaler_p.transform(imputer_p.transform(test[POISSON_HOME_FEATS]))

train_away = train[POISSON_AWAY_FEATS].copy(); train_away["elo_diff"] = -train_away["elo_diff"]
valid_away = valid[POISSON_AWAY_FEATS].copy(); valid_away["elo_diff"] = -valid_away["elo_diff"]
test_away  = test[POISSON_AWAY_FEATS].copy();  test_away["elo_diff"]  = -test_away["elo_diff"]

X_p_train_away = scaler_p.transform(imputer_p.transform(train_away.values))
X_p_valid_away = scaler_p.transform(imputer_p.transform(valid_away.values))
X_p_test_away  = scaler_p.transform(imputer_p.transform(test_away.values))

home_goal_model = Ridge(alpha=1.0)
home_goal_model.fit(X_p_train_home, train["home_score"].clip(0, 8), sample_weight=train_weights)
away_goal_model = Ridge(alpha=1.0)
away_goal_model.fit(X_p_train_away, train["away_score"].clip(0, 8), sample_weight=train_weights)

def poisson_prob_matrix(lh_arr: np.ndarray, la_arr: np.ndarray) -> np.ndarray:
    """Toplu Poisson olasılık matrisi — (N, 3)."""
    goals = np.arange(0, MAX_GOALS + 1)
    out = []
    for lh, la in zip(lh_arr, la_arr):
        r = poisson_match_probs(max(0.3, lh), max(0.3, la))
        out.append([r["p_home"], r["p_draw"], r["p_away"]])
    return np.array(out)

lh_valid = home_goal_model.predict(X_p_valid_home).clip(0.3, 5)
la_valid = away_goal_model.predict(X_p_valid_away).clip(0.3, 5)
lh_test  = home_goal_model.predict(X_p_test_home).clip(0.3, 5)
la_test  = away_goal_model.predict(X_p_test_away).clip(0.3, 5)

poi_valid_prob = poisson_prob_matrix(lh_valid, la_valid)
poi_test_prob  = poisson_prob_matrix(lh_test,  la_test)

mae_home = float(np.mean(np.abs(home_goal_model.predict(X_p_valid_home) - valid["home_score"])))
mae_away = float(np.mean(np.abs(away_goal_model.predict(X_p_valid_away) - valid["away_score"])))
print(f"  Home MAE: {mae_home:.3f}  Away MAE: {mae_away:.3f}")
print(f"  Valid LL: {log_loss(y_valid, poi_valid_prob, labels=[0,1,2]):.4f}")
print(f"  Test  LL: {log_loss(y_test,  poi_test_prob,  labels=[0,1,2]):.4f}")


# ── [4] LR + Poisson Ensemble ─────────────────────────────────────────────────
print("\n[4] Ensemble (LR 50% + Poisson 50%)...")
ens_valid_prob = (lr_valid_prob + poi_valid_prob) / 2
ens_test_prob  = (lr_test_prob  + poi_test_prob)  / 2
# normalize rows
ens_valid_prob /= ens_valid_prob.sum(axis=1, keepdims=True)
ens_test_prob  /= ens_test_prob.sum(axis=1, keepdims=True)
print(f"  Valid LL: {log_loss(y_valid, ens_valid_prob, labels=[0,1,2]):.4f}  "
      f"Acc: {(ens_valid_prob.argmax(1)==y_valid).mean()*100:.1f}%")
print(f"  Test  LL: {log_loss(y_test,  ens_test_prob,  labels=[0,1,2]):.4f}  "
      f"Acc: {(ens_test_prob.argmax(1)==y_test).mean()*100:.1f}%")


# ── [5] Random Forest ────────────────────────────────────────────────────────
print("\n[5] Random Forest eğitiliyor...")
rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=20,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
rf_model.fit(X_train_t, y_train, sample_weight=train_weights)
rf_valid_prob = rf_model.predict_proba(X_valid_t)
rf_test_prob  = rf_model.predict_proba(X_test_t)
print(f"  Valid LL: {log_loss(y_valid, rf_valid_prob, labels=[0,1,2]):.4f}  "
      f"Acc: {(rf_model.predict(X_valid_t)==y_valid).mean()*100:.1f}%")
print(f"  Test  LL: {log_loss(y_test,  rf_test_prob,  labels=[0,1,2]):.4f}  "
      f"Acc: {(rf_model.predict(X_test_t)==y_test).mean()*100:.1f}%")


# ── [6] XGBoost ───────────────────────────────────────────────────────────────
if HAS_XGB:
    print("\n[6] XGBoost eğitiliyor...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", use_label_encoder=False,
        random_state=42, n_jobs=-1,
    )
    xgb_model.fit(
        X_train_t, y_train,
        sample_weight=train_weights,
        eval_set=[(X_valid_t, y_valid)],
        verbose=False,
    )
    xgb_valid_prob = xgb_model.predict_proba(X_valid_t)
    xgb_test_prob  = xgb_model.predict_proba(X_test_t)
    print(f"  Valid LL: {log_loss(y_valid, xgb_valid_prob, labels=[0,1,2]):.4f}  "
          f"Acc: {(xgb_model.predict(X_valid_t)==y_valid).mean()*100:.1f}%")
    print(f"  Test  LL: {log_loss(y_test,  xgb_test_prob,  labels=[0,1,2]):.4f}  "
          f"Acc: {(xgb_model.predict(X_test_t)==y_test).mean()*100:.1f}%")
else:
    xgb_valid_prob = lr_valid_prob.copy()
    xgb_test_prob  = lr_test_prob.copy()


# ── Model karşılaştırma tablosu ───────────────────────────────────────────────
print("\n" + "=" * 55)
print("MODEL KARŞILAŞTIRMA TABLOSU")
print("=" * 55)

comparison_rows = []

model_probs = {
    "Elo Baseline":    (elo_valid_prob,  elo_test_prob),
    "Poisson":         (poi_valid_prob,  poi_test_prob),
    "LR":              (lr_valid_prob,   lr_test_prob),
    "Ensemble":        (ens_valid_prob,  ens_test_prob),
    "Random Forest":   (rf_valid_prob,   rf_test_prob),
}
if HAS_XGB:
    model_probs["XGBoost"] = (xgb_valid_prob, xgb_test_prob)

for mname, (vp, tp) in model_probs.items():
    vm = compute_metrics(y_valid, vp, "valid", mname)
    tm = compute_metrics(y_test,  tp, "test",  mname)
    comparison_rows.extend([vm, tm])
    print(f"  {mname:<16}  Valid LL={vm['log_loss']:.4f}  Test LL={tm['log_loss']:.4f}  "
          f"Test Acc={tm['accuracy']*100:.1f}%  Test Brier={tm['brier']:.4f}")

comp_df = pd.DataFrame(comparison_rows)
comp_path = os.path.join(PROCESSED_DIR, "model_comparison.csv")
comp_df.to_csv(comp_path, index=False)
print(f"\n  ✅  model_comparison.csv → {comp_path}")


# ── 2026 Fikstür tahminleri ───────────────────────────────────────────────────
print("\n[7] 2026 fikstür tahminleri...")

X_future   = future[FEATURE_COLS].copy()
X_future_t = preprocessor.transform(X_future)

future["p_home_lr"] = lr_model.predict_proba(X_future_t)[:, 0]
future["p_draw_lr"] = lr_model.predict_proba(X_future_t)[:, 1]
future["p_away_lr"] = lr_model.predict_proba(X_future_t)[:, 2]

# Poisson lambdaları
poisson_rows = []
for _, row in future.iterrows():
    att_h = float(row.get("attack_home",  1.0) or 1.0)
    def_a = float(row.get("defense_away", 1.0) or 1.0)
    att_a = float(row.get("attack_away",  1.0) or 1.0)
    def_h = float(row.get("defense_home", 1.0) or 1.0)
    elo_d = float(row.get("elo_diff", 0) or 0)

    elo_factor = 1 + np.clip(elo_d / 1000, -0.3, 0.3)
    lh = max(0.3, min(BASE_GOAL_RATE * att_h * def_a * elo_factor, 5.0))
    la = max(0.3, min(BASE_GOAL_RATE * att_a * def_h / elo_factor, 5.0))
    probs = poisson_match_probs(lh, la)
    poisson_rows.append({"match_id": row["match_id"], **probs})

poisson_df = pd.DataFrame(poisson_rows).rename(columns={
    "p_home": "p_home_poi", "p_draw": "p_draw_poi", "p_away": "p_away_poi",
    "lambda_home": "lambda_home", "lambda_away": "lambda_away",
    "over_2_5": "over_2_5", "btts": "btts", "top_scorelines": "top_scorelines",
})
future = future.merge(poisson_df, on="match_id", how="left")

# Ensemble
future["p_home"] = (future["p_home_lr"] + future["p_home_poi"]) / 2
future["p_draw"] = (future["p_draw_lr"] + future["p_draw_poi"]) / 2
future["p_away"] = (future["p_away_lr"] + future["p_away_poi"]) / 2
total = future["p_home"] + future["p_draw"] + future["p_away"]
future["p_home"] = (future["p_home"] / total).round(4)
future["p_draw"] = (future["p_draw"] / total).round(4)
future["p_away"] = (future["p_away"] / total).round(4)

def get_favourite(row):
    mx = max(row["p_home"], row["p_draw"], row["p_away"])
    if mx == row["p_home"]: return row["home_team"]
    if mx == row["p_draw"]:  return "Draw"
    return row["away_team"]

def upset_label_fn(s: float) -> str:
    if s >= 0.65: return "Yüksek"
    if s >= 0.45: return "Orta"
    return "Düşük"

future["favourite"] = future.apply(get_favourite, axis=1)
if "upset_risk" not in future.columns:
    future["upset_risk"] = 0.5
if "upset_label" not in future.columns:
    future["upset_label"] = future["upset_risk"].apply(upset_label_fn)


# ── Model kaydetme ────────────────────────────────────────────────────────────
print("\n[8] Modeller kaydediliyor...")
joblib.dump(lr_model,        os.path.join(MODEL_DIR, "lr_model.pkl"))
joblib.dump(preprocessor,    os.path.join(MODEL_DIR, "preprocessor.pkl"))
joblib.dump(home_goal_model, os.path.join(MODEL_DIR, "home_goal_model.pkl"))
joblib.dump(away_goal_model, os.path.join(MODEL_DIR, "away_goal_model.pkl"))
joblib.dump(imputer_p,       os.path.join(MODEL_DIR, "poisson_imputer.pkl"))
joblib.dump(scaler_p,        os.path.join(MODEL_DIR, "poisson_scaler.pkl"))
joblib.dump(rf_model,        os.path.join(MODEL_DIR, "rf_model.pkl"))
if HAS_XGB:
    joblib.dump(xgb_model,   os.path.join(MODEL_DIR, "xgb_model.pkl"))

print("  Kaydedilen modeller:")
for f in sorted(os.listdir(MODEL_DIR)):
    sz = os.path.getsize(os.path.join(MODEL_DIR, f)) / 1024
    print(f"    {f:<35} {sz:.0f} KB")


# ── Tahminler kaydetme ────────────────────────────────────────────────────────
pred_cols = [
    "match_id", "group", "date_utc", "venue",
    "home_team", "away_team",
    "elo_home", "elo_away", "elo_diff",
    "p_home", "p_draw", "p_away",
    "lambda_home", "lambda_away",
    "over_2_5", "btts",
    "top_scorelines", "favourite", "upset_risk", "upset_label",
]
avail_cols  = [c for c in pred_cols if c in future.columns]
predictions = future[avail_cols].copy()
pred_path   = os.path.join(PROCESSED_DIR, "predictions_latest.csv")
predictions.to_csv(pred_path, index=False)
print(f"\n  ✅  predictions_latest.csv → {len(predictions)} maç, {len(predictions.columns)} kolon")

# ── Per-model 2026 tahminleri ─────────────────────────────────────────────────
print("\n[9] Per-model 2026 tahminleri kaydediliyor...")

all_model_rows = []

base_cols = ["match_id", "group", "home_team", "away_team"]

def _add_model_rows(name, prob_arr, lh_arr=None, la_arr=None):
    for i, (_, row) in enumerate(future.iterrows()):
        ph, pd_, pa = float(prob_arr[i, 0]), float(prob_arr[i, 1]), float(prob_arr[i, 2])
        all_model_rows.append({
            "match_id":   int(row["match_id"]),
            "group":      row.get("group", ""),
            "home_team":  row["home_team"],
            "away_team":  row["away_team"],
            "model":      name,
            "p_home":     round(ph, 4),
            "p_draw":     round(pd_, 4),
            "p_away":     round(pa, 4),
            "lambda_home": round(float(lh_arr[i]), 3) if lh_arr is not None else None,
            "lambda_away": round(float(la_arr[i]), 3) if la_arr is not None else None,
        })

# Elo Baseline
elo_future_prob = elo_prob_matrix(X_future)
_add_model_rows("Elo Baseline", elo_future_prob)

# LR
_add_model_rows("LR", lr_model.predict_proba(X_future_t))

# Poisson — extract lambda arrays from already computed poisson_df
_lh = future["lambda_home"].values if "lambda_home" in future.columns else None
_la = future["lambda_away"].values if "lambda_away" in future.columns else None
poi_future = np.column_stack([
    future["p_home_poi"].values,
    future["p_draw_poi"].values,
    future["p_away_poi"].values,
])
_add_model_rows("Poisson", poi_future, _lh, _la)

# Ensemble
ens_future = np.column_stack([
    future["p_home"].values,
    future["p_draw"].values,
    future["p_away"].values,
])
_add_model_rows("Ensemble", ens_future, _lh, _la)

# Random Forest
_add_model_rows("Random Forest", rf_model.predict_proba(X_future_t))

# XGBoost
if HAS_XGB:
    _add_model_rows("XGBoost", xgb_model.predict_proba(X_future_t))

all_models_df = pd.DataFrame(all_model_rows)
all_models_path = os.path.join(PROCESSED_DIR, "predictions_all_models.csv")
all_models_df.to_csv(all_models_path, index=False)
print(f"  ✅  predictions_all_models.csv → {len(all_models_df)} satır "
      f"({all_models_df['model'].nunique()} model × {len(future)} maç)")

print("\nTüm dosyalar hazır ✅")
