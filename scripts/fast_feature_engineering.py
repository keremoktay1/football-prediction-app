"""
fast_feature_engineering.py

Notebook 02'nin hızlandırılmış versiyonu.
Row-by-row döngü yerine pre-indexed team histories kullanarak
~30x hızlanma sağlar.

Çıktılar:
  data/processed/features_historical.csv
  data/processed/features_2026_fixtures.csv
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

from config import FILES, TEAM_NAME_MAP, ROLLING_FORM_WINDOW, BASE_GOAL_RATE

PROCESSED_DIR = os.path.join(APP_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ── Veri yükleme ─────────────────────────────────────────────────────────────
print("Veriler yükleniyor...")

def load_csv(key, parse_dates=None):
    path = FILES[key]
    if not os.path.isfile(path):
        print(f"[WARN] {key} bulunamadı: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=parse_dates, low_memory=False)
    print(f"[OK]  {key:<20} → {len(df):>7,} satır")
    return df

def standardize(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].replace(TEAM_NAME_MAP)
    return df

results        = load_csv("results",        parse_dates=["date"])
elo_raw        = load_csv("elo",            parse_dates=["snapshot_date"])
group_fixtures = load_csv("group_fixtures", parse_dates=["date_utc"])
knockout_slots = load_csv("knockout_slots", parse_dates=["date_utc"])

results        = standardize(results,        ["home_team", "away_team"])
elo_raw        = standardize(elo_raw,        ["country"])
group_fixtures = standardize(group_fixtures, ["home_team", "away_team"])

print("Standardizasyon tamamlandı ✅")


# ── Elo lookup (pre-indexed) ──────────────────────────────────────────────────
print("\nElo lookup hazırlanıyor...")
elo_sorted = elo_raw[["snapshot_date", "country", "rating"]].copy()
elo_sorted = elo_sorted.sort_values("snapshot_date")

elo_lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
for country, grp in elo_sorted.groupby("country"):
    dates   = grp["snapshot_date"].values.astype("datetime64[ns]")
    ratings = grp["rating"].values.astype(float)
    elo_lookup[country] = (dates, ratings)

def fast_elo(team: str, date: pd.Timestamp) -> float:
    if team not in elo_lookup:
        return np.nan
    dates, ratings = elo_lookup[team]
    ts = np.datetime64(date, "ns")
    idx = np.searchsorted(dates, ts, side="right") - 1
    return float(ratings[idx]) if idx >= 0 else np.nan

print(f"Elo lookup hazır: {len(elo_lookup)} takım")


# ── Turnuva ağırlıkları ───────────────────────────────────────────────────────
TOURNAMENT_WEIGHTS = {
    "FIFA World Cup":          4.0,
    "UEFA Euro":               3.5,
    "Copa America":            3.5,
    "Africa Cup of Nations":   3.0,
    "AFC Asian Cup":           3.0,
    "CONCACAF Gold Cup":       3.0,
    "UEFA Nations League":     2.5,
    "Confederations Cup":      2.5,
    "World Cup qualification": 2.0,
    "UEFA Euro qualification": 1.5,
    "Friendly":                1.0,
}

def get_tournament_weight(tournament: str) -> float:
    if pd.isna(tournament):
        return 1.0
    t_lower = tournament.lower()
    for key, w in TOURNAMENT_WEIGHTS.items():
        if key.lower() in t_lower:
            return w
    return 1.5

results["tournament_weight"] = results["tournament"].apply(get_tournament_weight)


# ── Team history pre-indexing ─────────────────────────────────────────────────
print("\nTakım geçmişleri indeksleniyor...")

results_clean = results.dropna(subset=["home_score", "away_score"]).copy()
results_clean["home_score"] = results_clean["home_score"].astype(int)
results_clean["away_score"] = results_clean["away_score"].astype(int)

def _points(gf, ga):
    if gf > ga:   return 3
    if gf == ga:  return 1
    return 0

# Build per-team sorted arrays: (dates, goals_for, goals_against, points, tournament_weight)
team_history: dict[str, dict] = {}

all_teams = set(results_clean["home_team"]).union(set(results_clean["away_team"]))
for team in all_teams:
    home = results_clean[results_clean["home_team"] == team][
        ["date", "home_score", "away_score", "tournament_weight"]
    ].rename(columns={"home_score": "gf", "away_score": "ga"})

    away = results_clean[results_clean["away_team"] == team][
        ["date", "away_score", "home_score", "tournament_weight"]
    ].rename(columns={"away_score": "gf", "home_score": "ga"})

    combined = pd.concat([home, away], ignore_index=True).sort_values("date")
    combined["pts"] = combined.apply(lambda r: _points(r["gf"], r["ga"]), axis=1)

    team_history[team] = {
        "dates":   combined["date"].values.astype("datetime64[ns]"),
        "gf":      combined["gf"].values.astype(float),
        "ga":      combined["ga"].values.astype(float),
        "pts":     combined["pts"].values.astype(float),
        "tw":      combined["tournament_weight"].values.astype(float),
    }

print(f"Takım geçmişleri hazır: {len(team_history)} takım")


# ── Rolling form ──────────────────────────────────────────────────────────────
def get_rolling_form(team: str, date: pd.Timestamp, n: int = ROLLING_FORM_WINDOW) -> dict:
    if team not in team_history:
        return {"points_last_n": np.nan, "goal_diff_last_n": np.nan,
                "gf_last_n": np.nan, "ga_last_n": np.nan,
                "weighted_form": np.nan, "matches_played": 0}

    th = team_history[team]
    ts = np.datetime64(date, "ns")
    end_idx = int(np.searchsorted(th["dates"], ts, side="left"))  # strict past only

    if end_idx == 0:
        return {"points_last_n": np.nan, "goal_diff_last_n": np.nan,
                "gf_last_n": np.nan, "ga_last_n": np.nan,
                "weighted_form": np.nan, "matches_played": 0}

    start_idx = max(0, end_idx - n)
    pts  = th["pts"][start_idx:end_idx]
    gf   = th["gf"][start_idx:end_idx]
    ga   = th["ga"][start_idx:end_idx]

    weights = np.exp(np.linspace(0, 1, len(pts)))
    weights /= weights.sum()
    weighted_pts = float((pts * weights).sum() * n)

    return {
        "points_last_n":    float(pts.sum()),
        "goal_diff_last_n": float((gf - ga).sum()),
        "gf_last_n":        float(gf.sum()),
        "ga_last_n":        float(ga.sum()),
        "weighted_form":    weighted_pts,
        "matches_played":   len(pts),
    }


# ── Attack / Defense ──────────────────────────────────────────────────────────
def get_attack_defense(team: str, date: pd.Timestamp, window: int = 20) -> dict:
    if team not in team_history:
        return {"attack_strength": 1.0, "defense_weakness": 1.0, "form_matches": 0}

    th = team_history[team]
    ts = np.datetime64(date, "ns")
    end_idx = int(np.searchsorted(th["dates"], ts, side="left"))

    if end_idx < 3:
        return {"attack_strength": 1.0, "defense_weakness": 1.0, "form_matches": 0}

    start_idx = max(0, end_idx - window)
    gf = th["gf"][start_idx:end_idx]
    ga = th["ga"][start_idx:end_idx]

    avg_gf = float(gf.mean())
    avg_ga = float(ga.mean())
    global_avg = BASE_GOAL_RATE

    return {
        "attack_strength":  round(avg_gf / global_avg if global_avg > 0 else 1.0, 4),
        "defense_weakness": round(avg_ga / global_avg if global_avg > 0 else 1.0, 4),
        "form_matches":     end_idx - start_idx,
    }


# ── Upset risk ────────────────────────────────────────────────────────────────
def compute_upset_risk(elo_diff: float, underdog_form: float,
                       favorite_form: float, neutral: int) -> float:
    if np.isnan(elo_diff) or elo_diff == 0:
        return 0.5
    abs_diff = abs(elo_diff)
    elo_component = 1 - min(abs_diff / 600.0, 1.0)
    underdog_form_score = 0.5
    if not np.isnan(underdog_form) and underdog_form > 0:
        underdog_form_score = min(underdog_form / (ROLLING_FORM_WINDOW * 3), 1.0)
    fav_neg_momentum = 0.5
    if not np.isnan(favorite_form) and favorite_form > 0:
        fav_neg_momentum = 1 - min(favorite_form / (ROLLING_FORM_WINDOW * 3), 1.0)
    neutral_component = 0.3 if neutral else 0.0
    upset_risk = (
        0.40 * elo_component
        + 0.25 * underdog_form_score
        + 0.20 * fav_neg_momentum
        + 0.10 * neutral_component
        + 0.05 * 0.5
    )
    return round(min(max(upset_risk, 0.0), 1.0), 4)

def upset_label(score: float) -> str:
    if score >= 0.65: return "Yüksek"
    if score >= 0.45: return "Orta"
    return "Düşük"


# ── Tarihi maç feature matrisi ────────────────────────────────────────────────
print("\nTarihi feature matrisi oluşturuluyor...")

results_modern = results_clean[results_clean["date"] >= "2000-01-01"].reset_index(drop=True)
print(f"  2000+ maç sayısı: {len(results_modern):,}")

feature_rows = []
total = len(results_modern)
report_step = max(1, total // 20)

for i, (_, row) in enumerate(results_modern.iterrows()):
    if i % report_step == 0:
        print(f"  İlerleme: {i:>6,}/{total:,} ({100*i//total}%)", end="\r")

    home = row["home_team"]
    away = row["away_team"]
    date = row["date"]

    elo_home = fast_elo(home, date)
    elo_away = fast_elo(away, date)
    elo_diff = elo_home - elo_away if (not np.isnan(elo_home) and not np.isnan(elo_away)) else np.nan

    form_home = get_rolling_form(home, date)
    form_away = get_rolling_form(away, date)
    form_diff = (
        form_home["weighted_form"] - form_away["weighted_form"]
        if (not np.isnan(form_home["weighted_form"]) and not np.isnan(form_away["weighted_form"]))
        else np.nan
    )

    ad_home = get_attack_defense(home, date)
    ad_away = get_attack_defense(away, date)

    hs, as_ = int(row["home_score"]), int(row["away_score"])
    if hs > as_:    result_label = "H"
    elif hs == as_: result_label = "D"
    else:           result_label = "A"

    feature_rows.append({
        "date":               date,
        "home_team":          home,
        "away_team":          away,
        "elo_home":           elo_home,
        "elo_away":           elo_away,
        "elo_diff":           elo_diff,
        "form_home_pts":      form_home["points_last_n"],
        "form_away_pts":      form_away["points_last_n"],
        "form_diff":          form_diff,
        "weighted_form_home": form_home["weighted_form"],
        "weighted_form_away": form_away["weighted_form"],
        "gf_home_last_n":     form_home["gf_last_n"],
        "ga_home_last_n":     form_home["ga_last_n"],
        "gf_away_last_n":     form_away["gf_last_n"],
        "ga_away_last_n":     form_away["ga_last_n"],
        "attack_home":        ad_home["attack_strength"],
        "defense_home":       ad_home["defense_weakness"],
        "attack_away":        ad_away["attack_strength"],
        "defense_away":       ad_away["defense_weakness"],
        "neutral":            int(row.get("neutral", 0)),
        "tournament_weight":  get_tournament_weight(row.get("tournament", "")),
        "home_score":         hs,
        "away_score":         as_,
        "result":             result_label,
    })

features_df = pd.DataFrame(feature_rows)
print(f"\n  Feature matrisi hazır: {features_df.shape}")


# ── 2026 fikstür feature matrisi ──────────────────────────────────────────────
print("\n2026 fikstür feature matrisi oluşturuluyor...")

future_rows = []
for _, row in group_fixtures.iterrows():
    home = row["home_team"]
    away = row["away_team"]
    date = row["date_utc"]

    elo_home = fast_elo(home, date)
    elo_away = fast_elo(away, date)
    elo_diff = elo_home - elo_away if (not np.isnan(elo_home) and not np.isnan(elo_away)) else np.nan

    form_home = get_rolling_form(home, date)
    form_away = get_rolling_form(away, date)
    form_diff = (
        form_home["weighted_form"] - form_away["weighted_form"]
        if (not np.isnan(form_home["weighted_form"]) and not np.isnan(form_away["weighted_form"]))
        else np.nan
    )

    ad_home = get_attack_defense(home, date)
    ad_away = get_attack_defense(away, date)

    future_rows.append({
        "match_id":           row["match_id"],
        "group":              row["group"],
        "date_utc":           date,
        "venue":              row["venue"],
        "home_team":          home,
        "away_team":          away,
        "elo_home":           elo_home,
        "elo_away":           elo_away,
        "elo_diff":           elo_diff,
        "form_home_pts":      form_home["points_last_n"],
        "form_away_pts":      form_away["points_last_n"],
        "form_diff":          form_diff,
        "weighted_form_home": form_home["weighted_form"],
        "weighted_form_away": form_away["weighted_form"],
        "gf_home_last_n":     form_home["gf_last_n"],
        "ga_home_last_n":     form_home["ga_last_n"],
        "gf_away_last_n":     form_away["gf_last_n"],
        "ga_away_last_n":     form_away["ga_last_n"],
        "attack_home":        ad_home["attack_strength"],
        "defense_home":       ad_home["defense_weakness"],
        "attack_away":        ad_away["attack_strength"],
        "defense_away":       ad_away["defense_weakness"],
        "neutral":            1,
        "tournament_weight":  4.0,
    })

future_df = pd.DataFrame(future_rows)

# Upset risk ekle
future_df["upset_risk"] = future_df.apply(
    lambda r: compute_upset_risk(
        r["elo_diff"], r["weighted_form_away"], r["weighted_form_home"], r["neutral"]
    ),
    axis=1,
)
future_df["upset_label"] = future_df["upset_risk"].apply(upset_label)

print(f"  2026 feature matrisi hazır: {future_df.shape}")

missing_elo = future_df[future_df["elo_diff"].isna()][["home_team", "away_team", "elo_home", "elo_away"]]
if not missing_elo.empty:
    print(f"  ⚠️  Elo eksik maçlar ({len(missing_elo)}):")
    print(missing_elo.to_string())
else:
    print("  ✅  Tüm 2026 maçları için Elo mevcut")


# ── Kaydet ────────────────────────────────────────────────────────────────────
features_path = os.path.join(PROCESSED_DIR, "features_historical.csv")
future_path   = os.path.join(PROCESSED_DIR, "features_2026_fixtures.csv")

features_df.to_csv(features_path, index=False)
future_df.to_csv(future_path, index=False)

print(f"\n✅  features_historical.csv    → {os.path.getsize(features_path)/1024:.0f} KB")
print(f"✅  features_2026_fixtures.csv → {os.path.getsize(future_path)/1024:.0f} KB")

# Kısa özet
print("\nHedef dağılımı:")
vc = features_df["result"].value_counts()
for label, count in vc.items():
    print(f"  {label}: {count:>6,}  ({count/len(features_df)*100:.1f}%)")

print("\nEksik değer oranları:")
for c in ["elo_diff", "form_diff", "attack_home", "defense_home", "attack_away", "defense_away"]:
    pct = features_df[c].isna().mean() * 100
    print(f"  {c:<25}: {pct:.1f}%")
