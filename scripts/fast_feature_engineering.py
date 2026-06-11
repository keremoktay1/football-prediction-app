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
import math
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# src path
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(APP_DIR, "src"))

from config import FILES, TEAM_NAME_MAP, ROLLING_FORM_WINDOW, BASE_GOAL_RATE

RAW_DIR       = os.path.join(APP_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(APP_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Squad stats yükleme ───────────────────────────────────────────────────────
_squad_path = os.path.join(PROCESSED_DIR, "squad_stats.csv")
if os.path.isfile(_squad_path):
    _squad_df = pd.read_csv(_squad_path).set_index("team")
    _squad_map: dict = _squad_df.to_dict("index")
    _global_avg_age  = float(_squad_df["avg_age"].mean())
    _global_mv       = float(_squad_df["market_value_proxy"].mean())
    _global_t5       = float(_squad_df["top5_league_count"].mean())
    _global_gp90     = float(_squad_df["goals_per90"].mean())     if "goals_per90"   in _squad_df.columns else 0.09
    _global_ap90     = float(_squad_df["assists_per90"].mean())   if "assists_per90" in _squad_df.columns else 0.07
    _global_fgp90    = float(_squad_df["forward_goals_p90"].mean()) if "forward_goals_p90" in _squad_df.columns else 0.20
    print(f"[OK]  squad_stats.csv yüklendi: {len(_squad_df)} takım")
else:
    _squad_map = {}
    _global_avg_age, _global_mv, _global_t5 = 26.0, 30.0, 10.0
    _global_gp90, _global_ap90, _global_fgp90 = 0.09, 0.07, 0.20
    print("[INFO] squad_stats.csv bulunamadı — global ortalamalar kullanılacak")

def get_squad_feat(team: str, feat: str) -> float:
    """Squad stats'dan bir özellik döner. Bulunamazsa global ortalama."""
    if team in _squad_map:
        val = _squad_map[team].get(feat, None)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            return float(val)
    defaults = {
        "avg_age":            _global_avg_age,
        "market_value_proxy": _global_mv,
        "top5_league_count":  _global_t5,
        "goals_per90":        _global_gp90,
        "assists_per90":      _global_ap90,
        "forward_goals_p90":  _global_fgp90,
    }
    return defaults.get(feat, 0.0)


# ── Venue lookup (WC 2026 stadyumları) ───────────────────────────────────────
_venues_path = os.path.join(RAW_DIR, "wc2026_venues.csv")
_venue_map: dict = {}  # venue_key → {altitude_m, avg_june_temp_c, lat, lon}
if os.path.isfile(_venues_path):
    _vdf = pd.read_csv(_venues_path)
    for _, vrow in _vdf.iterrows():
        _venue_map[str(vrow["venue_key"]).strip()] = {
            "altitude_m":       float(vrow["altitude_m"]),
            "temp_celsius":     float(vrow["avg_june_temp_c"]),
            "lat":              float(vrow["lat"]),
            "lon":              float(vrow["lon"]),
        }
    print(f"[OK]  wc2026_venues.csv yüklendi: {len(_venue_map)} statdyum")
else:
    print("[INFO] wc2026_venues.csv bulunamadı — venue features atlanacak")

def _match_venue(venue_str: str) -> dict | None:
    """Venue string'inden altitude ve sıcaklık döner."""
    if not venue_str or pd.isna(venue_str):
        return None
    v = str(venue_str).strip()
    for key, data in _venue_map.items():
        if key.lower() in v.lower() or v.lower().startswith(key.lower()):
            return data
    # Fuzzy: first significant word match
    first_word = v.split(",")[0].split(" ")[0].lower()
    for key, data in _venue_map.items():
        if first_word in key.lower():
            return data
    return None

def get_venue_features(venue_str: str) -> dict:
    """Statdyum özelliklerini döner. Bulunamazsa global ortalama."""
    info = _match_venue(venue_str)
    if info:
        return {"altitude_m": info["altitude_m"], "temp_celsius": info["temp_celsius"],
                "venue_lat": info["lat"], "venue_lon": info["lon"]}
    return {"altitude_m": 200.0, "temp_celsius": 22.0, "venue_lat": 40.0, "venue_lon": -95.0}


# ── Takım ana lokasyonu (seyahat mesafesi için) ───────────────────────────────
TEAM_HOME_COORDS: dict[str, tuple[float, float]] = {
    # Kuzey Amerika
    "Mexico":            (19.43, -99.13),
    "United States":     (39.50, -98.35),
    "Canada":            (45.42, -75.70),
    # Güney Amerika
    "Brazil":            (-15.78, -47.93),
    "Argentina":         (-34.60, -58.38),
    "Uruguay":           (-34.90, -56.16),
    "Colombia":          (4.71,  -74.07),
    "Ecuador":           (-0.18, -78.47),
    "Peru":              (-12.05, -77.04),
    "Paraguay":          (-25.26, -57.58),
    "Venezuela":         (10.48, -66.90),
    "Bolivia":           (-16.50, -68.15),
    "Chile":             (-33.46, -70.65),
    "Haiti":             (18.59, -72.31),
    "Honduras":          (14.07, -87.21),
    "Costa Rica":        (9.93,  -84.09),
    "Panama":            (8.99,  -79.52),
    "Jamaica":           (17.97, -76.79),
    # Avrupa
    "France":            (48.86,   2.35),
    "Germany":           (52.52,  13.41),
    "Spain":             (40.42,  -3.70),
    "England":           (51.51,  -0.13),
    "Portugal":          (38.72,  -9.14),
    "Netherlands":       (52.37,   4.90),
    "Belgium":           (50.85,   4.35),
    "Italy":             (41.90,  12.50),
    "Croatia":           (45.82,  15.98),
    "Switzerland":       (46.95,   7.45),
    "Serbia":            (44.82,  20.46),
    "Poland":            (52.23,  21.01),
    "Austria":           (48.21,  16.37),
    "Hungary":           (47.50,  19.04),
    "Turkey":            (39.93,  32.86),
    "Ukraine":           (50.45,  30.52),
    "Scotland":          (55.86,  -4.25),
    "Denmark":           (55.68,  12.57),
    "Sweden":            (59.33,  18.07),
    "Norway":            (59.91,  10.75),
    "Greece":            (37.98,  23.73),
    "Romania":           (44.43,  26.10),
    "Czech Republic":    (50.08,  14.44),
    "Czechia":           (50.08,  14.44),
    "Slovakia":          (48.15,  17.11),
    "Bosnia and Herzegovina": (43.85, 18.36),
    "North Macedonia":   (41.99,  21.43),
    "Albania":           (41.33,  19.82),
    "Slovenia":          (46.05,  14.51),
    "Iceland":           (64.14, -21.94),
    "Wales":             (51.48,  -3.18),
    "Northern Ireland":  (54.60,  -5.93),
    "Republic of Ireland": (53.33, -6.25),
    "Ireland":           (53.33,  -6.25),
    "Curaçao":           (12.12, -68.88),
    # Afrika
    "Morocco":           (33.99,  -6.85),
    "Senegal":           (14.72, -17.47),
    "Nigeria":           (9.08,   8.68),
    "Cameroon":          (3.85,  11.50),
    "Ghana":             (5.56,  -0.20),
    "Côte d'Ivoire":     (5.36,  -4.01),
    "South Africa":      (-25.75, 28.19),
    "Egypt":             (30.04,  31.24),
    "Tunisia":           (36.82,  10.17),
    "Algeria":           (36.74,   3.09),
    "Mali":              (12.65,  -8.00),
    # Asya
    "Japan":             (35.68, 139.69),
    "South Korea":       (37.57, 126.98),
    "Iran":              (35.69,  51.39),
    "Saudi Arabia":      (24.69,  46.72),
    "Qatar":             (25.29,  51.53),
    "Australia":         (-35.31, 149.12),
    "New Zealand":       (-41.29, 174.78),
    "Indonesia":         (-6.21, 106.85),
    "Philippines":       (14.60, 120.98),
    "Thailand":          (13.76, 100.50),
    "Uzbekistan":        (41.30,  69.24),
    "Iraq":              (33.34,  44.40),
}

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """İki koordinat arasında km cinsinden büyük daire mesafesi."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi   = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return round(2 * R * math.asin(math.sqrt(a)))

def get_travel_km(team: str, venue_lat: float, venue_lon: float) -> float:
    """Takımın ülkesinden statdyuma km mesafesi."""
    coords = TEAM_HOME_COORDS.get(team)
    if coords is None:
        return 5000.0  # varsayılan: uzak takım
    return haversine_km(coords[0], coords[1], venue_lat, venue_lon)


# ── Antrenör (teknik direktör) lookup ─────────────────────────────────────────
_coaches_path = os.path.join(RAW_DIR, "wc2026_coaches.csv")
_coach_map: dict = {}
if os.path.isfile(_coaches_path):
    _cdf = pd.read_csv(_coaches_path)
    for _, crow in _cdf.iterrows():
        _coach_map[str(crow["team"]).strip()] = {
            "win_rate":    float(crow["win_rate"]),
            "wc_apps":     int(crow["wc_apps_as_coach"]),
            "intl_titles": int(crow["intl_titles"]),
        }
    print(f"[OK]  wc2026_coaches.csv yüklendi: {len(_coach_map)} antrenör")
else:
    print("[INFO] wc2026_coaches.csv bulunamadı — antrenör features atlanacak")

def get_coach_feat(team: str, feat: str) -> float:
    """Antrenör istatistiği döner. Bulunamazsa ortalama."""
    defaults = {"win_rate": 0.48, "wc_apps": 0.0, "intl_titles": 0.0}
    if team in _coach_map:
        return float(_coach_map[team].get(feat, defaults.get(feat, 0.0)))
    return float(defaults.get(feat, 0.0))


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

# ── Playoff override'larını uygula (gerçek takım adları → Elo/form verisi bulunabilir)
_overrides_path = os.path.join(PROCESSED_DIR, "playoff_overrides.csv")
if os.path.isfile(_overrides_path):
    _ov = pd.read_csv(_overrides_path, dtype=str)
    _override_map = {
        str(r["slot_name"]).strip(): str(r["actual_team"]).strip()
        for _, r in _ov.iterrows()
        if str(r.get("actual_team", "")).strip() not in ("", "nan", "None")
    }
    if _override_map:
        group_fixtures["home_team"] = group_fixtures["home_team"].replace(_override_map)
        group_fixtures["away_team"] = group_fixtures["away_team"].replace(_override_map)
        print(f"  Playoff override uygulandı: {_override_map}")

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

# ── 2000 öncesini filtrele ────────────────────────────────────────────────────
results_clean = results_clean[
    results_clean["date"] >= pd.Timestamp("2000-01-01")
].reset_index(drop=True)
print(f"  2000+ maç filtresi uygulandı: {len(results_clean):,} satır")

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


# ── Experience score (son N yıldaki uluslararası maç sayısı) ──────────────────
def get_experience_score(team: str, date: pd.Timestamp, years: int = 3) -> int:
    """Takımın verilen tarihten önceki son `years` yıldaki maç sayısı."""
    if team not in team_history:
        return 0
    th = team_history[team]
    ts     = np.datetime64(date, "ns")
    cutoff = np.datetime64(date - pd.DateOffset(years=years), "ns")
    end_idx   = int(np.searchsorted(th["dates"], ts,     side="left"))
    start_idx = int(np.searchsorted(th["dates"], cutoff, side="left"))
    return max(0, end_idx - start_idx)


def get_goal_trend(team: str, date: pd.Timestamp) -> float:
    """Pozitif = artan gol trendi, negatif = azalan."""
    if team not in team_history: return 0.0
    th = team_history[team]
    ts = np.datetime64(date, "ns")
    end_idx = int(np.searchsorted(th["dates"], ts, side="left"))
    if end_idx < 3: return 0.0
    gf_5  = th["gf"][max(0, end_idx-5):end_idx]
    gf_10 = th["gf"][max(0, end_idx-10):end_idx]
    avg5  = float(gf_5.mean())  if len(gf_5)  > 0 else 0.0
    avg10 = float(gf_10.mean()) if len(gf_10) > 0 else 0.0
    return round(avg5 - avg10, 4)


def get_form_consistency(team: str, date: pd.Timestamp,
                          n: int = ROLLING_FORM_WINDOW) -> float:
    if team not in team_history: return 1.0
    th = team_history[team]
    ts = np.datetime64(date, "ns")
    end_idx = int(np.searchsorted(th["dates"], ts, side="left"))
    if end_idx < 2: return 1.0
    pts = th["pts"][max(0, end_idx-n):end_idx]
    return round(float(pts.std()), 4) if len(pts) > 1 else 1.0


# ── H2H pre-indexing ──────────────────────────────────────────────────────────
print("\nH2H ve ortak rakip indexleri oluşturuluyor...")

# h2h_history: (team_a, team_b) alfabetik → sorted arrays
# diff: team_a lehine gol farkı (team_a - team_b)
_h2h_tmp: dict[tuple, dict] = {}
for _, row in results_clean.iterrows():
    h, a = row["home_team"], row["away_team"]
    hs, as_ = int(row["home_score"]), int(row["away_score"])
    d = row["date"]
    key = (min(h, a), max(h, a))
    # diff from perspective of min-alphabetical team
    diff = (hs - as_) if h <= a else (as_ - hs)
    if key not in _h2h_tmp:
        _h2h_tmp[key] = {"dates": [], "diff": []}
    _h2h_tmp[key]["dates"].append(d)
    _h2h_tmp[key]["diff"].append(diff)

h2h_history: dict[tuple, dict] = {}
for key, data in _h2h_tmp.items():
    sorted_pairs = sorted(zip(data["dates"], data["diff"]))
    h2h_history[key] = {
        "dates": np.array([x[0] for x in sorted_pairs], dtype="datetime64[ns]"),
        "diff":  np.array([x[1] for x in sorted_pairs], dtype=float),
    }

# team_vs_opp: team → opponent → sorted arrays (dates, gf, ga)
_tvo_tmp: dict[str, dict[str, dict]] = {}
for _, row in results_clean.iterrows():
    h, a = row["home_team"], row["away_team"]
    hs, as_ = int(row["home_score"]), int(row["away_score"])
    d = row["date"]
    for team, opp, gf, ga in [(h, a, hs, as_), (a, h, as_, hs)]:
        if team not in _tvo_tmp:
            _tvo_tmp[team] = {}
        if opp not in _tvo_tmp[team]:
            _tvo_tmp[team][opp] = {"dates": [], "gf": [], "ga": []}
        _tvo_tmp[team][opp]["dates"].append(d)
        _tvo_tmp[team][opp]["gf"].append(gf)
        _tvo_tmp[team][opp]["ga"].append(ga)

team_vs_opp: dict[str, dict[str, dict]] = {}
for team, opps in _tvo_tmp.items():
    team_vs_opp[team] = {}
    for opp, data in opps.items():
        sorted_triples = sorted(zip(data["dates"], data["gf"], data["ga"]))
        team_vs_opp[team][opp] = {
            "dates": np.array([x[0] for x in sorted_triples], dtype="datetime64[ns]"),
            "gf":    np.array([x[1] for x in sorted_triples], dtype=float),
            "ga":    np.array([x[2] for x in sorted_triples], dtype=float),
        }

print(f"H2H indexi hazır: {len(h2h_history)} çift")
print(f"Takım-rakip indexi hazır: {len(team_vs_opp)} takım")


# ── Rolling form ──────────────────────────────────────────────────────────────
_EMPTY_FORM = {
    "points_last_n": np.nan, "goal_diff_last_n": np.nan,
    "gf_last_n": np.nan, "ga_last_n": np.nan,
    "weighted_form": np.nan, "matches_played": 0,
    "win_streak": 0, "loss_streak": 0,
    "clean_sheet_rate": np.nan, "failed_to_score_rate": np.nan,
}

def get_rolling_form(team: str, date: pd.Timestamp, n: int = ROLLING_FORM_WINDOW) -> dict:
    if team not in team_history:
        return _EMPTY_FORM.copy()

    th = team_history[team]
    ts = np.datetime64(date, "ns")
    end_idx = int(np.searchsorted(th["dates"], ts, side="left"))  # strict past only

    if end_idx == 0:
        return _EMPTY_FORM.copy()

    start_idx = max(0, end_idx - n)
    pts  = th["pts"][start_idx:end_idx]
    gf   = th["gf"][start_idx:end_idx]
    ga   = th["ga"][start_idx:end_idx]

    weights = np.exp(np.linspace(0, 1, len(pts)))
    weights /= weights.sum()
    weighted_pts = float((pts * weights).sum() * n)

    # Streak: consecutive wins/losses from the most recent match backward
    pts_list = pts.tolist()
    win_streak = 0
    for p in reversed(pts_list):
        if p == 3.0:
            win_streak += 1
        else:
            break
    loss_streak = 0
    for p in reversed(pts_list):
        if p == 0.0:
            loss_streak += 1
        else:
            break

    n_m = len(pts)
    cs_rate  = float((ga == 0).mean()) if n_m > 0 else np.nan
    fts_rate = float((gf == 0).mean()) if n_m > 0 else np.nan

    return {
        "points_last_n":        float(pts.sum()),
        "goal_diff_last_n":     float((gf - ga).sum()),
        "gf_last_n":            float(gf.sum()),
        "ga_last_n":            float(ga.sum()),
        "weighted_form":        weighted_pts,
        "matches_played":       n_m,
        "win_streak":           win_streak,
        "loss_streak":          loss_streak,
        "clean_sheet_rate":     cs_rate,
        "failed_to_score_rate": fts_rate,
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


# ── H2H goal diff ─────────────────────────────────────────────────────────────
def get_h2h_goal_diff(home: str, away: str, date: pd.Timestamp, n: int = 5) -> float:
    """
    Son n H2H maçta ev sahibi lehine ortalama gol farkı.
    Veri yoksa 0.0 döner.
    """
    key = (min(home, away), max(home, away))
    if key not in h2h_history:
        return 0.0
    ts = np.datetime64(date, "ns")
    h2h = h2h_history[key]
    end_idx = int(np.searchsorted(h2h["dates"], ts, side="left"))
    if end_idx == 0:
        return 0.0
    start_idx = max(0, end_idx - n)
    diffs = h2h["diff"][start_idx:end_idx]
    # diff is from perspective of min(home,away); flip if home is the larger
    sign = 1.0 if home <= away else -1.0
    return float(diffs.mean() * sign)


# ── Common opponent score diff ────────────────────────────────────────────────
def get_common_opponent_diff(
    home: str, away: str, date: pd.Timestamp, window: int = 20
) -> float:
    """
    Ortak rakiplere karşı son window maçtaki ortalama gol farkı farkı.
    home_avg_gd(vs commons) - away_avg_gd(vs commons). Yoksa 0.0.
    """
    if home not in team_vs_opp or away not in team_vs_opp:
        return 0.0
    ts = np.datetime64(date, "ns")

    def _opp_stats(team: str) -> dict[str, float]:
        """opp → ortalama gol farkı (team - opp), son window maç"""
        stats: dict[str, float] = {}
        if team not in team_vs_opp:
            return stats
        for opp, data in team_vs_opp[team].items():
            end_idx = int(np.searchsorted(data["dates"], ts, side="left"))
            if end_idx == 0:
                continue
            s = max(0, end_idx - window)
            gf = data["gf"][s:end_idx]
            ga = data["ga"][s:end_idx]
            stats[opp] = float((gf - ga).mean())
        return stats

    home_stats = _opp_stats(home)
    away_stats = _opp_stats(away)

    commons = set(home_stats) & set(away_stats)
    if not commons:
        return 0.0

    home_avg = float(np.mean([home_stats[c] for c in commons]))
    away_avg = float(np.mean([away_stats[c] for c in commons]))
    return round(home_avg - away_avg, 4)


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


def compute_upset_risk_v2(
    elo_diff: float,
    underdog_form: float,
    favorite_form: float,
    neutral: int,
    underdog_mv: float = 0.0,
    favorite_mv: float = 0.0,
    underdog_exp: int = 0,
    favorite_exp: int = 0,
) -> float:
    """
    Genişletilmiş sürpriz riski (2026 fikstürleri için).
    Squad stats mevcut olduğunda market value ve deneyim bileşenlerini kullanır.
    """
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

    # Piyasa değeri bileşeni: ELO'ya kıyasla market value'su yüksek underdog
    value_component = 0.5
    if underdog_mv > 0 and favorite_mv > 0:
        mv_ratio  = underdog_mv / favorite_mv
        elo_ratio = 1 / (1 + 10 ** (-elo_diff / 400))
        value_component = min(1.0, max(0.0, 0.5 + (mv_ratio - elo_ratio) * 2))

    # Deneyim bileşeni
    exp_component = 0.6 if underdog_exp > 50 else (0.4 if underdog_exp > 30 else 0.3)

    upset_risk = (
        0.35 * elo_component
        + 0.20 * underdog_form_score
        + 0.15 * fav_neg_momentum
        + 0.10 * neutral_component
        + 0.10 * value_component
        + 0.10 * exp_component
    )
    return round(min(max(upset_risk, 0.0), 1.0), 4)


def compute_upset_risk_v3(
    elo_diff: float,
    underdog_form: float,
    favorite_form: float,
    neutral: int,
    underdog_mv: float = 0.0,
    favorite_mv: float = 0.0,
    underdog_exp: int = 0,
    favorite_exp: int = 0,
    underdog_coach_wr: float = 0.48,
    favorite_coach_wr: float = 0.48,
    favorite_coach_wc: int = 0,
    altitude_m: float = 200.0,
    travel_km_fav: float = 0.0,
    travel_km_dog: float = 0.0,
    temp_celsius: float = 22.0,
) -> float:
    """
    v3: v2 + antrenör farkı + irtifa yorgunluğu + seyahat mesafesi + sıcaklık.

    - Favori uzak seyahat ediyorsa → daha fazla sürpriz riski
    - Yüksek irtifa (>1500m) → küçük takım avantajı (konvansiyonel güç daha yorulur)
    - Sıcak/nemli hava (>28°C) → büyük takımların hazır beklenen kondisyon avantajı azalır
    - Deneyimli underdog antrenörü vs deneyimsiz favori antrenörü → daha fazla sürpriz
    """
    base = compute_upset_risk_v2(
        elo_diff, underdog_form, favorite_form, neutral,
        underdog_mv, favorite_mv, underdog_exp, favorite_exp,
    )

    # Antrenör deneyim bonusu
    coach_edge = (underdog_coach_wr - favorite_coach_wr) * 0.5
    wc_exp_fav = min(favorite_coach_wc / 3.0, 1.0)  # 3+ WC = tam puan
    coach_component = max(0.0, coach_edge) - wc_exp_fav * 0.05  # deneyimli favori biraz avantaj

    # İrtifa bileşeni (>1500m = önemli etki)
    alt_factor = min(1.0, max(0.0, (altitude_m - 500) / 2000.0))
    altitude_component = alt_factor * 0.1  # irtifa büyük takımları biraz daha zorlar

    # Seyahat yorgunluğu (favorinin yolu daha uzunsa)
    travel_diff = travel_km_fav - travel_km_dog
    travel_component = min(0.08, max(-0.03, travel_diff / 30000.0))

    # Sıcak hava yorgunluğu (>28°C önemli)
    heat_factor = max(0.0, (temp_celsius - 20) / 20.0) * 0.04

    adjustment = coach_component + altitude_component + travel_component + heat_factor
    result = base + adjustment
    return round(min(max(result, 0.0), 1.0), 4)


# ── Tarihi maç feature matrisi ────────────────────────────────────────────────
print("\nTarihi feature matrisi oluşturuluyor...")

# results_clean is already filtered to 2000+
results_modern = results_clean.reset_index(drop=True)
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

    form_home   = get_rolling_form(home, date)
    form_away   = get_rolling_form(away, date)
    form_home_5 = get_rolling_form(home, date, n=5)
    form_away_5 = get_rolling_form(away, date, n=5)

    form_diff = (
        form_home["weighted_form"] - form_away["weighted_form"]
        if (not np.isnan(form_home["weighted_form"]) and not np.isnan(form_away["weighted_form"]))
        else np.nan
    )

    ad_home = get_attack_defense(home, date)
    ad_away = get_attack_defense(away, date)

    h2h_diff   = get_h2h_goal_diff(home, away, date)
    co_diff    = get_common_opponent_diff(home, away, date)

    hs, as_ = int(row["home_score"]), int(row["away_score"])
    if hs > as_:    result_label = "H"
    elif hs == as_: result_label = "D"
    else:           result_label = "A"

    feature_rows.append({
        "date":                      date,
        "home_team":                 home,
        "away_team":                 away,
        "elo_home":                  elo_home,
        "elo_away":                  elo_away,
        "elo_diff":                  elo_diff,
        "form_home_pts":             form_home["points_last_n"],
        "form_away_pts":             form_away["points_last_n"],
        "form_diff":                 form_diff,
        "weighted_form_home":        form_home["weighted_form"],
        "weighted_form_away":        form_away["weighted_form"],
        "gf_home_last_n":            form_home["gf_last_n"],
        "ga_home_last_n":            form_home["ga_last_n"],
        "gf_away_last_n":            form_away["gf_last_n"],
        "ga_away_last_n":            form_away["ga_last_n"],
        "attack_home":               ad_home["attack_strength"],
        "defense_home":              ad_home["defense_weakness"],
        "attack_away":               ad_away["attack_strength"],
        "defense_away":              ad_away["defense_weakness"],
        "neutral":                   int(row.get("neutral", 0)),
        "tournament_weight":         get_tournament_weight(row.get("tournament", "")),
        # ── Yeni feature'lar ──
        "points_last_5_home":        form_home_5["points_last_n"],
        "points_last_5_away":        form_away_5["points_last_n"],
        "goal_diff_last_5_home":     form_home_5["goal_diff_last_n"],
        "goal_diff_last_5_away":     form_away_5["goal_diff_last_n"],
        "goals_for_last_5_home":     form_home_5["gf_last_n"],
        "goals_for_last_5_away":     form_away_5["gf_last_n"],
        "goals_against_last_5_home": form_home_5["ga_last_n"],
        "goals_against_last_5_away": form_away_5["ga_last_n"],
        "win_streak_home":           form_home["win_streak"],
        "win_streak_away":           form_away["win_streak"],
        "loss_streak_home":          form_home["loss_streak"],
        "loss_streak_away":          form_away["loss_streak"],
        "clean_sheet_rate_home":     form_home["clean_sheet_rate"],
        "clean_sheet_rate_away":     form_away["clean_sheet_rate"],
        "failed_to_score_rate_home": form_home["failed_to_score_rate"],
        "failed_to_score_rate_away": form_away["failed_to_score_rate"],
        "h2h_goal_diff":             h2h_diff,
        "common_opponent_diff":      co_diff,
        # ── Deneyim & kadro istatistikleri ──
        "experience_score_home":     get_experience_score(home, date),
        "experience_score_away":     get_experience_score(away, date),
        "avg_age_home":              get_squad_feat(home, "avg_age"),
        "avg_age_away":              get_squad_feat(away, "avg_age"),
        "market_value_proxy_home":   get_squad_feat(home, "market_value_proxy"),
        "market_value_proxy_away":   get_squad_feat(away, "market_value_proxy"),
        "top5_league_count_home":    get_squad_feat(home, "top5_league_count"),
        "top5_league_count_away":    get_squad_feat(away, "top5_league_count"),
        # ── Yeni feature'lar (43 toplam) ──
        "goal_trend_home":           get_goal_trend(home, date),
        "goal_trend_away":           get_goal_trend(away, date),
        "form_consistency_home":     get_form_consistency(home, date),
        "form_consistency_away":     get_form_consistency(away, date),
        "attack_ratio_home":         round(ad_home["attack_strength"] / (ad_away["defense_weakness"] + 0.1), 4),
        "attack_ratio_away":         round(ad_away["attack_strength"] / (ad_home["defense_weakness"] + 0.1), 4),
        "elo_form_interaction":      round((elo_diff / 400.0) * (form_diff or 0.0), 4) if not np.isnan(elo_diff or 0.0) else 0.0,
        # ── Yeni feature'lar (55 toplam) — tarihi için varsayılan/NaN ──
        "goals_per90_home":          get_squad_feat(home, "goals_per90"),
        "goals_per90_away":          get_squad_feat(away, "goals_per90"),
        "assists_per90_home":        get_squad_feat(home, "assists_per90"),
        "assists_per90_away":        get_squad_feat(away, "assists_per90"),
        "coach_win_rate_home":       get_coach_feat(home, "win_rate"),
        "coach_win_rate_away":       get_coach_feat(away, "win_rate"),
        "coach_wc_apps_home":        get_coach_feat(home, "wc_apps"),
        "coach_wc_apps_away":        get_coach_feat(away, "wc_apps"),
        "altitude_m":                200.0,    # tarihi maçlarda statdyum verisi yok
        "travel_km_diff":            0.0,      # tarihi maçlarda seyahat verisi yok
        "temp_celsius":              15.0,     # tarihi maçlarda hava verisi yok
        # ── Hedef ──
        "home_score":                hs,
        "away_score":                as_,
        "result":                    result_label,
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
    vfeat = get_venue_features(str(row.get("venue", "")))

    elo_home = fast_elo(home, date)
    elo_away = fast_elo(away, date)
    elo_diff = elo_home - elo_away if (not np.isnan(elo_home) and not np.isnan(elo_away)) else np.nan

    form_home   = get_rolling_form(home, date)
    form_away   = get_rolling_form(away, date)
    form_home_5 = get_rolling_form(home, date, n=5)
    form_away_5 = get_rolling_form(away, date, n=5)

    form_diff = (
        form_home["weighted_form"] - form_away["weighted_form"]
        if (not np.isnan(form_home["weighted_form"]) and not np.isnan(form_away["weighted_form"]))
        else np.nan
    )

    ad_home = get_attack_defense(home, date)
    ad_away = get_attack_defense(away, date)

    h2h_diff = get_h2h_goal_diff(home, away, date)
    co_diff  = get_common_opponent_diff(home, away, date)

    future_rows.append({
        "match_id":                  row["match_id"],
        "group":                     row["group"],
        "date_utc":                  date,
        "venue":                     row["venue"],
        "home_team":                 home,
        "away_team":                 away,
        "elo_home":                  elo_home,
        "elo_away":                  elo_away,
        "elo_diff":                  elo_diff,
        "form_home_pts":             form_home["points_last_n"],
        "form_away_pts":             form_away["points_last_n"],
        "form_diff":                 form_diff,
        "weighted_form_home":        form_home["weighted_form"],
        "weighted_form_away":        form_away["weighted_form"],
        "gf_home_last_n":            form_home["gf_last_n"],
        "ga_home_last_n":            form_home["ga_last_n"],
        "gf_away_last_n":            form_away["gf_last_n"],
        "ga_away_last_n":            form_away["ga_last_n"],
        "attack_home":               ad_home["attack_strength"],
        "defense_home":              ad_home["defense_weakness"],
        "attack_away":               ad_away["attack_strength"],
        "defense_away":              ad_away["defense_weakness"],
        "neutral":                   1,
        "tournament_weight":         4.0,
        # ── Yeni feature'lar ──
        "points_last_5_home":        form_home_5["points_last_n"],
        "points_last_5_away":        form_away_5["points_last_n"],
        "goal_diff_last_5_home":     form_home_5["goal_diff_last_n"],
        "goal_diff_last_5_away":     form_away_5["goal_diff_last_n"],
        "goals_for_last_5_home":     form_home_5["gf_last_n"],
        "goals_for_last_5_away":     form_away_5["gf_last_n"],
        "goals_against_last_5_home": form_home_5["ga_last_n"],
        "goals_against_last_5_away": form_away_5["ga_last_n"],
        "win_streak_home":           form_home["win_streak"],
        "win_streak_away":           form_away["win_streak"],
        "loss_streak_home":          form_home["loss_streak"],
        "loss_streak_away":          form_away["loss_streak"],
        "clean_sheet_rate_home":     form_home["clean_sheet_rate"],
        "clean_sheet_rate_away":     form_away["clean_sheet_rate"],
        "failed_to_score_rate_home": form_home["failed_to_score_rate"],
        "failed_to_score_rate_away": form_away["failed_to_score_rate"],
        "h2h_goal_diff":             h2h_diff,
        "common_opponent_diff":      co_diff,
        # ── Deneyim & kadro istatistikleri ──
        "experience_score_home":     get_experience_score(home, date),
        "experience_score_away":     get_experience_score(away, date),
        "avg_age_home":              get_squad_feat(home, "avg_age"),
        "avg_age_away":              get_squad_feat(away, "avg_age"),
        "market_value_proxy_home":   get_squad_feat(home, "market_value_proxy"),
        "market_value_proxy_away":   get_squad_feat(away, "market_value_proxy"),
        "top5_league_count_home":    get_squad_feat(home, "top5_league_count"),
        "top5_league_count_away":    get_squad_feat(away, "top5_league_count"),
        # ── Yeni feature'lar (55 toplam) ──
        "goal_trend_home":           get_goal_trend(home, date),
        "goal_trend_away":           get_goal_trend(away, date),
        "form_consistency_home":     get_form_consistency(home, date),
        "form_consistency_away":     get_form_consistency(away, date),
        "attack_ratio_home":         round(ad_home["attack_strength"] / (ad_away["defense_weakness"] + 0.1), 4),
        "attack_ratio_away":         round(ad_away["attack_strength"] / (ad_home["defense_weakness"] + 0.1), 4),
        "elo_form_interaction":      round((elo_diff / 400.0) * (form_diff or 0.0), 4) if not np.isnan(elo_diff or 0.0) else 0.0,
        # ── Yeni (squad + coach + venue + seyahat) ──
        "goals_per90_home":          get_squad_feat(home, "goals_per90"),
        "goals_per90_away":          get_squad_feat(away, "goals_per90"),
        "assists_per90_home":        get_squad_feat(home, "assists_per90"),
        "assists_per90_away":        get_squad_feat(away, "assists_per90"),
        "coach_win_rate_home":       get_coach_feat(home, "win_rate"),
        "coach_win_rate_away":       get_coach_feat(away, "win_rate"),
        "coach_wc_apps_home":        get_coach_feat(home, "wc_apps"),
        "coach_wc_apps_away":        get_coach_feat(away, "wc_apps"),
        "altitude_m":                vfeat["altitude_m"],
        "travel_km_diff":            round(get_travel_km(away, vfeat["venue_lat"], vfeat["venue_lon"])
                                     - get_travel_km(home, vfeat["venue_lat"], vfeat["venue_lon"])),
        "temp_celsius":              vfeat["temp_celsius"],
    })

future_df = pd.DataFrame(future_rows)

# Upset risk v3 ekle (en kapsamlı formül)
def _upset_v3_row(r):
    is_home_fav = (r.get("elo_diff", 0) or 0) > 0
    fav_team  = r["home_team"] if is_home_fav else r["away_team"]
    dog_team  = r["away_team"] if is_home_fav else r["home_team"]
    fav_form  = r["weighted_form_home"] if is_home_fav else r["weighted_form_away"]
    dog_form  = r["weighted_form_away"] if is_home_fav else r["weighted_form_home"]
    vl = get_venue_features(r.get("venue", ""))
    return compute_upset_risk_v3(
        r["elo_diff"],
        dog_form, fav_form, int(r["neutral"]),
        underdog_mv=float(min(r.get("market_value_proxy_home", 0) or 0, r.get("market_value_proxy_away", 0) or 0)),
        favorite_mv=float(max(r.get("market_value_proxy_home", 0) or 0, r.get("market_value_proxy_away", 0) or 0)),
        underdog_exp=int(min(r.get("experience_score_home", 0) or 0, r.get("experience_score_away", 0) or 0)),
        favorite_exp=int(max(r.get("experience_score_home", 0) or 0, r.get("experience_score_away", 0) or 0)),
        underdog_coach_wr=get_coach_feat(dog_team, "win_rate"),
        favorite_coach_wr=get_coach_feat(fav_team, "win_rate"),
        favorite_coach_wc=int(get_coach_feat(fav_team, "wc_apps")),
        altitude_m=float(r.get("altitude_m", 200) or 200),
        travel_km_fav=get_travel_km(fav_team, vl["venue_lat"], vl["venue_lon"]),
        travel_km_dog=get_travel_km(dog_team, vl["venue_lat"], vl["venue_lon"]),
        temp_celsius=float(r.get("temp_celsius", 22) or 22),
    )

# Venue features'ı future_rows loop'unda hesaplaması için vfeat eklendi;
# burada future_df üzerinden özet işlem yapıyoruz
future_df["upset_risk"] = future_df.apply(_upset_v3_row, axis=1)
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

print(f"\nYeni feature sütunları ({len(features_df.columns)} toplam):")
new_cols = [
    "points_last_5_home", "goal_diff_last_5_home", "win_streak_home",
    "loss_streak_home", "clean_sheet_rate_home", "failed_to_score_rate_home",
    "h2h_goal_diff", "common_opponent_diff",
    "experience_score_home", "avg_age_home", "market_value_proxy_home",
    "top5_league_count_home",
    "goal_trend_home", "form_consistency_home", "attack_ratio_home", "elo_form_interaction",
    "goals_per90_home", "assists_per90_home",
    "coach_win_rate_home", "coach_wc_apps_home",
    "altitude_m", "travel_km_diff", "temp_celsius",
]
for c in new_cols:
    if c in features_df.columns:
        pct_nan = features_df[c].isna().mean() * 100
        print(f"  {c:<35}: {pct_nan:.1f}% NaN")

print(f"\n2026 fikstür yeni sütunlar ({len(future_df.columns)} toplam):")
for c in ["altitude_m", "travel_km_diff", "temp_celsius",
          "goals_per90_home", "goals_per90_away",
          "coach_win_rate_home", "coach_win_rate_away"]:
    if c in future_df.columns:
        sample = future_df[c].describe()
        print(f"  {c:<30}: mean={sample['mean']:.2f}, min={sample['min']:.1f}, max={sample['max']:.1f}")
