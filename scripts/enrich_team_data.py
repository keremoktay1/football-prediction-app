"""
enrich_team_data.py

WC 2026 takımları için tek seferlik veri zenginleştirme scripti.

Çıktılar:
  data/processed/squad_stats.csv   — oyuncu/kadro istatistikleri (32 takım)
  data/processed/team_clusters.csv — K-Means K=5 cluster atamaları
"""
from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR       = os.path.join(APP_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(APP_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── FIFA 3-char kodu → WC 2026 fikstür takım adı eşlemesi ────────────────────
FIFA3_TO_TEAM: dict[str, str] = {
    "USA": "USA",
    "ENG": "England",
    "FRA": "France",
    "ARG": "Argentina",
    "BRA": "Brazil",
    "ESP": "Spain",
    "GER": "Germany",
    "POR": "Portugal",
    "NED": "Netherlands",
    "BEL": "Belgium",
    "SUI": "Switzerland",
    "CRO": "Croatia",
    "URU": "Uruguay",
    "COL": "Colombia",
    "MEX": "Mexico",
    "CAN": "Canada",
    "AUS": "Australia",
    "JPN": "Japan",
    "KOR": "South Korea",
    "MAR": "Morocco",
    "SEN": "Senegal",
    "GHA": "Ghana",
    "EGY": "Egypt",
    "TUN": "Tunisia",
    "ALG": "Algeria",
    "QAT": "Qatar",
    "KSA": "Saudi Arabia",
    "IRN": "Iran",
    "NOR": "Norway",
    "AUT": "Austria",
    "ECU": "Ecuador",
    "PAR": "Paraguay",
    "PAN": "Panama",
    "HAI": "Haiti",
    "CIV": "Côte d'Ivoire",
    "CPV": "Cabo Verde",
    "SCO": "Scotland",
    "NZL": "New Zealand",
    "JOR": "Jordan",
    "UZB": "Uzbekistan",
    "CUW": "Curaçao",
    "RSA": "South Africa",
    # Eksik olan ama veri setinde bulunan ülkeler
    "TUR": "Turkey",        # Türkiye — 19 oyuncu var (Çalhanoğlu, Güler, Kadıoğlu vb.)
    "SVK": "Slovakia",
    "SRB": "Serbia",
    "SVN": "Slovenia",
    "HUN": "Hungary",
    "ALB": "Albania",
    "ROU": "Romania",
    "GEO": "Georgia",
    "MKD": "North Macedonia",
}

TOP5_LEAGUE_KEYWORDS = [
    "premier league",
    "la liga",
    "bundesliga",
    "serie a",
    "ligue 1",
]

def is_top5(comp_str: str) -> bool:
    if pd.isna(comp_str):
        return False
    low = str(comp_str).lower()
    return any(kw in low for kw in TOP5_LEAGUE_KEYWORDS)

def is_forward(pos_str: str) -> bool:
    if pd.isna(pos_str):
        return False
    return "FW" in str(pos_str).upper()


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 1 — Squad İstatistikleri
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 55)
print("BÖLÜM 1 — Squad İstatistikleri")
print("=" * 55)

players_path = os.path.join(RAW_DIR, "players_data-2025_2026.csv")
if not os.path.isfile(players_path):
    print(f"[WARN] players_data-2025_2026.csv bulunamadı: {players_path}")
    players = pd.DataFrame()
else:
    players = pd.read_csv(players_path, low_memory=False)
    print(f"[OK]  players_data yüklendi: {len(players):,} oyuncu, {players.shape[1]} sütun")

squad_rows = []

if not players.empty:
    # Nation sütunundan 3-char FIFA kodu çıkar: "us USA" → "USA"
    def parse_nation(n):
        if pd.isna(n):
            return None
        parts = str(n).strip().split()
        return parts[-1] if parts else None

    players["nation_code"] = players["Nation"].apply(parse_nation)
    players["team_name"]   = players["nation_code"].map(FIFA3_TO_TEAM)

    # WC 2026 takımlarına filtrele
    wc_players = players[players["team_name"].notna()].copy()
    print(f"  WC 2026 oyuncuları: {len(wc_players):,}")

    # Numerik dönüşümler
    for col in ["Age", "90s", "Gls", "Ast"]:
        wc_players[col] = pd.to_numeric(wc_players[col], errors="coerce")

    wc_players["is_top5"]   = wc_players["Comp"].apply(is_top5)
    wc_players["is_forward"] = wc_players["Pos"].apply(is_forward)

    for team, grp in wc_players.groupby("team_name"):
        n = len(grp)
        avg_age = float(grp["Age"].mean()) if n > 0 else np.nan

        total_90s = float(grp["90s"].sum())
        total_gls = float(grp["Gls"].sum())
        total_ast = float(grp["Ast"].sum())

        goals_per90    = round(total_gls / total_90s, 4) if total_90s > 0 else 0.0
        assists_per90  = round(total_ast / total_90s, 4) if total_90s > 0 else 0.0
        top5_count     = int(grp["is_top5"].sum())

        fwd = grp[grp["is_forward"]]
        fwd_90s = float(fwd["90s"].sum())
        fwd_gls = float(fwd["Gls"].sum())
        forward_goals_p90 = round(fwd_gls / fwd_90s, 4) if fwd_90s > 0 else 0.0

        # Market value proxy: top5 * 2 + goals_per90 * 10
        market_value_proxy = round(top5_count * 2 + goals_per90 * 10, 2)

        squad_rows.append({
            "team":               team,
            "squad_size":         n,
            "avg_age":            round(avg_age, 2) if not np.isnan(avg_age) else 26.0,
            "goals_per90":        goals_per90,
            "assists_per90":      assists_per90,
            "top5_league_count":  top5_count,
            "forward_goals_p90":  forward_goals_p90,
            "market_value_proxy": market_value_proxy,
        })

squad_df = pd.DataFrame(squad_rows).sort_values("team").reset_index(drop=True)
squad_path = os.path.join(PROCESSED_DIR, "squad_stats.csv")
squad_df.to_csv(squad_path, index=False)
print(f"\n✅  squad_stats.csv → {len(squad_df)} takım")
print(squad_df[["team", "squad_size", "avg_age", "top5_league_count", "market_value_proxy"]].to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 2 — Takım Clustering (K-Means K=5)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("BÖLÜM 2 — Takım Clustering")
print("=" * 55)

features_path = os.path.join(PROCESSED_DIR, "features_2026_fixtures.csv")
if not os.path.isfile(features_path):
    print(f"[WARN] features_2026_fixtures.csv bulunamadı: {features_path}")
    print("  Önce fast_feature_engineering.py çalıştırın.")
else:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    fut = pd.read_csv(features_path)

    # Her takım için ev sahibi ve deplasman maçlarından ortalama hesapla
    team_feats = []
    all_teams = set(fut["home_team"].dropna()) | set(fut["away_team"].dropna())

    for team in sorted(all_teams):
        home_rows = fut[fut["home_team"] == team]
        away_rows = fut[fut["away_team"] == team]

        elo_vals    = pd.concat([home_rows["elo_home"],  away_rows["elo_away"]],  ignore_index=True)
        atk_vals    = pd.concat([home_rows["attack_home"], away_rows["attack_away"]], ignore_index=True)
        def_vals    = pd.concat([home_rows["defense_home"], away_rows["defense_away"]], ignore_index=True)
        form_vals   = pd.concat([home_rows["weighted_form_home"], away_rows["weighted_form_away"]], ignore_index=True)

        elo_rating      = float(elo_vals.mean()) if len(elo_vals) else np.nan
        attack_strength = float(atk_vals.mean()) if len(atk_vals) else np.nan
        defense_weakness = float(def_vals.mean()) if len(def_vals) else np.nan
        weighted_form   = float(form_vals.mean()) if len(form_vals) else np.nan

        if not np.isnan(elo_rating):
            team_feats.append({
                "team":             team,
                "elo_rating":       round(elo_rating, 1),
                "attack_strength":  round(attack_strength, 4) if not np.isnan(attack_strength) else 1.0,
                "defense_weakness": round(defense_weakness, 4) if not np.isnan(defense_weakness) else 1.0,
                "weighted_form":    round(weighted_form, 4) if not np.isnan(weighted_form) else 0.0,
            })

    tf_df = pd.DataFrame(team_feats)
    print(f"  Kümeleme için {len(tf_df)} takım hazır")

    # Playoff slot'larını çıkar (WC 2026 gerçek takımları)
    real_teams = tf_df[~tf_df["team"].str.contains("Playoff|TBD|Winner|Runner|Best", na=False)].copy()
    print(f"  Gerçek takım sayısı: {len(real_teams)}")

    cluster_features = ["elo_rating", "attack_strength", "defense_weakness", "weighted_form"]
    X = real_teams[cluster_features].fillna(real_teams[cluster_features].mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    K = 5
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=20)
    real_teams = real_teams.copy()
    real_teams["cluster_id"] = kmeans.fit_predict(X_scaled)

    # Centroid ELO sırasına göre etiket ata (yüksek ELO → Elite)
    centroids = kmeans.cluster_centers_
    elo_idx = cluster_features.index("elo_rating")
    centroid_elos = [(i, scaler.mean_[elo_idx] + centroids[i, elo_idx] * scaler.scale_[elo_idx])
                     for i in range(K)]
    centroid_elos_sorted = sorted(centroid_elos, key=lambda x: -x[1])  # büyükten küçüğe

    CLUSTER_LABELS = [
        "Elite Favorites",
        "Strong Contenders",
        "Solid Teams",
        "Dark Horses",
        "Underdogs",
    ]

    cluster_label_map = {
        centroid_elos_sorted[i][0]: CLUSTER_LABELS[i]
        for i in range(K)
    }
    real_teams["cluster_label"] = real_teams["cluster_id"].map(cluster_label_map)

    cluster_out = real_teams[["team", "cluster_id", "cluster_label",
                               "elo_rating", "attack_strength", "defense_weakness"]].copy()
    cluster_path = os.path.join(PROCESSED_DIR, "team_clusters.csv")
    cluster_out.to_csv(cluster_path, index=False)
    print(f"\n✅  team_clusters.csv → {len(cluster_out)} takım")

    for label in CLUSTER_LABELS:
        teams_in = cluster_out[cluster_out["cluster_label"] == label]["team"].tolist()
        print(f"  {label}: {', '.join(sorted(teams_in))}")

print("\nTamamlandı ✅")
