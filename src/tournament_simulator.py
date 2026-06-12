"""
tournament_simulator.py — Monte Carlo turnuva simülasyonu.

Akış:
  1. Oynanmamış grup maçlarını olasılıklarla simüle et
  2. Grup sıralamalarını hesapla (gerçek sonuçlar sabit tutulur)
  3. 32 takımı belirle (12 birinci + 12 ikinci + 8 en iyi üçüncü)
  4. Eleme turunu Elo tabanlı simüle et
  5. N simülasyon boyunca P(şampiyon), P(finalist) vb. hesapla

Sonuçlar data/processed/simulation_latest.csv'ye kaydedilir.
"""
from __future__ import annotations

import re
import os
import sys
import json
import math
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROCESSED_DIR
from prediction_engine import _elo_win_prob, UNKNOWN_ELO

SIMULATION_PATH = os.path.join(PROCESSED_DIR, "simulation_latest.csv")
SIM_META_PATH   = os.path.join(PROCESSED_DIR, "simulation_meta.json")

# Eleme turu başlangıç match_id
KO_START_ID = 73

# Turnuva aşama seviyeleri (yüksek = daha iyi)
ROUND_LEVEL = {
    "groups":    0,
    "round32":   1,
    "round16":   2,
    "quarter":   3,
    "semi":      4,
    "fourth":    4,
    "third":     5,
    "finalist":  6,
    "champion":  7,
}

# ── İyileştirme 2: Ev sahibi ülke kalabalık etkisi ────────────────────────────
HOST_BOOSTS: Dict[str, int] = {
    "United States": 40,
    "Mexico":        50,
    "Canada":        35,
}

VENUE_HOST: Dict[str, str] = {
    # ABD stadları
    "AT&T Stadium":           "United States",
    "MetLife Stadium":        "United States",
    "Rose Bowl":              "United States",
    "Levi's Stadium":         "United States",
    "Gillette Stadium":       "United States",
    "SoFi Stadium":           "United States",
    "Arrowhead Stadium":      "United States",
    "NRG Stadium":            "United States",
    "Lincoln Financial Field":"United States",
    "Lumen Field":            "United States",
    # Meksika stadları
    "Estadio Azteca":         "Mexico",
    "Estadio Akron":          "Mexico",
    "Estadio BBVA":           "Mexico",
    # Kanada stadları
    "BC Place":               "Canada",
    "BMO Field":              "Canada",
}

# ── İyileştirme 5: İrtifa adaptasyonu ────────────────────────────────────────
HIGH_ALT_TEAMS: Dict[str, int] = {
    "Mexico":        2240,
    "Colombia":      2600,
    "Peru":           500,
    "Ecuador":       2850,
    "Bolivia":       3640,
    "United States":  600,  # Denver etkisi — ortalama
}

# ── Venue tam bilgileri (wc2026_venues.csv + Hard Rock eki) ───────────────────
# Her stadyum: lat, lon, avg_june_temp_c, altitude_m
VENUE_INFO: Dict[str, dict] = {
    "Estadio Azteca":          {"lat": 19.3026, "lon": -99.1508,   "temp": 15, "alt": 2240},
    "Estadio Akron":           {"lat": 20.6849, "lon": -103.4666,  "temp": 18, "alt": 1566},
    "Estadio BBVA":            {"lat": 25.6694, "lon": -100.3093,  "temp": 28, "alt": 538},
    "MetLife Stadium":         {"lat": 40.8128, "lon": -74.0742,   "temp": 24, "alt": 4},
    "AT&T Stadium":            {"lat": 32.7480, "lon": -97.0928,   "temp": 32, "alt": 167},
    "SoFi Stadium":            {"lat": 33.9534, "lon": -118.3391,  "temp": 22, "alt": 37},
    "NRG Stadium":             {"lat": 29.6847, "lon": -95.4107,   "temp": 32, "alt": 15},
    "Arrowhead Stadium":       {"lat": 39.0489, "lon": -94.4839,   "temp": 27, "alt": 274},
    "Levi's Stadium":          {"lat": 37.4033, "lon": -121.9694,  "temp": 22, "alt": 7},
    "Lincoln Financial Field": {"lat": 39.9008, "lon": -75.1675,   "temp": 27, "alt": 5},
    "Gillette Stadium":        {"lat": 42.0909, "lon": -71.2643,   "temp": 23, "alt": 50},
    "Rose Bowl":               {"lat": 34.1613, "lon": -118.1676,  "temp": 27, "alt": 274},
    "Lumen Field":             {"lat": 47.5952, "lon": -122.3316,  "temp": 17, "alt": 4},
    "Empower Field":           {"lat": 39.7439, "lon": -105.0201,  "temp": 21, "alt": 1600},
    "BC Place":                {"lat": 49.2768, "lon": -123.1118,  "temp": 17, "alt": 5},
    "BMO Field":               {"lat": 43.6333, "lon": -79.4187,   "temp": 22, "alt": 75},
    "Stade Olympique":         {"lat": 45.5629, "lon": -73.5514,   "temp": 23, "alt": 57},
    "Hard Rock Stadium":       {"lat": 25.9579, "lon": -80.2389,   "temp": 32, "alt": 2},
}

# Geriye dönük uyumluluk için (eski kod VENUE_ALTITUDE kullanıyor)
VENUE_ALTITUDE: Dict[str, int] = {k: v["alt"] for k, v in VENUE_INFO.items()}

# ── Takım ana şehir koordinatları (başkent / büyük şehir) ────────────────────
TEAM_COORDS: Dict[str, Tuple[float, float]] = {
    # CONCACAF
    "United States":          (38.9, -77.0),
    "Mexico":                 (19.4, -99.1),
    "Canada":                 (45.4, -75.7),
    "Costa Rica":             (9.9,  -84.1),
    "Jamaica":                (18.0, -76.8),
    "Honduras":               (14.1, -87.2),
    "El Salvador":            (13.7, -89.2),
    "Panama":                 (9.0,  -79.5),
    "Trinidad & Tobago":      (10.7, -61.5),
    "Haiti":                  (18.5, -72.3),
    # CONMEBOL
    "Argentina":              (-34.6, -58.4),
    "Brazil":                 (-15.8, -47.9),
    "Colombia":               (4.7,  -74.1),
    "Ecuador":                (-0.2, -78.5),
    "Uruguay":                (-34.9, -56.2),
    "Chile":                  (-33.4, -70.7),
    "Peru":                   (-12.0, -77.0),
    "Paraguay":               (-25.3, -57.6),
    "Venezuela":              (10.5, -66.9),
    "Bolivia":                (-16.5, -68.1),
    # UEFA
    "France":                 (48.8,   2.3),
    "Germany":                (52.5,  13.4),
    "Spain":                  (40.4,  -3.7),
    "England":                (51.5,  -0.1),
    "Portugal":               (38.7,  -9.1),
    "Netherlands":            (52.4,   4.9),
    "Belgium":                (50.8,   4.4),
    "Italy":                  (41.9,  12.5),
    "Croatia":                (45.8,  16.0),
    "Denmark":                (55.7,  12.6),
    "Austria":                (48.2,  16.4),
    "Switzerland":            (47.4,   8.5),
    "Serbia":                 (44.8,  20.5),
    "Ukraine":                (50.4,  30.5),
    "Poland":                 (52.2,  21.0),
    "Türkiye":                (39.9,  32.9),
    "Turkey":                 (39.9,  32.9),
    "Hungary":                (47.5,  19.1),
    "Slovakia":               (48.1,  17.1),
    "Slovenia":               (46.1,  14.5),
    "Albania":                (41.3,  19.8),
    "Scotland":               (55.9,  -3.2),
    "Romania":                (44.4,  26.1),
    "Czechia":                (50.1,  14.4),
    "Czech Republic":         (50.1,  14.4),
    "Wales":                  (51.5,  -3.2),
    "Georgia":                (41.7,  44.8),
    "North Macedonia":        (42.0,  21.4),
    "Bosnia and Herzegovina": (43.8,  18.4),
    "Kosovo":                 (42.7,  21.2),
    "Montenegro":             (42.4,  19.3),
    "Finland":                (60.2,  24.9),
    "Norway":                 (59.9,  10.7),
    "Sweden":                 (59.3,  18.1),
    "Ireland":                (53.3,  -6.3),
    "Northern Ireland":       (54.6,  -5.9),
    "Greece":                 (37.9,  23.7),
    # AFC
    "Japan":                  (35.7,  139.7),
    "South Korea":            (37.6,  127.0),
    "Iran":                   (35.7,   51.4),
    "Saudi Arabia":           (24.7,   46.7),
    "Australia":              (-25.3, 133.8),
    "Qatar":                  (25.3,   51.5),
    "Iraq":                   (33.3,   44.4),
    "Jordan":                 (31.9,   35.9),
    "Uzbekistan":             (41.3,   69.2),
    "China PR":               (39.9,  116.4),
    "China":                  (39.9,  116.4),
    "Indonesia":              (-6.2,  106.8),
    "Bahrain":                (26.2,   50.6),
    "United Arab Emirates":   (24.5,   54.4),
    # CAF
    "Morocco":                (34.0,  -6.8),
    "Senegal":                (14.7, -17.4),
    "Nigeria":                (9.1,    7.4),
    "Cameroon":               (3.9,   11.5),
    "Ghana":                  (5.6,   -0.2),
    "Egypt":                  (30.1,  31.2),
    "Côte d'Ivoire":          (5.4,   -4.0),
    "Mali":                   (12.7,  -8.0),
    "Algeria":                (36.7,   3.2),
    "Tunisia":                (36.8,  10.2),
    "South Africa":           (-26.2, 28.0),
    "Tanzania":               (-6.8,  39.3),
    "Comoros":                (-11.6, 43.3),
    "Cabo Verde":             (14.9, -23.5),
    # OFC
    "New Zealand":            (-36.9, 174.8),
}

# ── WC 2026 irtifa Elo deflasyonu ────────────────────────────────────────────
# Bu takımların Elo'su tarihsel irtifa ev sahibi avantajıyla şişmiş durumda.
# WC 2026 Kuzey Amerika'da (düşük rakım) oynandığı için bu avantaj geçersiz.
ALTITUDE_ELO_DEFLATION: Dict[str, int] = {
    "Colombia": -40,  # Bogotá 2600m — maçların büyüğü deniz seviyesinde
    "Ecuador":  -35,  # Quito 2850m
    "Bolivia":  -50,  # La Paz 3640m (WC'de değil ama yedek)
}

# ── Serin iklimli takımlar: sıcakta Elo cezası alır ──────────────────────────
COOL_CLIMATE_TEAMS = {
    # Kuzey Avrupa
    "Finland", "Norway", "Sweden", "Denmark", "Scotland", "Ireland",
    "Northern Ireland", "Wales", "England",
    # Orta/Batı Avrupa
    "Germany", "Netherlands", "Belgium", "Poland", "Czechia", "Czech Republic",
    "Slovakia", "Slovenia", "Hungary", "Austria", "Switzerland",
    "Croatia", "Serbia", "Ukraine", "Romania", "Bosnia and Herzegovina",
    # Kuzey Amerika (kuzey)
    "Canada",
    # Doğu Asya
    "Japan", "South Korea",
    # Güney Yarıküre (kış mevsimi etkisi)
    "Argentina", "Uruguay", "Chile", "Australia", "New Zealand",
}


# ──────────────────────────────────────────────
# Yardımcı fonksiyonlar
# ──────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """İki koordinat arasındaki haversine mesafesini km cinsinden döner."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(min(a, 1.0)))


def _team_travel_km(team: str, venue_lat: float, venue_lon: float) -> float:
    """Takımın ana şehrinden stadyuma olan uçuş mesafesi (km)."""
    coords = TEAM_COORDS.get(team)
    if coords is None:
        return 8000.0  # bilinmeyenler için ortalama
    return _haversine_km(coords[0], coords[1], venue_lat, venue_lon)


def _ko_elo_win_prob(elo_a: float, elo_b: float) -> float:
    """Eleme turları için Elo kazanma olasılığı.

    Denominator 500 kullanılır (standart 400'ün yerine) — futbol, satrançtan
    çok daha sürpriz yoğun; geniş ölçek bu belirsizliği yansıtır.

    Karşılaştırma (home=0 avantajı):
      100 Elo fark  →  ~61% (400 ile ~64%)
      200 Elo fark  →  ~72% (400 ile ~76%)
      300 Elo fark  →  ~80% (400 ile ~85%)
    """
    return 1.0 / (1.0 + 10.0 ** (-(elo_a - elo_b) / 500.0))


# ──────────────────────────────────────────────
# Yardımcı: grup sıralama (hızlı Python, pandas yok)
# ──────────────────────────────────────────────

def _fast_standings(
    group_matches: List[dict],
    sim_results: Dict[int, Tuple[int, int]],
    elo_map: Dict[str, float],
) -> Dict[str, List[Tuple[str, int, int, int]]]:
    """
    Grup sıralamalarını dict olarak döner.
    group → [(team, pts, gd, gf), ...] sıralı (en iyi önce)
    Tie-break: Pts > GD > GF > Elo
    """
    records: Dict[str, Dict[str, list]] = {}
    # records[group][team] = [pts, gd, gf]

    for m in group_matches:
        g   = m["group"]
        mid = m["match_id"]
        ht  = m["home"]
        at  = m["away"]
        if g not in records:
            records[g] = {}
        for t in (ht, at):
            if t not in records[g]:
                records[g][t] = [0, 0, 0]  # pts, gd, gf

        hs, as_ = sim_results.get(mid, (0, 0))
        if hs > as_:
            records[g][ht][0] += 3
        elif hs == as_:
            records[g][ht][0] += 1
            records[g][at][0] += 1
        else:
            records[g][at][0] += 3

        records[g][ht][1] += hs - as_
        records[g][ht][2] += hs
        records[g][at][1] += as_ - hs
        records[g][at][2] += as_

    standings: Dict[str, List[Tuple[str, int, int, int]]] = {}
    for g, teams in records.items():
        rows = [
            (t, d[0], d[1], d[2], elo_map.get(t, UNKNOWN_ELO))
            for t, d in teams.items()
        ]
        # Sort: pts desc, gd desc, gf desc, elo desc
        rows.sort(key=lambda x: (-x[1], -x[2], -x[3], -x[4]))
        standings[g] = [(r[0], r[1], r[2], r[3]) for r in rows]

    return standings


def _compute_rest_days(team: str, ko_date, last_played: Dict[str, str]) -> int:
    """Takımın son maçından bu yana geçen gün sayısını döner."""
    prev = last_played.get(team)
    if prev is None or ko_date is None:
        return 7
    try:
        return max(0, (ko_date - pd.Timestamp(prev)).days)
    except Exception:
        return 7


def _sample_goals(lh: float, la: float, outcome: int, rng) -> Tuple[int, int]:
    """
    Poisson'dan gol sayısı örnekle; outcome (0=H, 1=D, 2=A) tutarlı olsun.
    100 denemede tutarsızsa düz skor kullan.
    """
    for _ in range(100):
        hg = rng.poisson(lh)
        ag = rng.poisson(la)
        if outcome == 0 and hg > ag:
            return hg, ag
        if outcome == 1 and hg == ag:
            return hg, ag
        if outcome == 2 and ag > hg:
            return hg, ag
    # Fallback
    if outcome == 0:
        return (2, 0)
    if outcome == 1:
        return (1, 1)
    return (0, 2)


def _select_best_thirds(
    standings: Dict[str, List[Tuple[str, int, int, int]]]
) -> List[str]:
    """En iyi 8 üçüncü takımı döner (Pts > GD > GF sıralı)."""
    thirds = []
    for g, rows in standings.items():
        if len(rows) >= 3:
            t, pts, gd, gf = rows[2]
            thirds.append((t, pts, gd, gf))
    thirds.sort(key=lambda x: (-x[1], -x[2], -x[3]))
    return [t[0] for t in thirds[:8]]


def _resolve_ko_slot(
    slot: str,
    standings: Dict[str, List[Tuple[str, int, int, int]]],
    best_thirds: List[str],
    ko_winners: Dict[int, str],
    bt_cursor_ref: List[int],
) -> str:
    """Bir knockout slotunu gerçek takım adına çevirir."""
    # Winner/Runner-up Group X
    m = re.match(r"(Winner|Runner-up)\s+Group\s+([A-Z])", slot)
    if m:
        role, grp = m.group(1), m.group(2)
        rows = standings.get(grp, [])
        if role == "Winner" and rows:
            return rows[0][0]
        if role == "Runner-up" and len(rows) >= 2:
            return rows[1][0]
        return slot

    # Best 3rd
    if slot.startswith("Best 3rd"):
        idx = bt_cursor_ref[0]
        if idx < len(best_thirds):
            bt_cursor_ref[0] += 1
            return best_thirds[idx]
        return slot

    # Winner/Loser Match XX
    m2 = re.match(r"(Winner|Loser)\s+Match\s+(\d+)", slot)
    if m2:
        role = m2.group(1)
        prev_mid = int(m2.group(2))
        winner = ko_winners.get(prev_mid)
        if winner is None:
            return slot
        if role == "Winner":
            return winner
        # Loser: need to track both teams — stored as "loser_MID"
        loser_key = f"loser_{prev_mid}"
        return ko_winners.get(loser_key, slot)

    return slot


def _simulate_ko_match(
    home: str,
    away: str,
    elo_map: Dict[str, float],
    rng,
    wc_pts: Optional[Dict[str, int]] = None,
    venue: str = "",
    rest_days_home: int = 7,
    rest_days_away: int = 7,
    penalty_stats: Optional[Dict[str, float]] = None,
    venue_altitude: int = 50,
    temp_celsius: float = 22.0,
    travel_km_h: float = 8000.0,
    travel_km_a: float = 8000.0,
) -> Tuple[str, str]:
    """(winner, loser) döner — beraberlik yok.

    İyileştirmeler:
      0. Maç bazlı Elo gürültüsü      — her maç ~N(0,100) gürültü
      1. WC form boost                 — grup aşaması performansı
      2. Ev sahibi ülke kalabalık      — host ülke Elo bonusu
      3. Penaltı istatistikleri        — yakın maçlarda etkili
      4. Dinlenme / yorgunluk          — 4 günden az dinlenme cezası
      5. İrtifa adaptasyonu            — yüksek rakım dezavantajı
      6. Sıcaklık stresi               — serin iklimli takımlara sıcakta ceza
      7. Seyahat yorgunluğu            — uzun yolculuk Elo cezası
    """
    eh = elo_map.get(home, UNKNOWN_ELO)
    ea = elo_map.get(away, UNKNOWN_ELO)

    # ── 0. Maç Bazlı Elo Gürültüsü ───────────────────────────────────────────
    # σ=100: ~200 puan farkı olan maçlarda bile gerçekçi sürprizler mümkün
    eh += float(rng.normal(0, 125))
    ea += float(rng.normal(0, 125))

    # ── 1. WC Form Boost ──────────────────────────────────────────────────────
    if wc_pts is not None:
        pts_h = wc_pts.get(home, 4)
        pts_a = wc_pts.get(away, 4)
        wc_boost = float(np.clip((pts_h - pts_a) * 5, -30, 30))
        eh += wc_boost

    # ── 2. Ev Sahibi Ülke Crowd Effect ───────────────────────────────────────
    venue_country = VENUE_HOST.get(venue, "")
    if venue_country:
        if home == venue_country:
            eh += HOST_BOOSTS.get(venue_country, 0)
        elif away == venue_country:
            ea += HOST_BOOSTS.get(venue_country, 0)

    # ── 4. Dinlenme Günü / Yorgunluk ──────────────────────────────────────────
    fatigue_h = max(0, (4 - rest_days_home)) * 8   # 4 günden az = ceza, max 24 Elo
    fatigue_a = max(0, (4 - rest_days_away)) * 8
    eh -= fatigue_h
    ea -= fatigue_a

    # ── 5. İrtifa Adaptasyonu ─────────────────────────────────────────────────
    if venue_altitude > 1000:
        home_alt = HIGH_ALT_TEAMS.get(home, 50)
        away_alt = HIGH_ALT_TEAMS.get(away, 50)
        alt_diff = (home_alt - away_alt) / 100  # her 100m fark = +2 Elo
        eh += float(np.clip(alt_diff * 2, -30, 30))

    # ── 6. Sıcaklık Stresi ────────────────────────────────────────────────────
    # 26°C üzerinde serin iklimli takımlar performans kaybeder
    # Her 1°C = 3 Elo ceza (max 24 Elo — 32°C'de)
    if temp_celsius > 26.0:
        heat_penalty = min((temp_celsius - 26.0) * 3.0, 24.0)
        if home in COOL_CLIMATE_TEAMS:
            eh -= heat_penalty
        if away in COOL_CLIMATE_TEAMS:
            ea -= heat_penalty

    # ── 7. Seyahat Yorgunluğu ─────────────────────────────────────────────────
    # 9,000 km'den uzun seyahat ek yorgunluk yaratır
    # Her 1,000 km (eşik üzerinde) = 4 Elo ceza (max 20 Elo)
    if travel_km_h > 9000:
        eh -= min((travel_km_h - 9000) / 1000 * 4.0, 20.0)
    if travel_km_a > 9000:
        ea -= min((travel_km_a - 9000) / 1000 * 4.0, 20.0)

    # ── Kazanma olasılığı (500 ölçekli — daha geniş belirsizlik) ─────────────
    ph = _ko_elo_win_prob(eh, ea)

    # ── 3. Penaltı İstatistikleri (yakın maçlarda etkili) ─────────────────────
    if penalty_stats is not None and 0.40 <= ph <= 0.60:
        pen_h = penalty_stats.get(home, 0.5)
        pen_weight = max(0.0, (0.5 - abs(ph - 0.5)) * 2)  # 0..1
        ph = ph * (1 - pen_weight * 0.15) + pen_h * pen_weight * 0.15

    if rng.uniform() < ph:
        return home, away
    return away, home


# ──────────────────────────────────────────────
# Ana simülasyon fonksiyonu
# ──────────────────────────────────────────────

def run_simulation(
    fixtures: pd.DataFrame,
    knockout_slots: pd.DataFrame,
    predictions: Optional[pd.DataFrame],
    updates: pd.DataFrame,
    elo_map: Dict[str, float],
    n_simulations: int = 5_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Monte Carlo turnuva simülasyonu.

    Returns
    -------
    pd.DataFrame — her takım için P(champion), P(finalist), P(semi),
                   P(quarter), P(round16), P(round32), P(groups) sütunları.
    """
    from data_loader import compute_penalty_stats
    rng = np.random.default_rng(seed)

    # ── İrtifa Elo deflasyonu (WC 2026 bağlamı) ───────────────────────────────
    # Orijinal elo_map'i değiştirmemek için kopya oluştur
    elo_map = dict(elo_map)
    for _team, _delta in ALTITUDE_ELO_DEFLATION.items():
        if _team in elo_map:
            elo_map[_team] = elo_map[_team] + _delta

    # ── Penaltı istatistiklerini yükle ────────────────────────────────────────
    penalty_stats = compute_penalty_stats()

    # ── Update dict: zaten oynanan maçlar ──────────────────────────────────
    update_dict: Dict[int, Tuple[int, int]] = {}
    if updates is not None and not updates.empty:
        for _, r in updates.iterrows():
            try:
                update_dict[int(r["match_id"])] = (int(r["home_score"]), int(r["away_score"]))
            except (ValueError, TypeError):
                pass

    # ── Tahmin sözlüğü ────────────────────────────────────────────────────
    pred_dict: Dict[int, dict] = {}
    if predictions is not None and not predictions.empty:
        for _, r in predictions.iterrows():
            try:
                mid = int(r["match_id"])
                pred_dict[mid] = {
                    "p_home": float(r.get("p_home", 1 / 3)),
                    "p_draw": float(r.get("p_draw", 1 / 3)),
                    "p_away": float(r.get("p_away", 1 / 3)),
                    "lh":     float(r["lambda_home"]) if "lambda_home" in r.index else 1.35,
                    "la":     float(r["lambda_away"]) if "lambda_away" in r.index else 1.35,
                }
            except (ValueError, TypeError, KeyError):
                pass

    # ── Grup maç listesi ──────────────────────────────────────────────────
    group_ids = set(fixtures["match_id"].astype(int))
    group_matches = []
    for _, row in fixtures.iterrows():
        mid  = int(row["match_id"])
        pred = pred_dict.get(mid, {"p_home": 1/3, "p_draw": 1/3, "p_away": 1/3, "lh": 1.35, "la": 1.35})
        group_matches.append({
            "match_id": mid,
            "group":    str(row["group"]),
            "home":     str(row["home_team"]),
            "away":     str(row["away_team"]),
            "ph":       pred["p_home"],
            "pd_":      pred["p_draw"],
            "pa":       pred["p_away"],
            "lh":       pred["lh"],
            "la":       pred["la"],
        })

    # ── Unplayed maç örneklemesi için numpy hazırlığı ─────────────────────
    unplayed = [m for m in group_matches if m["match_id"] not in update_dict]
    n_unplayed = len(unplayed)

    if n_unplayed > 0:
        probs = np.array([[m["ph"], m["pd_"], m["pa"]] for m in unplayed])
        cum_p = np.cumsum(probs, axis=1)
        u = rng.uniform(size=(n_simulations, n_unplayed))
        # 0=H, 1=D, 2=A
        outcomes_all = np.zeros((n_simulations, n_unplayed), dtype=np.int8)
        outcomes_all += (u >= cum_p[np.newaxis, :, 0]).astype(np.int8)
        outcomes_all += (u >= cum_p[np.newaxis, :, 1]).astype(np.int8)
    else:
        outcomes_all = np.empty((n_simulations, 0), dtype=np.int8)

    # ── Eleme slotları (Group/Best3rd çözümlenmemiş) ──────────────────────
    ko_list = []
    for _, row in knockout_slots.iterrows():
        # Venue: stadyum adını al (virgülden önce)
        raw_venue = str(row.get("venue", "")) if "venue" in row.index else ""
        venue_name = raw_venue.split(",")[0].strip()
        # Venue bilgileri (sıcaklık, koordinat, rakım)
        vinfo = VENUE_INFO.get(venue_name, {})
        v_alt  = vinfo.get("alt",  VENUE_ALTITUDE.get(venue_name, 50))
        v_temp = vinfo.get("temp", 22.0)
        v_lat  = vinfo.get("lat",  39.0)
        v_lon  = vinfo.get("lon", -95.0)
        # Date
        date_str = str(row.get("date_utc", "")) if "date_utc" in row.index else ""
        ko_list.append({
            "match_id":  int(row["match_id"]),
            "round":     str(row["round"]),
            "slot_h":    str(row["slot_home"]),
            "slot_a":    str(row["slot_away"]),
            "venue":     venue_name,
            "altitude":  v_alt,
            "temp":      v_temp,
            "lat":       v_lat,
            "lon":       v_lon,
            "date_str":  date_str,
        })

    # ── İstatistik sayaçları ──────────────────────────────────────────────
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # ── N simülasyon ──────────────────────────────────────────────────────
    for sim_idx in range(n_simulations):
        # 1. Simüle edilmiş maç sonuçlarını oluştur
        sim_results: Dict[int, Tuple[int, int]] = dict(update_dict)
        for j, m in enumerate(unplayed):
            outcome = int(outcomes_all[sim_idx, j])
            lh = m["lh"] or 1.35
            la = m["la"] or 1.35
            hg, ag = _sample_goals(lh, la, outcome, rng)
            sim_results[m["match_id"]] = (hg, ag)

        # 2. Grup sıralamalarını hesapla
        standings = _fast_standings(group_matches, sim_results, elo_map)

        # 3. En iyi 8 üçüncüyü seç
        best_thirds = _select_best_thirds(standings)

        # ── İyileştirme 1: WC Form Boost — grup puanları ──────────────────
        wc_pts: Dict[str, int] = {}
        for m in group_matches:
            res = sim_results.get(m["match_id"])
            if res:
                hs, as_ = res
                if hs > as_:
                    wc_pts[m["home"]] = wc_pts.get(m["home"], 0) + 3
                elif hs == as_:
                    wc_pts[m["home"]] = wc_pts.get(m["home"], 0) + 1
                    wc_pts[m["away"]] = wc_pts.get(m["away"], 0) + 1
                else:
                    wc_pts[m["away"]] = wc_pts.get(m["away"], 0) + 3

        # ── İyileştirme 4: Son oynanan tarih takibi ───────────────────────
        last_played: Dict[str, str] = {}

        # 4. Eleme turunu simüle et
        ko_winners: Dict[int, str] = {}
        bt_cursor = [0]
        sim_elo = dict(elo_map)  # Bu simülasyona özel Elo state

        for ko in ko_list:
            mid  = ko["match_id"]
            home = _resolve_ko_slot(ko["slot_h"], standings, best_thirds, ko_winners, bt_cursor)
            away = _resolve_ko_slot(ko["slot_a"], standings, best_thirds, ko_winners, bt_cursor)

            # Dinlenme günleri hesapla
            ko_date_str = ko.get("date_str", "")
            try:
                ko_date = pd.Timestamp(ko_date_str)
            except Exception:
                ko_date = None

            rest_h = _compute_rest_days(home, ko_date, last_played)
            rest_a = _compute_rest_days(away, ko_date, last_played)

            # Seyahat mesafeleri — takım koordinatlarından venue'ya
            v_lat = ko.get("lat", 39.0)
            v_lon = ko.get("lon", -95.0)
            tkm_h = _team_travel_km(home, v_lat, v_lon)
            tkm_a = _team_travel_km(away, v_lat, v_lon)

            ko_kwargs = dict(
                wc_pts=wc_pts,
                venue=ko.get("venue", ""),
                rest_days_home=rest_h,
                rest_days_away=rest_a,
                penalty_stats=penalty_stats if penalty_stats else None,
                venue_altitude=ko.get("altitude", 50),
                temp_celsius=float(ko.get("temp", 22.0)),
                travel_km_h=tkm_h,
                travel_km_a=tkm_a,
            )

            if ko["round"] == "Third-place playoff":
                w, l = _simulate_ko_match(home, away, sim_elo, rng, **ko_kwargs)
                ko_winners[mid] = w
                ko_winners[f"loser_{mid}"] = l
                counts[w]["third"] += 1
                counts[l]["fourth"] += 1
            else:
                w, l = _simulate_ko_match(home, away, sim_elo, rng, **ko_kwargs)
                ko_winners[mid] = w
                ko_winners[f"loser_{mid}"] = l

            # Dinamik Elo güncelle (K=30) — kazanan yükselir, kaybeden düşer
            elo_w = sim_elo.get(w, UNKNOWN_ELO)
            elo_l = sim_elo.get(l, UNKNOWN_ELO)
            expected_w = 1.0 / (1.0 + 10.0 ** (-(elo_w - elo_l) / 500.0))
            K_SIM = 30
            sim_elo[w] = elo_w + K_SIM * (1.0 - expected_w)
            sim_elo[l] = elo_l - K_SIM * (1.0 - expected_w)

            # Oynanan tarihi kaydet
            if ko_date_str:
                last_played[home] = ko_date_str
                last_played[away] = ko_date_str

        # 5. Şampiyonu belirle (Final kazananı)
        finals = [ko for ko in ko_list if ko["round"] == "Final"]
        if not finals:
            finals = [ko_list[-1]]  # fallback
        final_ko = finals[0]
        champion = ko_winners.get(final_ko["match_id"])
        finalist  = ko_winners.get(f"loser_{final_ko['match_id']}")

        # 6. Sayaçları güncelle — istatistikler round seviyesine göre toplanır
        if champion:
            counts[champion]["champion"] += 1
        if finalist:
            counts[finalist]["finalist"] += 1

        # Üçüncülük playoff'u oynayan takımları topla (double-count önlemi)
        third_place_teams: set = set()
        for _tp in ko_list:
            if _tp["round"] == "Third-place playoff":
                _tpw = ko_winners.get(_tp["match_id"])
                _tpl = ko_winners.get(f"loser_{_tp['match_id']}")
                if _tpw: third_place_teams.add(_tpw)
                if _tpl: third_place_teams.add(_tpl)

        # Semi-finalistler = yarı final kaybedenler (üçüncülük oynayacaklar hariç)
        semi_match_ids = [ko["match_id"] for ko in ko_list if ko["round"] == "Semi-final"]
        for smid in semi_match_ids:
            loser = ko_winners.get(f"loser_{smid}")
            if loser and loser not in third_place_teams:
                counts[loser]["semi"] += 1

        quarter_ids = [ko["match_id"] for ko in ko_list if ko["round"] == "Quarter-final"]
        for qmid in quarter_ids:
            loser = ko_winners.get(f"loser_{qmid}")
            if loser:
                counts[loser]["quarter"] += 1

        r16_ids = [ko["match_id"] for ko in ko_list if ko["round"] == "Round of 16"]
        for r16mid in r16_ids:
            loser = ko_winners.get(f"loser_{r16mid}")
            if loser:
                counts[loser]["round16"] += 1

        r32_ids = [ko["match_id"] for ko in ko_list if ko["round"] == "Round of 32"]
        for r32mid in r32_ids:
            loser = ko_winners.get(f"loser_{r32mid}")
            if loser:
                counts[loser]["round32"] += 1

        # Gruptan çıkamayanlar (3. sıra best_thirds'e girmeyenler + 4. sıra)
        for grp_rows in standings.values():
            for pos, (team, *_) in enumerate(grp_rows):
                if pos >= 2 and team not in best_thirds:
                    counts[team]["groups"] += 1

    # ── Normalize ve DataFrame oluştur ────────────────────────────────────
    n = n_simulations
    rows = []
    for team, stat in counts.items():
        rows.append({
            "team":       team,
            "p_champion": round(stat.get("champion", 0) / n, 4),
            "p_finalist": round(stat.get("finalist", 0) / n, 4),
            "p_semi":     round(stat.get("semi",     0) / n, 4),
            "p_fourth":   round(stat.get("fourth",   0) / n, 4),
            "p_third":    round(stat.get("third",    0) / n, 4),
            "p_quarter":  round(stat.get("quarter",  0) / n, 4),
            "p_round16":  round(stat.get("round16",  0) / n, 4),
            "p_round32":  round(stat.get("round32",  0) / n, 4),
            "p_groups":   round(stat.get("groups",   0) / n, 4),
        })

    df = pd.DataFrame(rows).sort_values("p_champion", ascending=False).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────
# Kaydet / Yükle
# ──────────────────────────────────────────────

def save_simulation(df: pd.DataFrame, n_simulations: int, played_count: int) -> None:
    """Simülasyon sonuçlarını diske kaydeder."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(SIMULATION_PATH, index=False)
    meta = {
        "n_simulations": n_simulations,
        "played_matches": played_count,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(SIM_META_PATH, "w") as f:
        json.dump(meta, f)


def load_simulation() -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Kaydedilmiş simülasyon sonuçlarını yükler."""
    df = None
    meta = None
    if os.path.exists(SIMULATION_PATH):
        try:
            df = pd.read_csv(SIMULATION_PATH)
        except Exception:
            pass
    if os.path.exists(SIM_META_PATH):
        try:
            with open(SIM_META_PATH) as f:
                meta = json.load(f)
        except Exception:
            pass
    return df, meta
