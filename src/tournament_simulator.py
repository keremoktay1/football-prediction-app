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


def _sample_goals(lh: float, la: float, outcome: int, rng) -> Tuple[int, int]:
    """
    Poisson'dan gol sayısı örnekle; outcome (0=H, 1=D, 2=A) tutarlı olsun.
    3 denemede tutarsızsa düz skor kullan.
    """
    for _ in range(3):
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
) -> Tuple[str, str]:
    """(winner, loser) döner — beraberlik yok."""
    eh = elo_map.get(home, UNKNOWN_ELO)
    ea = elo_map.get(away, UNKNOWN_ELO)
    ph = _elo_win_prob(eh, ea)
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
    rng = np.random.default_rng(seed)

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
        ko_list.append({
            "match_id": int(row["match_id"]),
            "round":    str(row["round"]),
            "slot_h":   str(row["slot_home"]),
            "slot_a":   str(row["slot_away"]),
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

        # 4. Eleme turunu simüle et
        ko_winners: Dict[int, str] = {}
        bt_cursor = [0]

        for ko in ko_list:
            mid  = ko["match_id"]
            home = _resolve_ko_slot(ko["slot_h"], standings, best_thirds, ko_winners, bt_cursor)
            away = _resolve_ko_slot(ko["slot_a"], standings, best_thirds, ko_winners, bt_cursor)

            if ko["round"] == "Third-place playoff":
                # 3. yer maçı için de sayım tut
                w, l = _simulate_ko_match(home, away, elo_map, rng)
                ko_winners[mid] = w
                ko_winners[f"loser_{mid}"] = l
                counts[w]["third"] += 1
                counts[l]["fourth"] += 1
            else:
                w, l = _simulate_ko_match(home, away, elo_map, rng)
                ko_winners[mid] = w
                ko_winners[f"loser_{mid}"] = l

        # 5. Şampiyonu belirle (Final kazananı)
        final_ko = ko_list[-1]  # Son eleman = Final (match_id 104)
        champion = ko_winners.get(final_ko["match_id"])
        finalist  = ko_winners.get(f"loser_{final_ko['match_id']}")

        # 6. Sayaçları güncelle — istatistikler round seviyesine göre toplanır
        if champion:
            counts[champion]["champion"] += 1
        if finalist:
            counts[finalist]["finalist"] += 1

        # Semi-finalistler = son 4 mağlup
        semi_match_ids = [ko["match_id"] for ko in ko_list if ko["round"] == "Semi-final"]
        for smid in semi_match_ids:
            loser = ko_winners.get(f"loser_{smid}")
            if loser:
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
