"""
group_standings.py — Grup puan tablosu hesaplama.

Oynanmış maçlar gerçek skoru kullanır.
Oynanmamış maçlar için (isteğe bağlı) tahmin bazlı beklenen puan (xPts)
eklenerek tahmini sıralama gösterilir.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Optional


# ──────────────────────────────────────────────
# Yardımcı fonksiyonlar
# ──────────────────────────────────────────────

def _outcome_points(home_score: int, away_score: int):
    """(home_pts, away_pts) döner."""
    if home_score > away_score:
        return 3, 0
    elif home_score == away_score:
        return 1, 1
    else:
        return 0, 3


def _empty_team_record() -> dict:
    return {
        "P": 0, "W": 0, "D": 0, "L": 0,
        "GF": 0, "GA": 0, "GD": 0, "Pts": 0,
        "xPts": 0.0,
    }


# ──────────────────────────────────────────────
# Ana fonksiyon
# ──────────────────────────────────────────────

def calculate_standings(
    fixtures: pd.DataFrame,
    updates: pd.DataFrame,
    predictions: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Grup puan tablolarını hesaplar.

    Parameters
    ----------
    fixtures    : GROUP_FIXTURES.CSV DataFrame
    updates     : match_updates.csv DataFrame (oynanan maçlar)
    predictions : predictions_latest.csv DataFrame (tahminler, isteğe bağlı)

    Returns
    -------
    dict: group_label -> sıralı puan tablosu DataFrame
        Sütunlar: Team, P, W, D, L, GF, GA, GD, Pts, xPts, TotalPts, Status
    """
    # ── update_dict: match_id → (home_score, away_score)
    update_dict = {}  # type: Dict[int, tuple]
    if updates is not None and not updates.empty:
        for _, row in updates.iterrows():
            try:
                update_dict[int(row["match_id"])] = (
                    int(row["home_score"]),
                    int(row["away_score"]),
                )
            except (ValueError, TypeError):
                pass

    # ── pred_dict: match_id → {p_home, p_draw, p_away}
    pred_dict = {}  # type: Dict[int, dict]
    if predictions is not None and not predictions.empty:
        for _, row in predictions.iterrows():
            try:
                mid = int(row["match_id"])
                pred_dict[mid] = {
                    "p_home": float(row.get("p_home", 1 / 3)),
                    "p_draw": float(row.get("p_draw", 1 / 3)),
                    "p_away": float(row.get("p_away", 1 / 3)),
                }
            except (ValueError, TypeError, KeyError):
                pass

    # ── Takım kayıtlarını grupla
    groups = {}  # type: Dict[str, Dict[str, dict]]

    for _, match in fixtures.iterrows():
        grp = str(match["group"])
        mid = int(match["match_id"])
        home = str(match["home_team"])
        away = str(match["away_team"])

        if grp not in groups:
            groups[grp] = {}
        if home not in groups[grp]:
            groups[grp][home] = _empty_team_record()
        if away not in groups[grp]:
            groups[grp][away] = _empty_team_record()

        if mid in update_dict:
            hs, as_ = update_dict[mid]
            hp, ap = _outcome_points(hs, as_)

            # Ev sahibi
            rec = groups[grp][home]
            rec["P"] += 1
            rec["GF"] += hs
            rec["GA"] += as_
            rec["GD"] += hs - as_
            rec["Pts"] += hp
            if hp == 3:
                rec["W"] += 1
            elif hp == 1:
                rec["D"] += 1
            else:
                rec["L"] += 1

            # Deplasman
            rec = groups[grp][away]
            rec["P"] += 1
            rec["GF"] += as_
            rec["GA"] += hs
            rec["GD"] += as_ - hs
            rec["Pts"] += ap
            if ap == 3:
                rec["W"] += 1
            elif ap == 1:
                rec["D"] += 1
            else:
                rec["L"] += 1

        elif mid in pred_dict:
            # Oynanmamış: beklenen puan
            p = pred_dict[mid]
            groups[grp][home]["xPts"] += 3 * p["p_home"] + 1 * p["p_draw"]
            groups[grp][away]["xPts"] += 3 * p["p_away"] + 1 * p["p_draw"]

    # ── DataFrame'lere çevir ve sırala
    result = {}  # type: Dict[str, pd.DataFrame]

    for grp, teams in sorted(groups.items()):
        rows = []
        for team, s in teams.items():
            rows.append(
                {
                    "Team":     team,
                    "P":        s["P"],
                    "W":        s["W"],
                    "D":        s["D"],
                    "L":        s["L"],
                    "GF":       s["GF"],
                    "GA":       s["GA"],
                    "GD":       s["GD"],
                    "Pts":      s["Pts"],
                    "xPts":     round(s["xPts"], 2),
                    "TotalPts": round(s["Pts"] + s["xPts"], 2),
                }
            )

        df = pd.DataFrame(rows)
        df = df.sort_values(
            ["TotalPts", "Pts", "GD", "GF"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

        # 1-tabanlı indeks → kolay erişim için
        df.index = range(1, len(df) + 1)

        # Durum etiketi
        df["Status"] = ""
        if len(df) >= 1:
            df.at[1, "Status"] = "W"
        if len(df) >= 2:
            df.at[2, "Status"] = "RU"
        if len(df) >= 3:
            df.at[3, "Status"] = "3rd"

        result[grp] = df

    return result


def calculate_deterministic_standings(
    fixtures: pd.DataFrame,
    outcomes: Dict[int, str],
    goal_map: Optional[Dict[int, tuple]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Her maç için verilen H/D/A sonucundan grup puan tablolarını hesaplar.
    Model-predicted veya gerçek sonuçlar için kullanılır.

    Parameters
    ----------
    fixtures  : GROUP_FIXTURES.CSV DataFrame
    outcomes  : {match_id: "H"|"D"|"A"}
    goal_map  : {match_id: (home_goals, away_goals)} — yoksa 1-0/1-1/0-1 kullanılır

    Returns
    -------
    dict: group_label -> sıralı puan tablosu DataFrame
    """
    STANDARD_GOALS: Dict[str, tuple] = {"H": (1, 0), "D": (1, 1), "A": (0, 1)}

    groups: Dict[str, Dict[str, dict]] = {}

    for _, match in fixtures.iterrows():
        grp  = str(match["group"])
        mid  = int(match["match_id"])
        home = str(match["home_team"])
        away = str(match["away_team"])

        if grp not in groups:
            groups[grp] = {}
        if home not in groups[grp]:
            groups[grp][home] = _empty_team_record()
        if away not in groups[grp]:
            groups[grp][away] = _empty_team_record()

        if mid not in outcomes:
            continue

        outcome = outcomes[mid]
        if goal_map and mid in goal_map:
            hs, as_ = int(goal_map[mid][0]), int(goal_map[mid][1])
        else:
            hs, as_ = STANDARD_GOALS.get(outcome, (1, 0))

        hp, ap = _outcome_points(hs, as_)

        for team, gf, ga, pts in [(home, hs, as_, hp), (away, as_, hs, ap)]:
            rec = groups[grp][team]
            rec["P"]  += 1
            rec["GF"] += gf
            rec["GA"] += ga
            rec["GD"] += gf - ga
            rec["Pts"] += pts
            if pts == 3:   rec["W"] += 1
            elif pts == 1: rec["D"] += 1
            else:          rec["L"] += 1

    result: Dict[str, pd.DataFrame] = {}
    for grp, teams in sorted(groups.items()):
        rows = [{"Team": t, **{k: v for k, v in s.items() if k != "xPts"},
                 "xPts": 0.0, "TotalPts": float(s["Pts"])}
                for t, s in teams.items()]
        df = pd.DataFrame(rows).sort_values(
            ["TotalPts", "Pts", "GD", "GF"], ascending=False
        ).reset_index(drop=True)
        df.index = range(1, len(df) + 1)
        df["Status"] = ""
        if len(df) >= 1: df.at[1, "Status"] = "W"
        if len(df) >= 2: df.at[2, "Status"] = "RU"
        if len(df) >= 3: df.at[3, "Status"] = "3rd"
        result[grp] = df

    return result


def is_group_complete(group_label: str, fixtures: pd.DataFrame, updates: pd.DataFrame) -> bool:
    """
    Grubun tüm maçları oynandıysa True döner.
    12 gruplu WC2026 formatında her grupta 6 maç var.
    """
    group_matches = fixtures[fixtures["group"] == group_label]["match_id"].tolist()
    played_ids = set(updates["match_id"].dropna().astype(int)) if not updates.empty else set()
    return all(mid in played_ids for mid in group_matches)


def get_best_third_placed(standings: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Her gruptan 3. sırayı alan takımları alır,
    TotalPts > GD > GF sırasıyla sıralar, en iyi 8'ini döner.
    WC 2026: 12 gruptan 8 en iyi 3.'ü Round of 32'ye geçer.
    """
    rows = []
    for grp, df in standings.items():
        if len(df) >= 3:
            row = df.iloc[2].copy()  # 3. sıra (0-tabanlı 2. indeks)
            row["Group"] = grp
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df_third = pd.DataFrame(rows)
    df_third = df_third.sort_values(
        ["TotalPts", "GD", "GF"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return df_third.head(8)
