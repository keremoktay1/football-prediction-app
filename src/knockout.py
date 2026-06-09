"""
knockout.py — Eleme turu slot doldurma ve bracket ilerletme.

Slot formatları:
  Grup aşaması → "Winner Group A", "Runner-up Group B", "Best 3rd (Groups ...)"
  Eleme aşaması → "Winner Match 73", "Loser Match 101"
"""
from __future__ import annotations

import re
import pandas as pd
from typing import Dict, Optional


def fill_slots(
    knockout_slots: pd.DataFrame,
    standings: dict,
    updates: pd.DataFrame,
    best_third: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Knockout bracket slotlarını gerçek takım adlarıyla doldurur.

    Parameters
    ----------
    knockout_slots : KNOCKOUT_SLOTS.CSV DataFrame
    standings      : calculate_standings() çıktısı (group_label → DataFrame)
    updates        : match_updates.csv DataFrame
    best_third     : get_best_third_placed() çıktısı (isteğe bağlı)

    Returns
    -------
    knockout_slots kopyası; 'resolved_home' ve 'resolved_away' sütunları ekli.
    Çözümlenemeyen slotlar orijinal metin olarak bırakılır.
    """
    df = knockout_slots.copy()

    # Çözümleme için resolved sütunları başlangıçta slot değerleriyle doldur
    df["resolved_home"] = df["slot_home"].astype(str)
    df["resolved_away"] = df["slot_away"].astype(str)

    # match_id → row indexi için hızlı lookup
    mid_to_idx = {int(r["match_id"]): i for i, r in df.iterrows()}

    # Updates dict
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

    # ── 1. Geçiş: Grup slotlarını çöz ──────────────────────────────────────

    def resolve_group_slot(slot: str) -> str:
        m = re.match(r"(Winner|Runner-up)\s+Group\s+([A-Za-z])", slot)
        if not m:
            return slot
        role, grp = m.group(1), m.group(2).upper()
        if grp not in standings:
            return slot
        grp_df = standings[grp]
        if role == "Winner" and len(grp_df) >= 1:
            return str(grp_df.iloc[0]["Team"])
        if role == "Runner-up" and len(grp_df) >= 2:
            return str(grp_df.iloc[1]["Team"])
        return slot

    for idx in df.index:
        df.at[idx, "resolved_home"] = resolve_group_slot(df.at[idx, "resolved_home"])
        df.at[idx, "resolved_away"] = resolve_group_slot(df.at[idx, "resolved_away"])

    # ── 2. Geçiş: "Best 3rd" slotlarını çöz ────────────────────────────────

    best_third_teams: list[str] = []
    if best_third is not None and not best_third.empty and "Team" in best_third.columns:
        best_third_teams = list(best_third["Team"].astype(str).values)

    bt_cursor = 0
    for idx in df.index:
        for col in ("resolved_home", "resolved_away"):
            val = df.at[idx, col]
            if str(val).startswith("Best 3rd"):
                if bt_cursor < len(best_third_teams):
                    df.at[idx, col] = best_third_teams[bt_cursor]
                    bt_cursor += 1

    # ── 3. Geçiş (çoklu): Knockout kazanan/mağlup slotlarını çöz ───────────
    # Bracket derinliği ~5 tur olduğu için 6 geçiş yeterli

    def get_team_from_match(mid: int, role: str) -> Optional[str]:
        """'Winner' veya 'Loser' rolüne göre mid'den takım döner."""
        if mid not in update_dict:
            return None
        hs, as_ = update_dict[mid]
        row = df[df["match_id"] == mid]
        if row.empty:
            return None
        fh = str(row.iloc[0]["resolved_home"])
        fa = str(row.iloc[0]["resolved_away"])
        if role == "Winner":
            if hs > as_:
                return fh
            elif as_ > hs:
                return fa
            else:
                return f"{fh} / {fa}"  # Beraberlik (penaltılar gerekebilir)
        elif role == "Loser":
            if hs > as_:
                return fa
            elif as_ > hs:
                return fh
            else:
                return f"{fh} / {fa}"
        return None

    for _pass in range(6):
        changed = False
        for idx in df.index:
            for col in ("resolved_home", "resolved_away"):
                val = str(df.at[idx, col])
                m = re.match(r"(Winner|Loser)\s+Match\s+(\d+)", val)
                if not m:
                    continue
                role = m.group(1)
                prev_mid = int(m.group(2))
                resolved = get_team_from_match(prev_mid, role)
                if resolved is not None:
                    df.at[idx, col] = resolved
                    changed = True
        if not changed:
            break

    return df


def get_match_result_teams(
    match_id: int,
    knockout_slots_resolved: pd.DataFrame,
    updates: pd.DataFrame,
) -> dict:
    """
    Belirli bir knockout maçının kazanan/mağlup takımını döner.

    Returns
    -------
    {
      'home_team': str,
      'away_team': str,
      'home_score': int | None,
      'away_score': int | None,
      'winner': str | None,
      'loser':  str | None,
      'played': bool,
    }
    """
    result = {
        "home_team":  None,
        "away_team":  None,
        "home_score": None,
        "away_score": None,
        "winner":     None,
        "loser":      None,
        "played":     False,
    }

    row = knockout_slots_resolved[knockout_slots_resolved["match_id"] == match_id]
    if row.empty:
        return result

    result["home_team"] = str(row.iloc[0]["resolved_home"])
    result["away_team"] = str(row.iloc[0]["resolved_away"])

    if updates is None or updates.empty:
        return result

    upd = updates[updates["match_id"] == match_id]
    if upd.empty:
        return result

    try:
        hs = int(upd.iloc[0]["home_score"])
        as_ = int(upd.iloc[0]["away_score"])
    except (ValueError, TypeError):
        return result

    result["home_score"] = hs
    result["away_score"] = as_
    result["played"] = True

    if hs > as_:
        result["winner"] = result["home_team"]
        result["loser"]  = result["away_team"]
    elif as_ > hs:
        result["winner"] = result["away_team"]
        result["loser"]  = result["home_team"]
    else:
        result["winner"] = f"{result['home_team']} (PSO)"
        result["loser"]  = f"{result['away_team']} (PSO)"

    return result
