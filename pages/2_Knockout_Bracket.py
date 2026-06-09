"""
2_Knockout_Bracket.py — Eleme Turu Bracket.

Özellikler:
  - Grup sıralamasından otomatik slot doldurma
  - Round | Slot Home → Slot Away | Tahmin %
  - Oynanan maçlarda kazananı göster, bir üst tura ilet
  - Inline skor girişi (expander olmadan):
      Oynanmış   → ✏️ toggle
      Oynanmamış → inputlar her zaman görünür (TBD yoksa)
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(APP_DIR, "src"))

from data_loader import (
    load_fixtures,
    load_knockout_slots,
    load_match_updates,
    load_predictions,
    load_elo_ratings,
    build_elo_map,
    save_match_update,
    delete_match_update,
)
from group_standings import calculate_standings, get_best_third_placed
from knockout import fill_slots, get_match_result_teams
from prediction_engine import get_knockout_prediction
from tournament_simulator import load_simulation
from utils import fmt_date, team_display, clamp_pct

# ── Sayfa yapılandırması ─────────────────────────────────────────────────────
st.set_page_config(page_title="Knockout Bracket", page_icon="🏆", layout="wide")
st.title("🏆 Eleme Turu Bracket")

# ── Veri yükle ───────────────────────────────────────────────────────────────
try:
    fixtures       = load_fixtures()
    knockout_slots = load_knockout_slots()
    updates        = load_match_updates()
    predictions    = load_predictions()
    elo_map        = build_elo_map(load_elo_ratings())
except Exception as exc:
    st.error(f"Veri yükleme hatası: {exc}")
    st.stop()

if knockout_slots is None:
    st.error("❌ KNOCKOUT_SLOTS.CSV bulunamadı.")
    st.stop()

if fixtures is None:
    st.error("❌ GROUP_FIXTURES.CSV bulunamadı.")
    st.stop()

# ── Puan tablosu & En iyi 3.'ler ─────────────────────────────────────────────
try:
    standings  = calculate_standings(fixtures, updates, predictions)
    best_third = get_best_third_placed(standings)
except Exception as exc:
    st.warning(f"Puan tablosu hesaplanamadı: {exc}")
    standings  = {}
    best_third = pd.DataFrame()

# ── Slot doldurma ─────────────────────────────────────────────────────────────
try:
    ko_resolved = fill_slots(knockout_slots, standings, updates, best_third)
except Exception as exc:
    st.warning(f"Slot doldurma hatası: {exc}")
    ko_resolved = knockout_slots.copy()
    ko_resolved["resolved_home"] = ko_resolved["slot_home"]
    ko_resolved["resolved_away"] = ko_resolved["slot_away"]

# ── update_dict ───────────────────────────────────────────────────────────────
update_dict: dict[int, tuple[int, int]] = {}
if updates is not None and not updates.empty:
    for _, r in updates.iterrows():
        try:
            update_dict[int(r["match_id"])] = (int(r["home_score"]), int(r["away_score"]))
        except (ValueError, TypeError):
            pass

ROUND_ORDER = {
    "Round of 32": 1, "Round of 16": 2, "Quarter-final": 3,
    "Semi-final": 4, "Third-place playoff": 5, "Final": 6,
}
ko_resolved["round_order"] = ko_resolved["round"].map(ROUND_ORDER).fillna(99)

available_rounds = sorted(
    ko_resolved["round"].unique().tolist(),
    key=lambda r: ROUND_ORDER.get(r, 99),
)

# ── Tur filtresi + Şampiyonluk özeti ─────────────────────────────────────────
selected_round = st.selectbox("Tur Filtresi", ["Tümü"] + available_rounds, key="round_filter")

sim_df, _ = load_simulation()
if sim_df is not None and not sim_df.empty:
    st.markdown("#### 🏆 Şampiyonluk Favorileri (Monte Carlo)")
    favs = " | ".join(
        f"**{r['team']}** {r['p_champion']:.0%}"
        for _, r in sim_df.head(5).iterrows()
    )
    st.markdown(favs)
    st.caption("Detaylar için 🏆 Champion Prediction sayfasına bakın.")

st.markdown("---")

view_ko = ko_resolved if selected_round == "Tümü" else ko_resolved[ko_resolved["round"] == selected_round]

# ── Bracket tablosu ──────────────────────────────────────────────────────────
for round_name in available_rounds:
    round_matches = view_ko[view_ko["round"] == round_name]
    if round_matches.empty:
        continue

    st.markdown(f"### {round_name}")

    hdr = st.columns([0.5, 3.0, 0.5, 3.0, 2.0, 1.2])
    for h, t in zip(hdr, ["Maç #", "Ev Sahibi", "", "Deplasman", "Tahmin %", "Sonuç"]):
        h.markdown(f"**{t}**")

    for _, match in round_matches.iterrows():
        mid      = int(match["match_id"])
        rh       = str(match["resolved_home"])
        ra       = str(match["resolved_away"])
        tbd_h    = rh.startswith(("Winner", "Runner", "Best", "Loser"))
        tbd_a    = ra.startswith(("Winner", "Runner", "Best", "Loser"))
        has_result = mid in update_dict

        # Olasılık
        if not tbd_h and not tbd_a:
            try:
                kp = get_knockout_prediction(rh, ra, elo_map, neutral=True)
                ph = clamp_pct(kp.get("p_home", 0.5))
                pa = clamp_pct(kp.get("p_away", 0.5))
                prob_str = f"{ph:.0%} — {pa:.0%}"
            except Exception:
                prob_str = "—"
        else:
            prob_str = "TBD"

        # Sonuç
        if has_result:
            hs, as_ = update_dict[mid]
            winner  = rh if hs > as_ else (ra if as_ > hs else None)
            result_str = f"**{hs} – {as_}**"
            badge = "✅"
        else:
            winner     = None
            result_str = "— : —"
            badge      = "🔲"

        date_disp = fmt_date(match.get("date_utc", ""))

        # ── Satır 1: maç bilgisi ──────────────────────────────────────────
        row = st.columns([0.5, 3.0, 0.5, 3.0, 2.0, 1.2])
        row[0].markdown(f"{badge} **{mid}**")
        rh_disp = f"**{rh}** 🏅" if (has_result and winner == rh) else team_display(rh)
        ra_disp = f"**{ra}** 🏅" if (has_result and winner == ra) else team_display(ra)
        row[1].markdown(rh_disp)
        row[2].markdown("**vs**")
        row[3].markdown(ra_disp)
        row[4].markdown(prob_str)
        row[5].markdown(result_str)

        # ── Satır 2: inline skor girişi ───────────────────────────────────
        if tbd_h or tbd_a:
            st.caption(f"⏳ {date_disp} — Slot henüz belirlenmedi")
        else:
            edit_key = f"ko_edit_{mid}"
            if has_result:
                show_input = st.session_state.get(edit_key, False)
                tgl_col, _ = st.columns([1, 10])
                with tgl_col:
                    if st.button("✏️", key=f"ko_tgl_{mid}", help=f"{date_disp} — düzenle"):
                        st.session_state[edit_key] = not show_input
                        st.rerun()
            else:
                show_input = True

            if show_input:
                ic1, ic2, ic3, ic4, ic5, ic6, _ = st.columns([1.5, 1.2, 0.6, 0.4, 0.6, 1.2, 4])
                with ic1:
                    st.caption(date_disp)
                with ic2:
                    dh = update_dict[mid][0] if has_result else 0
                    h_val = st.number_input(
                        rh, min_value=0, max_value=20, value=dh,
                        key=f"ko_h_{mid}", label_visibility="collapsed",
                    )
                with ic3:
                    st.markdown("**–**")
                with ic4:
                    da = update_dict[mid][1] if has_result else 0
                    a_val = st.number_input(
                        ra, min_value=0, max_value=20, value=da,
                        key=f"ko_a_{mid}", label_visibility="collapsed",
                    )
                with ic5:
                    if st.button("💾", key=f"ko_save_{mid}", help="Kaydet"):
                        save_match_update(mid, int(h_val), int(a_val))
                        st.session_state.pop(edit_key, None)
                        st.rerun()
                with ic6:
                    if has_result and st.button("🗑️", key=f"ko_del_{mid}", help="Sil"):
                        delete_match_update(mid)
                        st.session_state.pop(edit_key, None)
                        st.rerun()

        st.markdown("---")

# ── En iyi 3.'ler özeti ───────────────────────────────────────────────────────
if not best_third.empty:
    with st.expander("📊 En İyi 3. Sıra Takımlar (Round of 32'ye Geçenler)"):
        cols_to_show = [c for c in ["Group", "Team", "P", "W", "D", "L", "GD", "Pts", "TotalPts"] if c in best_third.columns]
        st.dataframe(best_third[cols_to_show], use_container_width=True, hide_index=True)

st.caption(
    "`[TBD]` = slot henüz dolmadı | 🏅 = kazanan | ✅ = oynanmış | 🔲 = oynanmamış"
)
