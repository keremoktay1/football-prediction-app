"""
2_Knockout_Bracket.py — Eleme Turu Bracket.

Özellikler:
  - Grup sıralamasından otomatik slot doldurma
  - Round | Slot Home → Slot Away | Tahmin %
  - Oynanan maçlarda kazananı göster, bir üst tura ilet
  - Skor girişi expander (knockout maçlar için)
"""
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

# ── Sayfa yapılandırması ─────────────────────────────────────────────────────
st.set_page_config(page_title="Knockout Bracket", page_icon="🏆", layout="wide")
st.title("🏆 Eleme Turu Bracket")

# ── Veri yükle ───────────────────────────────────────────────────────────────
try:
    fixtures       = load_fixtures()
    knockout_slots = load_knockout_slots()
    updates        = load_match_updates()
    predictions    = load_predictions()
    elo_df         = load_elo_ratings()
    elo_map        = build_elo_map(elo_df)
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

# ── Eleme maçları için Elo tabanlı tahmin ────────────────────────────────────
# Knockout maçları için predictions_latest.csv'de satır yoktur;
# dinamik olarak o maçtaki iki takımın Elo'sundan hesaplanır.
def _ko_pred(home: str, away: str) -> dict:
    """İki takım arası eleme tahmini (beraberlik yok)."""
    if home.startswith(("Winner", "Runner", "Best", "Loser")):
        return {"p_home": 0.5, "p_away": 0.5, "tbd": True}
    try:
        return get_knockout_prediction(home, away, elo_map, neutral=True)
    except Exception:
        return {"p_home": 0.5, "p_away": 0.5, "tbd": False}

# ── update_dict ───────────────────────────────────────────────────────────────
update_dict: dict[int, tuple[int, int]] = {}
if updates is not None and not updates.empty:
    for _, r in updates.iterrows():
        try:
            update_dict[int(r["match_id"])] = (int(r["home_score"]), int(r["away_score"]))
        except (ValueError, TypeError):
            pass

# ── Tur sırası ────────────────────────────────────────────────────────────────
ROUND_ORDER = {
    "Round of 32": 1,
    "Round of 16": 2,
    "Quarter-final": 3,
    "Semi-final": 4,
    "Third-place playoff": 5,
    "Final": 6,
}
ko_resolved["round_order"] = ko_resolved["round"].map(ROUND_ORDER).fillna(99)

# ── Tur filtresi ──────────────────────────────────────────────────────────────
available_rounds = sorted(
    ko_resolved["round"].unique().tolist(),
    key=lambda r: ROUND_ORDER.get(r, 99),
)
selected_round = st.selectbox("Tur Filtresi", ["Tümü"] + available_rounds, key="round_filter")

if selected_round != "Tümü":
    view_ko = ko_resolved[ko_resolved["round"] == selected_round]
else:
    view_ko = ko_resolved

st.markdown("---")

# ── Şampiyonluk ihtimali özeti (simülasyon varsa) ─────────────────────────────
sim_df, sim_meta = load_simulation()
if sim_df is not None and not sim_df.empty:
    st.markdown("#### 🏆 Şampiyonluk Favorileri (Monte Carlo)")
    top5 = sim_df.head(5)
    favs = " | ".join(
        f"**{r['team']}** {r['p_champion']:.0%}"
        for _, r in top5.iterrows()
    )
    st.markdown(favs)
    st.caption("Detaylar için 🏆 Champion Prediction sayfasına bakın.")
    st.markdown("---")

# ── Bracket tablosu ──────────────────────────────────────────────────────────
for round_name in available_rounds:
    round_matches = view_ko[view_ko["round"] == round_name]
    if round_matches.empty:
        continue

    st.markdown(f"### {round_name}")

    header_cols = st.columns([0.5, 3.0, 0.5, 3.0, 2.0, 1.2])
    for hcol, htxt in zip(
        header_cols, ["Maç #", "Ev Sahibi", "", "Deplasman", "Tahmin %", "Sonuç"]
    ):
        hcol.markdown(f"**{htxt}**")

    for _, match in round_matches.iterrows():
        mid        = int(match["match_id"])
        rh         = str(match["resolved_home"])
        ra         = str(match["resolved_away"])
        is_tbd_h   = rh.startswith(("Winner", "Runner", "Best", "Loser"))
        is_tbd_a   = ra.startswith(("Winner", "Runner", "Best", "Loser"))
        has_result = mid in update_dict

        # Olasılık — Elo tabanlı dinamik hesaplama
        if not is_tbd_h and not is_tbd_a:
            kp = _ko_pred(rh, ra)
            ph_disp = kp.get("p_home", 0.5)
            pa_disp = kp.get("p_away", 0.5)
            prob_str = f"{ph_disp:.0%} — {pa_disp:.0%}"
        else:
            prob_str = "TBD"

        # Sonuç
        if has_result:
            hs, as_ = update_dict[mid]
            result_str = f"**{hs} – {as_}**"
            winner = rh if hs > as_ else (ra if as_ > hs else f"{rh}/{ra}")
            badge  = "✅"
        else:
            result_str = "— : —"
            winner     = None
            badge      = "🔲"

        # Tarih
        try:
            date_disp = pd.to_datetime(match["date_utc"]).strftime("%d %b %H:%M")
        except Exception:
            date_disp = str(match.get("date_utc", ""))

        row_cols = st.columns([0.5, 3.0, 0.5, 3.0, 2.0, 1.2])
        with row_cols[0]:
            st.markdown(f"{badge} **{mid}**")
        with row_cols[1]:
            style = "*" if is_tbd_h else ""
            winner_mark = " 🏅" if (has_result and winner == rh) else ""
            st.markdown(f"{style}{rh}{style}{winner_mark}")
        with row_cols[2]:
            st.markdown("**vs**")
        with row_cols[3]:
            style = "*" if is_tbd_a else ""
            winner_mark = " 🏅" if (has_result and winner == ra) else ""
            st.markdown(f"{style}{ra}{style}{winner_mark}")
        with row_cols[4]:
            st.markdown(prob_str)
        with row_cols[5]:
            st.markdown(result_str)

        # Skor giriş expander
        with st.expander(f"✏️  Maç #{mid} skor gir — {date_disp}"):
            if is_tbd_h or is_tbd_a:
                st.info("Slot(lar) henüz belirlenmedi (grup aşaması tamamlanmamış).")
                if not is_tbd_h and not is_tbd_a:
                    pass  # yine de girişe izin ver
            else:
                ec1, ec2, ec3, ec4, ec5 = st.columns([2, 1, 0.5, 1, 2])
                with ec1:
                    st.markdown(f"**{rh}**")
                with ec2:
                    default_h = update_dict[mid][0] if has_result else 0
                    home_input = st.number_input(
                        "Ev",
                        min_value=0, max_value=20,
                        value=default_h,
                        key=f"ko_h_{mid}",
                        label_visibility="collapsed",
                    )
                with ec3:
                    st.markdown("**–**")
                with ec4:
                    default_a = update_dict[mid][1] if has_result else 0
                    away_input = st.number_input(
                        "Dep",
                        min_value=0, max_value=20,
                        value=default_a,
                        key=f"ko_a_{mid}",
                        label_visibility="collapsed",
                    )
                with ec5:
                    st.markdown(f"**{ra}**")

                btn1, btn2, _ = st.columns([1, 1, 3])
                with btn1:
                    if st.button("💾 Kaydet", key=f"ko_save_{mid}"):
                        save_match_update(mid, int(home_input), int(away_input))
                        st.success("Kaydedildi!")
                        st.rerun()
                with btn2:
                    if has_result and st.button("🗑️ Sil", key=f"ko_del_{mid}"):
                        delete_match_update(mid)
                        st.rerun()

        st.markdown("---")

# ── En iyi 3.'ler özeti ───────────────────────────────────────────────────────
if not best_third.empty:
    with st.expander("📊 En İyi 3. Sıra Takımlar (Round of 32'ye Geçenler)"):
        cols_to_show = [c for c in ["Group","Team","P","W","D","L","GD","Pts","TotalPts"] if c in best_third.columns]
        st.dataframe(best_third[cols_to_show], use_container_width=True, hide_index=True)

# ── Bilgi notu ────────────────────────────────────────────────────────────────
st.caption(
    "İtalik = slot henüz dolmadı (TBD) | "
    "🏅 = kazanan takım | "
    "✅ = oynanmış maç | "
    "🔲 = oynanmamış"
)
