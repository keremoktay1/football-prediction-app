"""
5_Tournament_View.py — Turnuva Görünümü

İki sekme:
  🔮 Model Tahmini — seçili model tahminlerine göre grup tabloları,
                     en iyi 3.'ler ve knockout bracket (Elo destekli)
  ⚽ Canlı         — önyüzden girilen gerçek skorlara göre aynı görünüm
                     + inline skor girişi (expander'sız)
"""
from __future__ import annotations

import os
import sys
from typing import Dict

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
    load_per_model_predictions,
)
from group_standings import (
    calculate_standings,
    calculate_deterministic_standings,
    get_best_third_placed,
)
from knockout import fill_slots, fill_slots_predicted
from prediction_engine import get_knockout_prediction
from utils import fmt_date, team_display, clamp_pct

# ── Sayfa yapılandırması ──────────────────────────────────────────────────────
st.set_page_config(page_title="Turnuva Görünümü", page_icon="🌍", layout="wide")
st.title("🌍 Turnuva Görünümü")

# ── Veri yükleme ─────────────────────────────────────────────────────────────
try:
    fixtures       = load_fixtures()
    knockout_slots = load_knockout_slots()
    updates        = load_match_updates()
    predictions    = load_predictions()
    elo_df         = load_elo_ratings()
    elo_map        = build_elo_map(elo_df)
    all_model_preds = load_per_model_predictions()
except Exception as exc:
    st.error(f"Veri yükleme hatası: {exc}")
    st.stop()

if fixtures is None:
    st.error("❌ GROUP_FIXTURES.CSV bulunamadı.")
    st.stop()
if knockout_slots is None:
    st.error("❌ KNOCKOUT_SLOTS.CSV bulunamadı.")
    st.stop()

# ── update_dict ───────────────────────────────────────────────────────────────
update_dict: Dict[int, tuple] = {}
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

# ── Yardımcı: Grup tablosu göster ────────────────────────────────────────────

def _show_standings_grid(standings: dict) -> None:
    """12 grup tablosunu 4'lü kolon grids halinde gösterir."""
    groups_sorted = sorted(standings.keys())
    COLS = 4
    for row_start in range(0, len(groups_sorted), COLS):
        grp_batch = groups_sorted[row_start : row_start + COLS]
        cols = st.columns(COLS)
        for col, grp in zip(cols, grp_batch):
            with col:
                st.markdown(f"**Grup {grp}**")
                df = standings[grp]
                display_cols = [c for c in ["Team", "P", "W", "D", "L", "GD", "Pts", "TotalPts", "Status"] if c in df.columns]
                st.dataframe(
                    df[display_cols].rename(columns={"TotalPts": "Toplam"}),
                    use_container_width=True,
                    hide_index=False,
                )


def _show_best_third(best_third: pd.DataFrame) -> None:
    if best_third is None or best_third.empty:
        st.info("Henüz yeterli maç sonucu yok.")
        return
    cols_show = [c for c in ["Group", "Team", "P", "W", "D", "L", "GD", "Pts", "TotalPts"] if c in best_third.columns]
    st.dataframe(best_third[cols_show], use_container_width=True, hide_index=True)


def _show_bracket(ko_df: pd.DataFrame, use_pred_winner: bool = False) -> None:
    """Knockout bracket tablosunu gösterir."""
    rounds_available = sorted(
        [r for r in ko_df["round"].unique() if pd.notna(r)],
        key=lambda r: ROUND_ORDER.get(r, 99),
    )
    for rnd in rounds_available:
        rnd_matches = ko_df[ko_df["round"] == rnd]
        if rnd_matches.empty:
            continue
        st.markdown(f"#### {rnd}")
        hdr = st.columns([0.5, 3, 0.5, 3, 2, 2])
        for h, t in zip(hdr, ["#", "Ev Sahibi", "", "Deplasman", "Olasılık", "Kazanan"]):
            h.markdown(f"**{t}**")
        for _, m in rnd_matches.iterrows():
            mid = int(m["match_id"])
            rh  = str(m.get("resolved_home", m.get("slot_home", "TBD")))
            ra  = str(m.get("resolved_away", m.get("slot_away", "TBD")))
            tbd_h = rh.startswith(("Winner", "Runner", "Best", "Loser"))
            tbd_a = ra.startswith(("Winner", "Runner", "Best", "Loser"))

            if not tbd_h and not tbd_a:
                try:
                    kp = get_knockout_prediction(rh, ra, elo_map, neutral=True)
                    ph = clamp_pct(kp["p_home"])
                    pa = clamp_pct(kp["p_away"])
                    prob_str = f"{ph:.0%} — {pa:.0%}"
                except Exception:
                    prob_str = "—"
            else:
                prob_str = "TBD"

            if use_pred_winner:
                winner_str = str(m.get("pred_winner", "")) or "—"
            else:
                if mid in update_dict:
                    hs, as_ = update_dict[mid]
                    if hs > as_:
                        winner_str = f"🏅 {rh}"
                    elif as_ > hs:
                        winner_str = f"🏅 {ra}"
                    else:
                        winner_str = f"{rh}/{ra} (PSO)"
                else:
                    winner_str = "—"

            row = st.columns([0.5, 3, 0.5, 3, 2, 2])
            row[0].markdown(f"**{mid}**")
            row[1].markdown(team_display(rh) if tbd_h else rh)
            row[2].markdown("vs")
            row[3].markdown(team_display(ra) if tbd_a else ra)
            row[4].markdown(prob_str)
            row[5].markdown(winner_str)
        st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# Sekmeler
# ═══════════════════════════════════════════════════════════════════════════════
tab_pred, tab_live = st.tabs(["🔮 Model Tahmini", "⚽ Canlı"])

# ══════════════════════════════════════════════════════
# TAB 1 — Model Tahmini
# ══════════════════════════════════════════════════════
with tab_pred:
    if all_model_preds is None:
        st.error(
            "❌ `predictions_all_models.csv` bulunamadı.\n\n"
            "`scripts/fast_model_training.py` dosyasını çalıştırın."
        )
    else:
        available_models = sorted(all_model_preds["model"].unique().tolist())
        MODEL_DISPLAY = {
            "Ensemble": "🤝 Ensemble (LR+Poisson)",
            "XGBoost":  "⚡ XGBoost",
            "Poisson":  "📐 Poisson xG",
            "Logistic Regression": "📊 Logistic Regression",
            "Random Forest": "🌲 Random Forest",
            "Elo Baseline": "⚖️ Elo Baseline",
        }
        model_labels = [MODEL_DISPLAY.get(m, m) for m in available_models]
        label_to_model = {MODEL_DISPLAY.get(m, m): m for m in available_models}

        selected_label = st.selectbox(
            "Model Seç", model_labels,
            index=model_labels.index(MODEL_DISPLAY.get("Ensemble", model_labels[0]))
                  if MODEL_DISPLAY.get("Ensemble") in model_labels else 0,
            key="model_select_tv",
        )
        selected_model = label_to_model[selected_label]

        model_df = all_model_preds[all_model_preds["model"] == selected_model].copy()
        model_df = model_df.merge(
            fixtures[["match_id"]].assign(match_id=fixtures["match_id"].astype(int)),
            on="match_id", how="inner",
        )

        # ── Maç sonuçlarını belirle (argmax) ─────────────────────────────────
        def _to_outcome(row) -> str:
            vals = {"H": row["p_home"], "D": row["p_draw"], "A": row["p_away"]}
            return max(vals, key=vals.get)

        model_df["outcome"] = model_df.apply(_to_outcome, axis=1)
        outcomes: Dict[int, str] = dict(zip(model_df["match_id"].astype(int), model_df["outcome"]))

        # ── Deterministic standings ───────────────────────────────────────────
        try:
            pred_standings = calculate_deterministic_standings(fixtures, outcomes)
            pred_best_third = get_best_third_placed(pred_standings)
        except Exception as exc:
            st.warning(f"Puan tablosu hesaplanamadı: {exc}")
            pred_standings = {}
            pred_best_third = pd.DataFrame()

        # ── Knockout bracket ──────────────────────────────────────────────────
        try:
            ko_pred = fill_slots_predicted(
                knockout_slots, pred_standings, pred_best_third, elo_map
            )
        except Exception as exc:
            st.warning(f"Bracket çözümlenemedi: {exc}")
            ko_pred = knockout_slots.copy()
            ko_pred["resolved_home"] = ko_pred["slot_home"]
            ko_pred["resolved_away"] = ko_pred["slot_away"]
            ko_pred["pred_winner"]   = ""

        # ── Maç sonuçları tablosu ─────────────────────────────────────────────
        with st.expander("📋 Tüm Grup Maçı Tahminleri", expanded=False):
            disp_cols = [c for c in ["group", "match_id", "home_team", "away_team",
                                     "p_home", "p_draw", "p_away", "outcome"]
                         if c in model_df.columns]
            pct_df = model_df[disp_cols].copy()
            for c in ("p_home", "p_draw", "p_away"):
                if c in pct_df.columns:
                    pct_df[c] = (pct_df[c] * 100).round(1).astype(str) + "%"
            outcome_map = {"H": "🏠 Ev", "D": "🤝 Berabere", "A": "✈️ Dep"}
            pct_df["outcome"] = pct_df["outcome"].map(outcome_map).fillna(pct_df["outcome"])
            pct_df = pct_df.sort_values(["group", "match_id"])
            st.dataframe(pct_df, use_container_width=True, hide_index=True)

        # ── Grup tabloları ────────────────────────────────────────────────────
        st.markdown("### 📊 Grup Puan Tabloları")
        if pred_standings:
            _show_standings_grid(pred_standings)
        else:
            st.info("Puan tablosu hesaplanamadı.")

        # ── En iyi 3.'ler ─────────────────────────────────────────────────────
        st.markdown("### 🥉 En İyi 3. Sıra Takımlar (Round of 32'ye Geçenler)")
        _show_best_third(pred_best_third)

        # ── Knockout bracket ──────────────────────────────────────────────────
        st.markdown("### 🏆 Eleme Turu Bracket (Tahmin)")
        _show_bracket(ko_pred, use_pred_winner=True)


# ══════════════════════════════════════════════════════
# TAB 2 — Canlı
# ══════════════════════════════════════════════════════
with tab_live:
    # ── Skor girişi ──────────────────────────────────────────────────────────
    st.markdown("### ✏️ Skor Girişi (Grup Maçları)")
    st.caption("Maç sonuçlarını doğrudan buradan girin. Kaydettiğinizde tüm görünüm güncellenir.")

    if fixtures is not None:
        live_groups = sorted(fixtures["group"].unique())
        gcols_nav = st.columns(min(len(live_groups), 12))
        live_grp_sel = st.selectbox(
            "Grup Filtresi", ["Tümü"] + list(live_groups), key="live_grp_tv"
        )

        live_view = fixtures.copy()
        if live_grp_sel != "Tümü":
            live_view = live_view[live_view["group"] == live_grp_sel]

        for _, fmatch in live_view.iterrows():
            fmid   = int(fmatch["match_id"])
            fhome  = str(fmatch["home_team"])
            faway  = str(fmatch["away_team"])
            played = fmid in update_dict
            fdate  = fmt_date(fmatch.get("date_utc", ""))

            c1, c2, c3, c4, c5, c6, c7 = st.columns([0.5, 2.5, 1, 0.5, 1, 2.5, 1.5])
            with c1:
                st.markdown(f"**{fmid}**")
            with c2:
                st.markdown(f"{'✅ ' if played else ''}{fhome}")
            with c3:
                default_h = update_dict[fmid][0] if played else 0
                h_val = st.number_input(
                    fhome, min_value=0, max_value=20,
                    value=default_h,
                    key=f"live_h_{fmid}",
                    label_visibility="collapsed",
                )
            with c4:
                st.markdown("**–**")
            with c5:
                default_a = update_dict[fmid][1] if played else 0
                a_val = st.number_input(
                    faway, min_value=0, max_value=20,
                    value=default_a,
                    key=f"live_a_{fmid}",
                    label_visibility="collapsed",
                )
            with c6:
                st.markdown(faway)
            with c7:
                btn_col, del_col = st.columns(2)
                with btn_col:
                    if st.button("💾", key=f"live_save_{fmid}", help="Kaydet"):
                        save_match_update(fmid, int(h_val), int(a_val))
                        st.rerun()
                with del_col:
                    if played and st.button("🗑️", key=f"live_del_{fmid}", help="Sil"):
                        delete_match_update(fmid)
                        st.rerun()

    st.markdown("---")

    # ── Canlı puan tablosu hesapla ────────────────────────────────────────────
    # (her render'da güncel updates kullanılır)
    try:
        live_updates = load_match_updates()
        live_standings = calculate_standings(fixtures, live_updates, predictions)
        live_best_third = get_best_third_placed(live_standings)
    except Exception as exc:
        st.warning(f"Canlı puan tablosu hesaplanamadı: {exc}")
        live_standings = {}
        live_best_third = pd.DataFrame()

    try:
        ko_live = fill_slots(knockout_slots, live_standings, live_updates, live_best_third)
    except Exception as exc:
        st.warning(f"Canlı bracket çözümlenemedi: {exc}")
        ko_live = knockout_slots.copy()
        ko_live["resolved_home"] = ko_live["slot_home"]
        ko_live["resolved_away"] = ko_live["slot_away"]

    # ── Özet: kaç maç oynandı ────────────────────────────────────────────────
    n_played = len(update_dict)
    n_total  = len(fixtures) if fixtures is not None else 72
    st.info(f"**{n_played} / {n_total}** grup maçı oynandı.")

    # ── Grup tabloları ────────────────────────────────────────────────────────
    st.markdown("### 📊 Canlı Grup Puan Tabloları")
    if live_standings:
        _show_standings_grid(live_standings)
    else:
        st.info("Henüz maç sonucu girilmemiş.")

    # ── En iyi 3.'ler ─────────────────────────────────────────────────────────
    st.markdown("### 🥉 En İyi 3. Sıra Takımlar")
    _show_best_third(live_best_third)

    # ── Knockout bracket ──────────────────────────────────────────────────────
    st.markdown("### 🏆 Eleme Turu Bracket (Canlı)")
    _show_bracket(ko_live, use_pred_winner=False)
