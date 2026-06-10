"""
1_Fixtures_Live.py — Canlı Grup Fikstürü ve Skor Girişi.

Özellikler:
  - Grup filtresi (A–L veya Tümü)
  - Her maç için: H% / D% / A% olasılıkları, favori, gerçek skor
  - Renk kodu: doğru tahmin → yeşil, yanlış → kırmızı
  - Inline skor girişi (expander olmadan):
      Oynanmamış → inputlar her zaman görünür
      Oynanmış   → ✏️ butonuyla toggle
  - Grup puan tablosu (isteğe bağlı)
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
    load_match_updates,
    load_predictions,
    save_match_update,
    delete_match_update,
)
from group_standings import calculate_standings
from utils import fmt_date, clamp_pct

# ── Sayfa yapılandırması ─────────────────────────────────────────────────────
st.set_page_config(page_title="Fixtures Live", page_icon="📋", layout="wide")
st.title("📋 Canlı Grup Fikstürü")

# ── Veri yükle ───────────────────────────────────────────────────────────────
try:
    fixtures    = load_fixtures()
    updates     = load_match_updates()
    predictions = load_predictions()
except Exception as exc:
    st.error(f"Veri yükleme hatası: {exc}")
    st.stop()

if fixtures is None:
    st.error("❌ GROUP_FIXTURES.CSV bulunamadı.")
    st.stop()

if predictions is None:
    st.warning("⚠️ predictions_latest.csv bulunamadı — olasılıklar gösterilmeyecek.")

# ── update_dict: match_id → (home_score, away_score) ────────────────────────
update_dict: dict[int, tuple[int, int]] = {}
if updates is not None and not updates.empty:
    for _, r in updates.iterrows():
        try:
            update_dict[int(r["match_id"])] = (int(r["home_score"]), int(r["away_score"]))
        except (ValueError, TypeError):
            pass

# ── Tahminleri birleştir ─────────────────────────────────────────────────────
if predictions is not None:
    _xg_cols = ("p_home", "p_draw", "p_away", "lambda_home", "lambda_away", "over_2_5", "btts", "top_scorelines")
    pred_cols = ["match_id"] + [c for c in _xg_cols if c in predictions.columns]
    merged = fixtures.merge(predictions[pred_cols], on="match_id", how="left")
else:
    merged = fixtures.copy()
    for c in ("p_home", "p_draw", "p_away"):
        merged[c] = None

# ── Filtreler ────────────────────────────────────────────────────────────────
groups = sorted(merged["group"].unique())

fcol1, fcol2, fcol3 = st.columns([2, 1, 1])
with fcol1:
    selected_group = st.selectbox("Grup", ["Tümü"] + list(groups), key="grp_filter")
with fcol2:
    only_played = st.checkbox("Sadece Oynananlar", key="played_filter")
with fcol3:
    show_standings = st.checkbox("Puan Tablosu Göster", key="standings_toggle")

view_df = merged if selected_group == "Tümü" else merged[merged["group"] == selected_group].copy()
if only_played:
    view_df = view_df[view_df["match_id"].isin(update_dict)]

st.markdown("---")

# ── Puan tablosu (isteğe bağlı) ──────────────────────────────────────────────
if show_standings:
    try:
        standings = calculate_standings(fixtures, updates, predictions)
        groups_to_show = [selected_group] if selected_group != "Tümü" else sorted(standings.keys())
        for row_start in range(0, len(groups_to_show), 3):
            tab_cols = st.columns(min(3, len(groups_to_show) - row_start))
            for col_w, grp in zip(tab_cols, groups_to_show[row_start:row_start + 3]):
                if grp not in standings:
                    continue
                with col_w:
                    st.markdown(f"**Grup {grp}**")
                    gdf = standings[grp].reset_index().rename(columns={"index": "Sıra"})
                    show_cols = [c for c in ["Sıra", "Team", "P", "W", "D", "L", "GD", "Pts", "Status"] if c in gdf.columns]
                    st.dataframe(gdf[show_cols], use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"Puan tablosu hesaplanamadı: {exc}")
    st.markdown("---")

# ── Tablo başlığı ─────────────────────────────────────────────────────────────
hdr = st.columns([0.4, 0.5, 2.0, 2.0, 2.2, 1.6, 1.5, 1.2, 0.6])
for col, txt in zip(hdr, ["", "Grp", "Ev Sahibi", "Deplasman", "H% / B% / D%", "Bek. Skor", "Favori", "Skor", "GF"]):
    col.markdown(f"**{txt}**")
st.markdown("---")

# ── Maç satırları ────────────────────────────────────────────────────────────
for _, match in view_df.iterrows():
    mid  = int(match["match_id"])
    home = str(match["home_team"])
    away = str(match["away_team"])
    grp  = str(match["group"])

    # Olasılıklar
    try:
        ph  = clamp_pct(match["p_home"]) if pd.notna(match.get("p_home")) else None
        pdr = clamp_pct(match["p_draw"]) if pd.notna(match.get("p_draw")) else None
        pa  = clamp_pct(match["p_away"]) if pd.notna(match.get("p_away")) else None
        has_pred = ph is not None
    except (TypeError, ValueError):
        has_pred = False
        ph = pdr = pa = None

    if has_pred:
        fav_key   = max({"H": ph, "D": pdr, "A": pa}, key=lambda k: {"H": ph, "D": pdr, "A": pa}[k])
        fav_label = {"H": home, "D": "Beraberlik", "A": away}[fav_key]
        prob_str  = f"{ph:.0%} / {pdr:.0%} / {pa:.0%}"
    else:
        fav_key   = None
        fav_label = "—"
        prob_str  = "—"

    # Gerçek skor
    has_result = mid in update_dict
    if has_result:
        hs, as_ = update_dict[mid]
        actual_key = "H" if hs > as_ else ("D" if hs == as_ else "A")
        score_str  = f"**{hs} – {as_}**"
        gf_str     = f"{hs - as_:+d}"
        badge      = ("🟢" if fav_key == actual_key else "🔴") if has_pred else "⚪"
    else:
        actual_key = None
        score_str  = "— : —"
        gf_str     = "—"
        badge      = "⬜"

    date_disp = fmt_date(match.get("date_utc", ""))

    # xG (Beklenen Skor) hesapla
    try:
        lh = match.get("lambda_home")
        la = match.get("lambda_away")
        if pd.notna(lh) and pd.notna(la):
            xg_main = f"**{float(lh):.1f} – {float(la):.1f}**"
            o25 = match.get("over_2_5")
            btts_v = match.get("btts")
            xg_sub  = f"Ü:{float(o25)*100:.0f}%  BT:{float(btts_v)*100:.0f}%" if (pd.notna(o25) and pd.notna(btts_v)) else ""
            # En olası skor
            top_s = match.get("top_scorelines")
            if pd.notna(top_s):
                import ast
                try:
                    lst = ast.literal_eval(str(top_s))
                    if lst:
                        _h, _a, _p = lst[0]
                        xg_sub += f"  {_h}-{_a}(%{_p*100:.0f})"
                except Exception:
                    pass
        else:
            xg_main = "—"
            xg_sub  = ""
    except Exception:
        xg_main = "—"
        xg_sub  = ""

    # ── Satır 1: maç bilgisi ──────────────────────────────────────────────
    row = st.columns([0.4, 0.5, 2.0, 2.0, 2.2, 1.6, 1.5, 1.2, 0.6])
    row[0].markdown(badge)
    row[1].markdown(f"**{grp}**")
    row[2].markdown(f"**{home}** ✓" if (has_result and actual_key == "H") else home)
    row[3].markdown(f"**{away}** ✓" if (has_result and actual_key == "A") else away)
    row[4].markdown(prob_str)
    with row[5]:
        st.markdown(xg_main)
        if xg_sub:
            st.caption(xg_sub)
    row[6].markdown(f"**{fav_label}**")
    row[7].markdown(score_str)
    row[8].markdown(gf_str)

    # ── Satır 2: inline skor girişi ───────────────────────────────────────
    edit_key = f"edit_{mid}"
    if has_result:
        # Oynanmış → küçük ✏️ toggle butonu
        show_input = st.session_state.get(edit_key, False)
        tgl_col, _ = st.columns([1, 10])
        with tgl_col:
            if st.button("✏️", key=f"tgl_{mid}", help=f"{date_disp} — düzenle"):
                st.session_state[edit_key] = not show_input
                st.rerun()
    else:
        # Oynanmamış → inputlar her zaman görünür
        show_input = True

    if show_input:
        ic1, ic2, ic3, ic4, ic5, ic6, _ = st.columns([1.5, 1.2, 0.6, 0.4, 0.6, 1.2, 4])
        with ic1:
            st.caption(f"{date_disp}")
        with ic2:
            default_h = update_dict[mid][0] if has_result else 0
            h_val = st.number_input(
                home, min_value=0, max_value=20, value=default_h,
                key=f"inp_h_{mid}", label_visibility="collapsed",
            )
        with ic3:
            st.markdown("**–**")
        with ic4:
            default_a = update_dict[mid][1] if has_result else 0
            a_val = st.number_input(
                away, min_value=0, max_value=20, value=default_a,
                key=f"inp_a_{mid}", label_visibility="collapsed",
            )
        with ic5:
            if st.button("💾", key=f"save_{mid}", help="Kaydet"):
                save_match_update(mid, int(h_val), int(a_val))
                st.session_state.pop(edit_key, None)
                st.rerun()
        with ic6:
            if has_result and st.button("🗑️", key=f"del_{mid}", help="Sil"):
                delete_match_update(mid)
                st.session_state.pop(edit_key, None)
                st.rerun()

    st.markdown("---")

# ── Alt özet ─────────────────────────────────────────────────────────────────
played_count = len(update_dict)
total_count  = len(fixtures)
st.caption(
    f"Toplam: {total_count} maç | Oynandı: {played_count} | Kalan: {total_count - played_count} | "
    "🟢 = doğru tahmin  🔴 = yanlış  ⚪ = tahmin yok  ⬜ = oynanmadı"
)
