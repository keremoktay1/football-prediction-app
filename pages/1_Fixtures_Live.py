"""
1_Fixtures_Live.py — Canlı Grup Fikstürü ve Skor Girişi.

Özellikler:
  - Grup filtresi (A–L veya Tümü)
  - Her maç için: H% / D% / A% olasılıkları, favori, gerçek skor
  - Renk kodu: doğru tahmin → yeşil, yanlış → kırmızı
  - Her maç için expander ile skor girişi
  - Grup puan tablosu (isteğe bağlı)
"""
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
    st.error("❌ GROUP_FIXTURES.CSV bulunamadı. Veri dizinini kontrol edin.")
    st.stop()

if predictions is None:
    st.warning("⚠️ predictions_latest.csv bulunamadı — olasılık sütunları boş gösterilecek.")

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
    pred_cols = ["match_id"]
    for c in ("p_home", "p_draw", "p_away"):
        if c in predictions.columns:
            pred_cols.append(c)
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

if selected_group != "Tümü":
    view_df = merged[merged["group"] == selected_group].copy()
else:
    view_df = merged.copy()

if only_played:
    view_df = view_df[view_df["match_id"].isin(update_dict.keys())]

st.markdown("---")

# ── Puan tablosu (isteğe bağlı) ──────────────────────────────────────────────
if show_standings:
    try:
        standings = calculate_standings(fixtures, updates, predictions)
        groups_to_show = [selected_group] if selected_group != "Tümü" else sorted(standings.keys())
        cols_per_row = 3
        grp_list = list(groups_to_show)
        for row_start in range(0, len(grp_list), cols_per_row):
            row_groups = grp_list[row_start: row_start + cols_per_row]
            tab_cols = st.columns(len(row_groups))
            for col_widget, grp in zip(tab_cols, row_groups):
                if grp not in standings:
                    continue
                with col_widget:
                    st.markdown(f"**Grup {grp}**")
                    grp_df = standings[grp].reset_index()
                    grp_df = grp_df.rename(columns={"index": "Sıra"})
                    display_cols = [c for c in ["Sıra","Team","P","W","D","L","GD","Pts","Status"] if c in grp_df.columns]
                    st.dataframe(grp_df[display_cols], use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"Puan tablosu hesaplanamadı: {exc}")

    st.markdown("---")

# ── Tablo başlığı ─────────────────────────────────────────────────────────────
header_cols = st.columns([0.4, 0.5, 2.2, 2.2, 2.5, 1.8, 1.3, 0.7])
headers = ["", "Grp", "Ev Sahibi", "Deplasman", "H% / B% / D%", "Favori", "Skor", "GF"]
for hcol, htxt in zip(header_cols, headers):
    hcol.markdown(f"**{htxt}**")

st.markdown("---")

# ── Maç satırları ────────────────────────────────────────────────────────────
for _, match in view_df.iterrows():
    mid  = int(match["match_id"])
    home = str(match["home_team"])
    away = str(match["away_team"])
    grp  = str(match["group"])

    # Olasılıklar
    try:
        ph = float(match["p_home"]) if pd.notna(match.get("p_home")) else None
        pd_ = float(match["p_draw"]) if pd.notna(match.get("p_draw")) else None
        pa = float(match["p_away"]) if pd.notna(match.get("p_away")) else None
        has_pred = (ph is not None and pd_ is not None and pa is not None)
    except (TypeError, ValueError):
        has_pred = False
        ph = pd_ = pa = None

    if has_pred:
        fav_key = max({"H": ph, "D": pd_, "A": pa}, key=lambda k: {"H": ph, "D": pd_, "A": pa}[k])
        fav_label = {"H": home, "D": "Beraberlik", "A": away}[fav_key]
        prob_str = f"{ph:.0%} / {pd_:.0%} / {pa:.0%}"
    else:
        fav_key   = None
        fav_label = "—"
        prob_str  = "—"

    # Gerçek skor
    has_result = mid in update_dict
    if has_result:
        hs, as_ = update_dict[mid]
        actual_key = "H" if hs > as_ else ("D" if hs == as_ else "A")
        score_str  = f"{hs} – {as_}"
        gf_str     = f"{hs - as_:+d}"

        if has_pred:
            correct = (fav_key == actual_key)
            badge   = "🟢" if correct else "🔴"
        else:
            badge = "⚪"
    else:
        actual_key = None
        score_str  = "— : —"
        gf_str     = "—"
        badge      = "⬜"

    # Tarih
    try:
        date_disp = pd.to_datetime(match["date_utc"]).strftime("%d %b %H:%M")
    except Exception:
        date_disp = str(match.get("date_utc", ""))

    # ── Satır ─────────────────────────────────────────────────────────────
    row_cols = st.columns([0.4, 0.5, 2.2, 2.2, 2.5, 1.8, 1.3, 0.7])
    with row_cols[0]:
        st.markdown(badge)
    with row_cols[1]:
        st.markdown(f"**{grp}**")
    with row_cols[2]:
        if has_result and actual_key == "H":
            st.markdown(f"**{home}** ✓")
        else:
            st.markdown(home)
    with row_cols[3]:
        if has_result and actual_key == "A":
            st.markdown(f"**{away}** ✓")
        else:
            st.markdown(away)
    with row_cols[4]:
        st.markdown(prob_str)
    with row_cols[5]:
        st.markdown(f"**{fav_label}**")
    with row_cols[6]:
        if has_result:
            st.markdown(f"**{score_str}**")
        else:
            st.markdown(score_str)
    with row_cols[7]:
        st.markdown(gf_str)

    # ── Skor giriş expander ───────────────────────────────────────────────
    with st.expander(f"✏️  Maç #{mid} skor gir — {date_disp}"):
        ec1, ec2, ec3, ec4, ec5 = st.columns([2, 1, 0.5, 1, 2])
        with ec1:
            st.markdown(f"**{home}**")
        with ec2:
            default_h = update_dict[mid][0] if has_result else 0
            home_input = st.number_input(
                "Ev",
                min_value=0, max_value=20,
                value=default_h,
                key=f"inp_h_{mid}",
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
                key=f"inp_a_{mid}",
                label_visibility="collapsed",
            )
        with ec5:
            st.markdown(f"**{away}**")

        btn1, btn2, _ = st.columns([1, 1, 3])
        with btn1:
            if st.button("💾 Kaydet", key=f"save_{mid}"):
                save_match_update(mid, int(home_input), int(away_input))
                st.success("Kaydedildi!")
                st.rerun()
        with btn2:
            if has_result and st.button("🗑️ Sil", key=f"del_{mid}"):
                delete_match_update(mid)
                st.rerun()

    st.markdown("---")

# ── Alt özet ─────────────────────────────────────────────────────────────────
played_count = len(update_dict)
total_count  = len(fixtures)
st.caption(
    f"Toplam: {total_count} maç | Oynandı: {played_count} | "
    f"Kalan: {total_count - played_count} | "
    "🟢 = doğru tahmin  🔴 = yanlış tahmin  ⚪ = tahmin yok  ⬜ = oynanmadı"
)
