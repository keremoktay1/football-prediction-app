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

# Bahis oranları (opsiyonel)
_odds_path = os.path.join(APP_DIR, "data", "processed", "odds_cache.csv")
odds_cache: pd.DataFrame | None = None
if os.path.isfile(_odds_path):
    try:
        odds_cache = pd.read_csv(_odds_path)
    except Exception:
        pass

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

# Bahis oranlarını birleştir (opsiyonel)
if odds_cache is not None and not odds_cache.empty:
    _odds_keep = [c for c in ["match_id", "odds_home", "odds_draw", "odds_away",
                               "implied_home", "implied_draw", "implied_away"]
                  if c in odds_cache.columns]
    if "match_id" in _odds_keep:
        _odds_sub = odds_cache[_odds_keep].copy()
        _odds_sub["match_id"] = pd.to_numeric(_odds_sub["match_id"], errors="coerce")
        merged = merged.merge(_odds_sub, on="match_id", how="left")

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

# ── Tahmin Performans Dashboard ───────────────────────────────────────────────
with st.expander("📊 Tahmin Performansı & İstatistikler", expanded=False):
    try:
        import plotly.graph_objects as go
        import plotly.express as px

        # ── Temel metrikler ──────────────────────────────────────────────────
        played_ids = [mid for mid in update_dict]
        total_played = len(played_ids)
        total_goals = sum(hs + as_ for hs, as_ in update_dict.values())

        # Tahmin sözlüğü oluştur
        _pred_dict: dict = {}
        if predictions is not None and not predictions.empty:
            for _, _r in predictions.iterrows():
                try:
                    _mid = int(_r["match_id"])
                    _pred_dict[_mid] = {
                        "ph": float(_r.get("p_home", 1/3)),
                        "pd": float(_r.get("p_draw", 1/3)),
                        "pa": float(_r.get("p_away", 1/3)),
                        "lh": float(_r["lambda_home"]) if "lambda_home" in _r.index else None,
                        "la": float(_r["lambda_away"]) if "lambda_away" in _r.index else None,
                    }
                except (ValueError, TypeError, KeyError):
                    pass

        # Doğruluk hesabı
        correct = 0
        cat_correct = {"H": 0, "D": 0, "A": 0}
        cat_total   = {"H": 0, "D": 0, "A": 0}
        wrong_count = 0

        for mid_k, (hs, as_) in update_dict.items():
            if mid_k not in _pred_dict:
                continue
            actual_k = "H" if hs > as_ else ("D" if hs == as_ else "A")
            pred = _pred_dict[mid_k]
            pred_k = max({"H": pred["ph"], "D": pred["pd"], "A": pred["pa"]},
                         key=lambda k: {"H": pred["ph"], "D": pred["pd"], "A": pred["pa"]}[k])
            cat_total[actual_k] = cat_total.get(actual_k, 0) + 1
            if actual_k == pred_k:
                correct += 1
                cat_correct[actual_k] = cat_correct.get(actual_k, 0) + 1
            else:
                wrong_count += 1

        matched_count = sum(cat_total.values())
        accuracy_pct  = (correct / matched_count * 100) if matched_count > 0 else 0.0

        # ── Metrik kartlar ───────────────────────────────────────────────────
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Oynanan Maç", total_played)
        mc2.metric("Doğru Tahmin", correct)
        mc3.metric("Doğruluk %", f"{accuracy_pct:.1f}%")
        mc4.metric("Toplam Gol", total_goals)

        if matched_count == 0:
            st.info("Henüz karşılaştırılacak tahmin + sonuç çifti yok.")
        else:
            pc1, pc2 = st.columns(2)

            # ── Confusion Donut Chart ─────────────────────────────────────────
            with pc1:
                st.markdown("**Tahmin Kategorisi Doğruluğu**")
                donut_labels = [
                    "Ev Kazandı ✓", "Beraberlik ✓", "Deplasman ✓", "Yanlış"
                ]
                donut_values = [
                    cat_correct.get("H", 0),
                    cat_correct.get("D", 0),
                    cat_correct.get("A", 0),
                    wrong_count,
                ]
                donut_colors = ["#2ecc71", "#f39c12", "#3498db", "#e74c3c"]
                fig_donut = go.Figure(go.Pie(
                    labels=donut_labels,
                    values=donut_values,
                    hole=0.5,
                    marker_colors=donut_colors,
                    textinfo="label+percent",
                    textfont_size=11,
                ))
                fig_donut.update_layout(
                    height=300,
                    paper_bgcolor="#0E1117",
                    font_color="white",
                    showlegend=False,
                    margin={"t": 10, "b": 10, "l": 10, "r": 10},
                )
                st.plotly_chart(fig_donut, use_container_width=True)

            # ── Beklenen vs Gerçek Gol Scatter ───────────────────────────────
            with pc2:
                st.markdown("**Beklenen vs Gerçek Gol (Her Maç)**")
                xg_rows = []
                for _mid, (hs, as_) in update_dict.items():
                    pred = _pred_dict.get(_mid)
                    if pred and pred["lh"] is not None and pred["la"] is not None:
                        xg_rows.append({
                            "lambda_home": pred["lh"],
                            "lambda_away": pred["la"],
                            "actual_home": hs,
                            "actual_away": as_,
                        })
                if xg_rows:
                    xg_df = pd.DataFrame(xg_rows)
                    fig_xg = go.Figure()
                    fig_xg.add_trace(go.Scatter(
                        x=xg_df["lambda_home"],
                        y=xg_df["actual_home"],
                        mode="markers",
                        name="Ev Sahibi",
                        marker={"color": "#3498db", "size": 7},
                    ))
                    fig_xg.add_trace(go.Scatter(
                        x=xg_df["lambda_away"],
                        y=xg_df["actual_away"],
                        mode="markers",
                        name="Deplasman",
                        marker={"color": "#e74c3c", "size": 7},
                    ))
                    # Diagonal çizgi
                    _max_val = max(
                        xg_df[["lambda_home", "lambda_away", "actual_home", "actual_away"]].max().max(),
                        3.0
                    )
                    fig_xg.add_shape(
                        type="line", x0=0, y0=0, x1=_max_val, y1=_max_val,
                        line={"color": "white", "width": 1, "dash": "dot"},
                    )
                    fig_xg.update_layout(
                        height=300,
                        paper_bgcolor="#0E1117",
                        plot_bgcolor="#1a1a2e",
                        font_color="white",
                        xaxis_title="Beklenen Gol (λ)",
                        yaxis_title="Gerçek Gol",
                        margin={"t": 10, "b": 30, "l": 40, "r": 10},
                        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0},
                    )
                    st.plotly_chart(fig_xg, use_container_width=True)
                else:
                    st.info("Gol beklentisi verisi yok.")

            # ── En Büyük Sürpriz Maçlar (Upsets) ─────────────────────────────
            st.markdown("**En Büyük Sürprizler — Favori Kaybetti**")
            upsets = []
            for _mid, (hs, as_) in update_dict.items():
                pred = _pred_dict.get(_mid)
                if pred is None:
                    continue
                actual_k = "H" if hs > as_ else ("D" if hs == as_ else "A")
                pred_k   = max({"H": pred["ph"], "D": pred["pd"], "A": pred["pa"]},
                               key=lambda k: {"H": pred["ph"], "D": pred["pd"], "A": pred["pa"]}[k])
                if actual_k == pred_k:
                    continue  # Doğru tahmin, sürpriz değil
                # Sürpriz büyüklüğü: favori olasılığı - gerçek sonuç olasılığı
                fav_prob    = max(pred["ph"], pred["pd"], pred["pa"])
                actual_prob = {"H": pred["ph"], "D": pred["pd"], "A": pred["pa"]}[actual_k]
                upset_size  = fav_prob - actual_prob

                # Takım adlarını bul
                _fix_row = merged[merged["match_id"] == _mid]
                if _fix_row.empty:
                    continue
                _home = str(_fix_row.iloc[0]["home_team"])
                _away = str(_fix_row.iloc[0]["away_team"])
                _score = f"{hs}–{as_}"
                upsets.append({
                    "match": f"{_home} vs {_away}",
                    "score": _score,
                    "upset_size": round(upset_size * 100, 1),
                })

            if upsets:
                upset_df = pd.DataFrame(upsets).sort_values("upset_size", ascending=False).head(10)
                fig_ups = px.bar(
                    upset_df,
                    x="upset_size",
                    y="match",
                    text="score",
                    orientation="h",
                    labels={"upset_size": "Sürpriz Büyüklüğü (%)", "match": "Maç"},
                    color="upset_size",
                    color_continuous_scale="Reds",
                )
                fig_ups.update_layout(
                    height=max(250, len(upset_df) * 35),
                    yaxis={"autorange": "reversed"},
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117",
                    font_color="white",
                    coloraxis_showscale=False,
                    margin={"t": 10, "b": 10, "l": 10, "r": 10},
                )
                fig_ups.update_traces(textposition="outside")
                st.plotly_chart(fig_ups, use_container_width=True)
            else:
                st.info("Henüz sürpriz sonuç yok (veya tüm tahminler doğru!).")

    except ImportError:
        st.info("Plotly kurulu değil; performans grafiği gösterilemiyor.")
    except Exception as _dash_exc:
        st.warning(f"Performans dashboard hatası: {_dash_exc}")

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

    # Bahis oranları
    try:
        _oh = match.get("odds_home")
        _od = match.get("odds_draw")
        _oa = match.get("odds_away")
        _ih = match.get("implied_home")
        _ia = match.get("implied_away")
        has_odds = (
            _oh is not None and not (isinstance(_oh, float) and pd.isna(_oh))
        )
        if has_odds:
            _oh = float(_oh)
            _od = float(_od) if (_od is not None and not (isinstance(_od, float) and pd.isna(_od))) else 0.0
            _oa = float(_oa)
            _ih = float(_ih) if (_ih is not None and not (isinstance(_ih, float) and pd.isna(_ih))) else 0.0
            _ia = float(_ia) if (_ia is not None and not (isinstance(_ia, float) and pd.isna(_ia))) else 0.0
            odds_display = f"🎰 {_oh:.2f} / {_od:.2f} / {_oa:.2f}"
            kelly_parts = []
            if has_pred and _ih > 0 and _ia > 0:
                value_h = (ph or 0) - _ih
                value_a = (pa or 0) - _ia
                if value_h > 0.08:
                    odds_display += f"  🟩 VALUE Ev +{value_h:.0%}"
                elif value_a > 0.08:
                    odds_display += f"  🟩 VALUE Dep +{value_a:.0%}"
                # Kelly Criterion: f = (p*b - (1-p)) / b, burada b = decimal_odds - 1
                # Yarı-Kelly kullanılır (daha güvenli)
                for label, p_model, odds_dec in [("Ev", ph or 0, _oh), ("Dep", pa or 0, _oa)]:
                    if odds_dec > 1.01 and p_model > 0:
                        b = odds_dec - 1.0
                        kelly_full = (p_model * b - (1.0 - p_model)) / b
                        half_kelly = max(0.0, kelly_full / 2.0)
                        if half_kelly > 0.005:
                            kelly_parts.append(f"Kelly {label}: {half_kelly:.1%}")
            if kelly_parts:
                odds_display += "  📐 " + "  ".join(kelly_parts)
        else:
            odds_display = ""
    except Exception:
        has_odds = False
        odds_display = ""

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

    if odds_display:
        st.caption(odds_display)

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
