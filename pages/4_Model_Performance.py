"""
4_Model_Performance.py — Canlı Model Doğruluk Takibi.

Özellikler:
  - Oynanan maçlarda: tahmin vs gerçek sonuç karşılaştırması
  - Kümülatif log-loss, Brier score, accuracy çizgisi (Plotly)
  - Tablo: Maç | Tahmin edilen kazanan | Gerçek kazanan | Doğru? | Log Loss
"""
import os
import sys
import math

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(APP_DIR, "src"))

from data_loader import load_fixtures, load_match_updates, load_predictions

# ── Sayfa yapılandırması ─────────────────────────────────────────────────────
st.set_page_config(page_title="Model Performance", page_icon="📊", layout="wide")
st.title("📊 Model Performans Takibi")
st.markdown("Maç girdikçe modelin doğruluğu anlık olarak izlenir.")

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
    st.warning(
        "⚠️ predictions_latest.csv bulunamadı. "
        "Lütfen 03_model_training.ipynb'yi çalıştırın."
    )
    st.stop()

if updates is None or updates.empty:
    st.info("ℹ️ Henüz skor girilmemiş. Fixtures Live sayfasından maç sonuçlarını girin.")
    st.stop()

# ── Tahmin ve gerçek sonuçları birleştir ──────────────────────────────────────
# Sadece grup maçları (match_id 1–72)
group_match_ids = set(fixtures["match_id"].astype(int))
played = updates[updates["match_id"].isin(group_match_ids)].copy()

if played.empty:
    st.info("ℹ️ Hiç grup maçı oynandı olarak işaretlenmemiş.")
    st.stop()

# Gerçek sonuç
def actual_outcome(hs: int, as_: int) -> str:
    if hs > as_:
        return "H"
    elif as_ > hs:
        return "A"
    return "D"

played["actual"] = played.apply(
    lambda r: actual_outcome(int(r["home_score"]), int(r["away_score"])), axis=1
)

# Tahminler ile birleştir
pred_cols = ["match_id"] + [c for c in ("p_home","p_draw","p_away") if c in predictions.columns]
merged = played.merge(
    predictions[pred_cols],
    on="match_id",
    how="left",
).merge(
    fixtures[["match_id","group","home_team","away_team"]],
    on="match_id",
    how="left",
)

# Gerekli sütunlar eksikse doldur
for col in ("p_home", "p_draw", "p_away"):
    if col not in merged.columns:
        merged[col] = 1 / 3

merged[["p_home","p_draw","p_away"]] = merged[["p_home","p_draw","p_away"]].fillna(1 / 3)

# ── Tahmin edilen kazanan ─────────────────────────────────────────────────────
def predicted_outcome(row) -> str:
    probs = {"H": row["p_home"], "D": row["p_draw"], "A": row["p_away"]}
    return max(probs, key=probs.get)

merged["predicted"] = merged.apply(predicted_outcome, axis=1)
merged["correct"]   = (merged["predicted"] == merged["actual"])

# ── Log-loss per maç ──────────────────────────────────────────────────────────
CLIP = 1e-7

def row_log_loss(row) -> float:
    actual = row["actual"]
    p = {
        "H": max(CLIP, min(1 - CLIP, row["p_home"])),
        "D": max(CLIP, min(1 - CLIP, row["p_draw"])),
        "A": max(CLIP, min(1 - CLIP, row["p_away"])),
    }
    return -math.log(p[actual])

merged["log_loss"] = merged.apply(row_log_loss, axis=1)

# ── Brier score per maç ───────────────────────────────────────────────────────
def row_brier(row) -> float:
    actual = row["actual"]
    y = {"H": 0.0, "D": 0.0, "A": 0.0}
    y[actual] = 1.0
    bs = (row["p_home"] - y["H"]) ** 2 + (row["p_draw"] - y["D"]) ** 2 + (row["p_away"] - y["A"]) ** 2
    return bs

merged["brier"] = merged.apply(row_brier, axis=1)

# Kümülatif metrikler
merged = merged.sort_values("updated_at").reset_index(drop=True)
merged["cum_logloss"]  = merged["log_loss"].expanding().mean()
merged["cum_brier"]    = merged["brier"].expanding().mean()
merged["cum_accuracy"] = merged["correct"].expanding().mean() * 100
merged["match_num"]    = range(1, len(merged) + 1)

# ── Özet metrikler ────────────────────────────────────────────────────────────
n = len(merged)
avg_logloss  = merged["log_loss"].mean()
avg_brier    = merged["brier"].mean()
accuracy_pct = merged["correct"].mean() * 100

st.markdown("---")
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Oynanan Maç", n)
with m2:
    st.metric("Doğru Tahmin", f"{merged['correct'].sum()} / {n}")
with m3:
    st.metric("Accuracy", f"{accuracy_pct:.1f}%")
with m4:
    st.metric("Ortalama Log-Loss", f"{avg_logloss:.3f}")

st.markdown("---")

# ── Kümülatif grafikler ──────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Kümülatif Accuracy (%)", "Kümülatif Log-Loss", "Kümülatif Brier Score"),
    )

    x = merged["match_num"]

    fig.add_trace(
        go.Scatter(x=x, y=merged["cum_accuracy"], mode="lines+markers",
                   name="Accuracy", line={"color": "#4CAF50", "width": 2}),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=merged["cum_logloss"], mode="lines+markers",
                   name="Log-Loss", line={"color": "#F44336", "width": 2}),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=x, y=merged["cum_brier"], mode="lines+markers",
                   name="Brier", line={"color": "#2196F3", "width": 2}),
        row=1, col=3,
    )

    # Referans çizgisi: rastgele model (accuracy ~33%, logloss ~1.099)
    fig.add_hline(y=33.33, row=1, col=1, line_dash="dash", line_color="gray",
                  annotation_text="Rastgele (~33%)", annotation_position="bottom right")
    fig.add_hline(y=1.099, row=1, col=2, line_dash="dash", line_color="gray",
                  annotation_text="Rastgele", annotation_position="bottom right")

    fig.update_layout(
        height=350,
        showlegend=False,
        margin={"t": 50, "b": 30, "l": 30, "r": 30},
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font_color="white",
    )
    fig.update_xaxes(title_text="Maç #", gridcolor="#333")
    fig.update_yaxes(gridcolor="#333")

    st.plotly_chart(fig, use_container_width=True)

except ImportError:
    st.info("Grafik için Plotly gerekli (`pip install plotly`).")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Accuracy", f"{accuracy_pct:.1f}%")
    with c2:
        st.metric("Log-Loss", f"{avg_logloss:.3f}")
    with c3:
        st.metric("Brier Score", f"{avg_brier:.3f}")

st.markdown("---")

# ── Detay tablosu ─────────────────────────────────────────────────────────────
st.markdown("#### Maç Bazlı Detay")

def predicted_winner_label(row) -> str:
    pred = row["predicted"]
    return {
        "H": str(row.get("home_team", "Ev")),
        "D": "Beraberlik",
        "A": str(row.get("away_team", "Dep")),
    }.get(pred, pred)

def actual_winner_label(row) -> str:
    actual = row["actual"]
    return {
        "H": str(row.get("home_team", "Ev")),
        "D": "Beraberlik",
        "A": str(row.get("away_team", "Dep")),
    }.get(actual, actual)

table_df = merged.copy()
table_df["Maç"] = table_df["home_team"] + " vs " + table_df["away_team"]
table_df["Skor"] = table_df["home_score"].astype(str) + "–" + table_df["away_score"].astype(str)
table_df["Tahmini Kazanan"] = table_df.apply(predicted_winner_label, axis=1)
table_df["Gerçek Kazanan"]  = table_df.apply(actual_winner_label, axis=1)
table_df["Doğru?"]          = table_df["correct"].map({True: "✅", False: "❌"})
table_df["Log Loss"]        = table_df["log_loss"].round(3)
table_df["H% / B% / D%"]   = (
    table_df["p_home"].map("{:.0%}".format) + " / " +
    table_df["p_draw"].map("{:.0%}".format) + " / " +
    table_df["p_away"].map("{:.0%}".format)
)

display_cols = ["match_id","group","Maç","Skor","H% / B% / D%","Tahmini Kazanan","Gerçek Kazanan","Doğru?","Log Loss"]
display_cols = [c for c in display_cols if c in table_df.columns]

st.dataframe(
    table_df[display_cols].rename(columns={"match_id": "ID", "group": "Grup"}),
    use_container_width=True,
    hide_index=True,
)

# ── Grup bazlı performans ─────────────────────────────────────────────────────
if "group" in merged.columns:
    with st.expander("📋 Grup Bazlı Performans Özeti"):
        grp_stats = (
            merged.groupby("group")
            .agg(
                Maçlar=("correct", "count"),
                Doğru=("correct", "sum"),
                Accuracy=("correct", "mean"),
                LogLoss=("log_loss", "mean"),
                Brier=("brier", "mean"),
            )
            .reset_index()
        )
        grp_stats["Accuracy"] = (grp_stats["Accuracy"] * 100).round(1).astype(str) + "%"
        grp_stats["LogLoss"]  = grp_stats["LogLoss"].round(3)
        grp_stats["Brier"]    = grp_stats["Brier"].round(3)
        grp_stats = grp_stats.rename(columns={"group": "Grup"})
        st.dataframe(grp_stats, use_container_width=True, hide_index=True)
