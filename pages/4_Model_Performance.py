"""
4_Model_Performance.py — Model Karşılaştırması & Canlı Doğruluk Takibi.

Bölümler:
  1. Test seti karşılaştırma tablosu + bar chart (6 model)
  2. Valid vs Test: Log Loss / Brier / Accuracy kıyası
  3. Canlı doğruluk: girilen gerçek maç sonuçları üzerinden kümülatif metrikler
  4. Maç bazlı detay tablosu
"""
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(APP_DIR, "src"))

from data_loader import (
    load_fixtures,
    load_match_updates,
    load_predictions,
    load_model_comparison,
)

st.set_page_config(page_title="Model Performansı", page_icon="📊", layout="wide")
st.title("📊 Model Karşılaştırması & Performans Takibi")

# ── Veri yükle ───────────────────────────────────────────────────────────────
fixtures    = load_fixtures()
updates     = load_match_updates()
predictions = load_predictions()
comp_df     = load_model_comparison()

MODEL_COLORS = {
    "Elo Baseline":  "#8c8c8c",
    "Poisson":       "#4e9af1",
    "LR":            "#f4a442",
    "Ensemble":      "#2ecc71",
    "Random Forest": "#9b59b6",
    "XGBoost":       "#e74c3c",
}

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 1 — Test seti karşılaştırması
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 1 · Test Seti Karşılaştırması  (2022 → bugün)")

if comp_df is None:
    st.warning(
        "model_comparison.csv bulunamadı. "
        "`scripts/fast_model_training.py` çalıştırın."
    )
else:
    test_df = comp_df[comp_df["split"] == "test"].copy()

    if not test_df.empty:
        best_ll    = test_df.loc[test_df["log_loss"].idxmin()]
        best_acc   = test_df.loc[test_df["accuracy"].idxmax()]
        best_brier = test_df.loc[test_df["brier"].idxmin()]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test Örnekleri", f"{int(test_df['n'].iloc[0]):,}")
        c2.metric("En Düşük Log Loss",  f"{best_ll['log_loss']:.4f}",
                  delta=best_ll["model"], delta_color="off")
        c3.metric("En Yüksek Accuracy", f"{best_acc['accuracy']*100:.1f}%",
                  delta=best_acc["model"], delta_color="off")
        c4.metric("En Düşük Brier",     f"{best_brier['brier']:.4f}",
                  delta=best_brier["model"], delta_color="off")

    st.markdown("---")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        test_sorted  = test_df.sort_values("log_loss")
        models       = test_sorted["model"].tolist()
        colors       = [MODEL_COLORS.get(m, "#888") for m in models]
        acc_sorted   = test_df.sort_values("accuracy", ascending=False)
        brier_sorted = test_df.sort_values("brier")

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=["Log Loss ↓", "Accuracy % ↑", "Brier Score ↓"],
        )
        fig.add_trace(go.Bar(
            x=models, y=test_sorted["log_loss"].tolist(),
            marker_color=colors,
            text=[f"{v:.4f}" for v in test_sorted["log_loss"]],
            textposition="outside", name="Log Loss",
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=acc_sorted["model"].tolist(),
            y=(acc_sorted["accuracy"] * 100).tolist(),
            marker_color=[MODEL_COLORS.get(m, "#888") for m in acc_sorted["model"]],
            text=[f"{v:.1f}%" for v in acc_sorted["accuracy"] * 100],
            textposition="outside", name="Accuracy",
        ), row=1, col=2)
        fig.add_trace(go.Bar(
            x=brier_sorted["model"].tolist(),
            y=brier_sorted["brier"].tolist(),
            marker_color=[MODEL_COLORS.get(m, "#888") for m in brier_sorted["model"]],
            text=[f"{v:.4f}" for v in brier_sorted["brier"]],
            textposition="outside", name="Brier",
        ), row=1, col=3)
        fig.update_layout(
            height=380, showlegend=False,
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            font_color="white",
            margin=dict(t=50, b=20, l=10, r=10),
        )
        fig.update_yaxes(gridcolor="#333")
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.info("Plotly kurulu değil — tablo gösteriliyor.")

    # Detay tablosu
    display_test = test_df.sort_values("log_loss").copy()
    display_test["accuracy_pct"] = (display_test["accuracy"] * 100).round(1).astype(str) + "%"
    rename_map = {
        "model":        "Model",
        "log_loss":     "Log Loss ↓",
        "accuracy_pct": "Accuracy ↑",
        "brier":        "Brier ↓",
        "mce":          "Kalibrasyon Hatası ↓",
        "n":            "Örnek",
    }
    show_cols = [c for c in rename_map if c in display_test.columns]
    st.dataframe(
        display_test[show_cols].rename(columns=rename_map),
        use_container_width=True, hide_index=True,
    )

    # ── Valid vs Test karşılaştırması ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 2 · Validation vs Test — Log Loss")

    try:
        pivot = comp_df.pivot(index="model", columns="split",
                              values="log_loss").reset_index()
        pivot = pivot.sort_values("test" if "test" in pivot.columns
                                  else pivot.columns[-1])

        fig2 = go.Figure()
        for split, color in [("valid", "#4e9af1"), ("test", "#e74c3c")]:
            if split in pivot.columns:
                fig2.add_trace(go.Bar(
                    name=split.capitalize(),
                    x=pivot["model"], y=pivot[split],
                    marker_color=color,
                    text=[f"{v:.4f}" for v in pivot[split]],
                    textposition="outside",
                ))
        fig2.update_layout(
            title="Log Loss: Validation vs Test", barmode="group",
            height=320,
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            font_color="white",
            margin=dict(t=50, b=10, l=10, r=10),
            yaxis_gridcolor="#333",
        )
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as exc:
        st.warning(f"Karşılaştırma grafiği oluşturulamadı: {exc}")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3 — Canlı Doğruluk
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 3 · Canlı Tahmin Doğruluğu")

if fixtures is None or predictions is None:
    st.info("Fikstür veya tahmin verisi bulunamadı.")
else:
    update_dict = {}
    if updates is not None and not updates.empty:
        for _, r in updates.iterrows():
            try:
                update_dict[int(r["match_id"])] = (int(r["home_score"]),
                                                    int(r["away_score"]))
            except (ValueError, TypeError):
                pass

    if not update_dict:
        st.info("Henüz skor girilmedi. Fikstür sayfasından maç sonuçlarını girin.")
    else:
        pred_sub = predictions[["match_id", "home_team", "away_team",
                                 "p_home", "p_draw", "p_away"]].copy()
        played_df = fixtures[fixtures["match_id"].isin(update_dict)].merge(
            pred_sub, on="match_id", how="left"
        )

        rows = []
        for _, m in played_df.iterrows():
            mid = int(m["match_id"])
            if mid not in update_dict:
                continue
            hs, as_ = update_dict[mid]
            actual  = "H" if hs > as_ else ("D" if hs == as_ else "A")
            ph  = float(m["p_home"]) if pd.notna(m.get("p_home")) else 1 / 3
            pd_ = float(m["p_draw"]) if pd.notna(m.get("p_draw")) else 1 / 3
            pa  = float(m["p_away"]) if pd.notna(m.get("p_away")) else 1 / 3

            pred_key = "H" if ph >= pd_ and ph >= pa else ("D" if pd_ >= pa else "A")
            correct  = pred_key == actual
            ll_match = -np.log(max({"H": ph, "D": pd_, "A": pa}[actual], 1e-10))

            rows.append({
                "match_id": mid,
                "Maç":      f"{m['home_team']} – {m['away_team']}",
                "H%":       f"{ph:.0%}",
                "B%":       f"{pd_:.0%}",
                "D%":       f"{pa:.0%}",
                "Tahmini":  {"H": m["home_team"], "D": "Bera.", "A": m["away_team"]}[pred_key],
                "Gerçek":   actual,
                "Skor":     f"{hs}–{as_}",
                "Doğru?":   "✅" if correct else "❌",
                "Log Loss": round(ll_match, 3),
                "_correct": correct,
            })

        if rows:
            live_df   = pd.DataFrame(rows)
            n_played  = len(live_df)
            n_correct = int(live_df["_correct"].sum())
            avg_ll    = float(live_df["Log Loss"].mean())

            lc1, lc2, lc3 = st.columns(3)
            lc1.metric("Oynanan Maç", n_played)
            lc2.metric("Doğru Tahmin",
                       f"{n_correct}/{n_played}  ({n_correct/n_played*100:.0f}%)")
            lc3.metric("Ort. Log Loss", f"{avg_ll:.3f}")

            try:
                import plotly.express as px

                live_sorted = live_df.sort_values("match_id").reset_index(drop=True)
                live_sorted["cum_acc"] = (live_sorted["_correct"]
                                          .expanding().mean() * 100)
                fig3 = px.line(
                    live_sorted,
                    x=live_sorted.index + 1, y="cum_acc",
                    labels={"x": "Maç Sırası", "cum_acc": "Kümülatif Doğruluk %"},
                    title="Kümülatif Tahmin Doğruluğu",
                    markers=True,
                )
                fig3.update_layout(
                    height=260,
                    paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                    font_color="white",
                    yaxis=dict(range=[0, 100], gridcolor="#333"),
                    xaxis_gridcolor="#333",
                    margin=dict(t=40, b=10, l=10, r=10),
                )
                fig3.add_hline(y=33.3, line_dash="dot", line_color="#888",
                               annotation_text="Rastgele (33%)")
                st.plotly_chart(fig3, use_container_width=True)
            except ImportError:
                pass

            st.markdown("#### Maç Bazlı Detay")
            show = ["Maç", "H%", "B%", "D%", "Tahmini", "Gerçek",
                    "Skor", "Doğru?", "Log Loss"]
            st.dataframe(live_df[show], use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "Log Loss: düşük = iyi (güvenli & doğru tahmin) · "
    "Brier: ortalama kare hata · "
    "Kalibrasyon Hatası: tahmin olasılığı ↔ gerçekleşme oranı sapması · "
    "Test seti: 2022-01-01 → bugün (~4.500 maç)"
)
