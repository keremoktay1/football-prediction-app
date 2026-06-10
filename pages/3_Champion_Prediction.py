"""
3_Champion_Prediction.py — Monte Carlo Şampiyonluk Tahmini.

Özellikler:
  - "Simülasyonu Çalıştır" butonu: 5,000 Monte Carlo turu
  - P(şampiyon), P(finalist), P(yarı finalist), P(çeyrek final) per takım
  - Skor girildikten sonra re-run ile güncel olasılıklar
  - Son simülasyon zamanı ve kaç maç oynanmıştı bilgisi
"""
import os
import sys
import streamlit as st
import pandas as pd

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(APP_DIR, "src"))

from data_loader import (
    load_fixtures, load_knockout_slots, load_match_updates,
    load_predictions, load_elo_ratings, build_elo_map,
)
from tournament_simulator import (
    run_simulation, save_simulation, load_simulation,
)

# ── Sayfa yapılandırması ─────────────────────────────────────────────────────
st.set_page_config(page_title="Şampiyonluk Tahmini", page_icon="🏆", layout="wide")
st.title("🏆 Monte Carlo Şampiyonluk Tahmini")
st.markdown(
    "Her simülasyonda oynanmamış maçlar olasılıkla simüle edilir, "
    "eleme turu Elo ile ilerler. "
    "Gerçek skorları girdikçe **Yeniden Çalıştır** butonuna basın."
)

# ── Veri yükle ───────────────────────────────────────────────────────────────
try:
    fixtures    = load_fixtures()
    ko_slots    = load_knockout_slots()
    updates     = load_match_updates()
    predictions = load_predictions()
    elo_df      = load_elo_ratings()
    elo_map     = build_elo_map(elo_df)
except Exception as exc:
    st.error(f"Veri yükleme hatası: {exc}")
    st.stop()

if fixtures is None:
    st.error("❌ GROUP_FIXTURES.CSV bulunamadı.")
    st.stop()
if ko_slots is None:
    st.error("❌ KNOCKOUT_SLOTS.CSV bulunamadı.")
    st.stop()

played_count = len(updates) if (updates is not None and not updates.empty) else 0

# ── Kaydedilmiş simülasyon ────────────────────────────────────────────────────
sim_df, sim_meta = load_simulation()

st.markdown("---")
mcol1, mcol2 = st.columns([3, 1])

with mcol1:
    if sim_meta:
        st.info(
            f"📅 Son simülasyon: `{sim_meta.get('timestamp','?')}` | "
            f"{sim_meta.get('n_simulations',0):,} tur | "
            f"O sırada {sim_meta.get('played_matches',0)} maç oynandı"
        )
    else:
        st.warning("⚠️ Henüz simülasyon çalıştırılmadı. Aşağıdaki butona basın.")

with mcol2:
    n_sim_choice = st.selectbox(
        "Simülasyon sayısı",
        options=[1_000, 5_000, 10_000],
        index=1,
        key="n_sim",
    )

if played_count > 0:
    update_btn = st.button(
        "🔄  Tahminleri Güncelle (oynanan maçlara göre)",
        use_container_width=True,
    )
    if update_btn:
        import subprocess
        result = subprocess.run(
            ["python", "scripts/update_tournament_predictions.py"],
            capture_output=True, text=True, cwd=APP_DIR
        )
        if result.returncode == 0:
            st.success("✅ Tahminler güncellendi!")
            predictions = load_predictions()  # yeniden yükle
        else:
            st.error(f"Hata: {result.stderr}")

run_btn = st.button(
    "▶️  Simülasyonu Çalıştır / Yeniden Çalıştır",
    use_container_width=True,
    type="primary",
)

if run_btn:
    with st.spinner(f"{n_sim_choice:,} Monte Carlo simülasyonu çalışıyor..."):
        try:
            sim_df = run_simulation(
                fixtures=fixtures,
                knockout_slots=ko_slots,
                predictions=predictions,
                updates=updates,
                elo_map=elo_map,
                n_simulations=n_sim_choice,
                seed=None,   # Her çalıştırmada farklı random tohum → gerçek Monte Carlo
            )
            save_simulation(sim_df, n_sim_choice, played_count)
            st.success(f"✅ {n_sim_choice:,} simülasyon tamamlandı!")
        except Exception as exc:
            st.error(f"Simülasyon hatası: {exc}")
            st.stop()

if sim_df is None or sim_df.empty:
    st.info("Simülasyon verisi yok. Butona basın.")
    st.stop()

st.markdown("---")

# ── Özet metrikler ────────────────────────────────────────────────────────────
top3 = sim_df.head(3)
c1, c2, c3 = st.columns(3)
medals = ["🥇", "🥈", "🥉"]
for col, medal, (_, row) in zip([c1, c2, c3], medals, top3.iterrows()):
    with col:
        st.metric(
            label=f"{medal} {row['team']}",
            value=f"{row['p_champion']:.1%}",
            help=f"Finalist: {row['p_finalist']:.1%} | "
                 f"Yarı: {row['p_semi']:.1%} | "
                 f"Çeyrek: {row['p_quarter']:.1%}",
        )

st.markdown("---")

# ── Şampiyonluk ihtimali bar chart ───────────────────────────────────────────
st.markdown("#### Şampiyonluk Olasılığı — İlk 16")

try:
    import plotly.express as px

    top16 = sim_df.head(16).copy()
    top16["label"] = (top16["p_champion"] * 100).round(1).astype(str) + "%"

    fig = px.bar(
        top16,
        x="p_champion",
        y="team",
        orientation="h",
        text="label",
        color="p_champion",
        color_continuous_scale="Blues",
        labels={"p_champion": "P(Şampiyon)", "team": "Takım"},
    )
    fig.update_layout(
        height=500,
        yaxis={"autorange": "reversed"},
        showlegend=False,
        coloraxis_showscale=False,
        margin={"t": 20, "b": 10, "l": 10, "r": 10},
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font_color="white",
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    # ── Aşama × Takım olasılık ısı haritası ──────────────────────────────────
    st.markdown("#### Turnuva Eleme Haritası — İlk 20 Takım")
    import plotly.graph_objects as go

    _stages = [
        ("p_round32",  "Son 32"),
        ("p_round16",  "Son 16"),
        ("p_quarter",  "Çeyrek"),
        ("p_semi",     "Yarı Final"),
        ("p_finalist", "Final"),
        ("p_champion", "Şampiyon"),
    ]
    avail = [(col, lbl) for col, lbl in _stages if col in sim_df.columns]
    if avail:
        top20 = sim_df.head(20).copy()
        z = (top20[[col for col, _ in avail]].values * 100).round(1)
        text_z = [[f"{v:.1f}%" for v in row] for row in z]
        fig_heat = go.Figure(go.Heatmap(
            z=z,
            x=[lbl for _, lbl in avail],
            y=top20["team"].tolist(),
            colorscale="Blues",
            text=text_z,
            texttemplate="%{text}",
            textfont={"size": 10},
            showscale=False,
        ))
        fig_heat.update_layout(
            height=560,
            yaxis={"autorange": "reversed"},
            margin={"t": 10, "b": 10, "l": 10, "r": 10},
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font_color="white",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

except ImportError:
    # Plotly yoksa tablo göster
    top16 = sim_df.head(16).copy()
    top16["p_champion"] = (top16["p_champion"] * 100).round(1).astype(str) + "%"
    st.dataframe(top16[["team", "p_champion"]], use_container_width=True, hide_index=True)

st.markdown("---")

# ── Detaylı tablo ─────────────────────────────────────────────────────────────
st.markdown("#### Tüm Turnuva Olasılıkları")

display = sim_df.copy()
for col in ["p_champion", "p_finalist", "p_semi", "p_quarter", "p_round16", "p_round32"]:
    if col in display.columns:
        display[col] = (display[col] * 100).round(1).astype(str) + "%"

col_labels = {
    "team":       "Takım",
    "p_champion": "🏆 Şampiyon",
    "p_finalist": "🥈 Finalist",
    "p_semi":     "4'lü Final",
    "p_quarter":  "Çeyrek",
    "p_round16":  "Son 16",
    "p_round32":  "Son 32",
}
show_cols = [c for c in col_labels if c in display.columns]
st.dataframe(
    display[show_cols].rename(columns=col_labels),
    use_container_width=True,
    hide_index=True,
)

# ── Grup bazlı şampiyonluk ────────────────────────────────────────────────────
if fixtures is not None:
    with st.expander("📊 Grup Bazlı Şampiyonluk Olasılıkları"):
        all_teams_in_group = (
            pd.concat([
                fixtures[["group", "home_team"]].rename(columns={"home_team": "team"}),
                fixtures[["group", "away_team"]].rename(columns={"away_team": "team"}),
            ])
            .drop_duplicates()
        )
        merged = all_teams_in_group.merge(
            sim_df[["team", "p_champion", "p_finalist"]],
            on="team",
            how="left",
        ).fillna(0)
        merged = merged.sort_values(["group", "p_champion"], ascending=[True, False])

        groups_sorted = sorted(merged["group"].unique())
        cols_per_row = 3
        for i in range(0, len(groups_sorted), cols_per_row):
            row_grps = groups_sorted[i: i + cols_per_row]
            tcols = st.columns(len(row_grps))
            for tcol, g in zip(tcols, row_grps):
                with tcol:
                    sub = merged[merged["group"] == g][["team", "p_champion"]].copy()
                    sub["p_champion"] = (sub["p_champion"] * 100).round(1).astype(str) + "%"
                    st.markdown(f"**Grup {g}**")
                    st.dataframe(
                        sub.rename(columns={"team": "Takım", "p_champion": "🏆%"}),
                        use_container_width=True,
                        hide_index=True,
                    )

st.markdown("---")
st.caption(
    f"Simülasyon: {sim_meta.get('n_simulations', '?'):,} tur | "
    f"Yöntem: Grup aşaması → Poisson örneklemesi, "
    f"Eleme aşaması → Elo tabanlı"
    if sim_meta else "Simülasyon henüz çalıştırılmadı."
)
