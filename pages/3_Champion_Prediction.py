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

    # ── Grafik 1: Konfederasyon Treemap ──────────────────────────────────────
    st.markdown("#### Konfederasyon Bazlı Şampiyonluk Dağılımı")

    CONFEDERATION = {
        # UEFA
        "France": "UEFA", "Germany": "UEFA", "Spain": "UEFA", "England": "UEFA",
        "Portugal": "UEFA", "Netherlands": "UEFA", "Belgium": "UEFA", "Italy": "UEFA",
        "Croatia": "UEFA", "Denmark": "UEFA", "Austria": "UEFA", "Switzerland": "UEFA",
        "Serbia": "UEFA", "Ukraine": "UEFA", "Poland": "UEFA", "Türkiye": "UEFA",
        "Turkey": "UEFA", "Hungary": "UEFA", "Slovakia": "UEFA", "Slovenia": "UEFA",
        "Albania": "UEFA", "Scotland": "UEFA", "Romania": "UEFA", "Czechia": "UEFA",
        "Czech Republic": "UEFA", "Wales": "UEFA", "Georgia": "UEFA",
        "North Macedonia": "UEFA", "Kosovo": "UEFA", "Bosnia and Herzegovina": "UEFA",
        "Montenegro": "UEFA", "Finland": "UEFA", "Norway": "UEFA", "Sweden": "UEFA",
        # CONMEBOL
        "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Uruguay": "CONMEBOL",
        "Colombia": "CONMEBOL", "Chile": "CONMEBOL", "Ecuador": "CONMEBOL",
        "Peru": "CONMEBOL", "Paraguay": "CONMEBOL", "Venezuela": "CONMEBOL",
        "Bolivia": "CONMEBOL",
        # CONCACAF
        "United States": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
        "Costa Rica": "CONCACAF", "Jamaica": "CONCACAF", "Honduras": "CONCACAF",
        "El Salvador": "CONCACAF", "Panama": "CONCACAF", "Trinidad & Tobago": "CONCACAF",
        "Haiti": "CONCACAF",
        # AFC
        "Japan": "AFC", "South Korea": "AFC", "Iran": "AFC", "Saudi Arabia": "AFC",
        "Australia": "AFC", "Qatar": "AFC", "Iraq": "AFC", "Jordan": "AFC",
        "Uzbekistan": "AFC", "China PR": "AFC", "China": "AFC",
        # CAF
        "Morocco": "CAF", "Senegal": "CAF", "Nigeria": "CAF", "Cameroon": "CAF",
        "Ghana": "CAF", "Egypt": "CAF", "Côte d'Ivoire": "CAF", "Mali": "CAF",
        "Algeria": "CAF", "Tunisia": "CAF", "South Africa": "CAF",
        # OFC
        "New Zealand": "OFC",
    }
    CONF_COLORS = {
        "UEFA":     "#1f77b4",
        "CONMEBOL": "#2ca02c",
        "CONCACAF": "#d62728",
        "AFC":      "#ff7f0e",
        "CAF":      "#9467bd",
        "OFC":      "#8c564b",
    }

    top24 = sim_df.head(24).copy()
    top24["p_champion_pct"] = (top24["p_champion"] * 100).round(2)
    top24["confederation"] = top24["team"].map(lambda t: CONFEDERATION.get(t, "Diğer"))

    fig_tree = px.treemap(
        top24,
        path=["confederation", "team"],
        values="p_champion_pct",
        color="confederation",
        color_discrete_map=CONF_COLORS,
        title="Şampiyonluk Olasılığı — Konfederasyon Dağılımı",
        labels={"p_champion_pct": "P(Şampiyon) %"},
    )
    fig_tree.update_layout(
        height=480,
        paper_bgcolor="#0E1117",
        font_color="white",
        margin={"t": 50, "b": 10, "l": 10, "r": 10},
    )
    fig_tree.update_traces(textinfo="label+value")
    st.plotly_chart(fig_tree, use_container_width=True)

    # ── Grafik 2: Elo vs P(Şampiyon) Scatter ────────────────────────────────
    st.markdown("#### Elo Derecesi vs Şampiyonluk Olasılığı")
    if elo_map:
        scatter_df = sim_df.head(32).copy()
        scatter_df["elo"] = scatter_df["team"].map(lambda t: elo_map.get(t, None))
        scatter_df["p_champion_pct"] = (scatter_df["p_champion"] * 100).round(2)
        scatter_df["confederation"] = scatter_df["team"].map(
            lambda t: CONFEDERATION.get(t, "Diğer")
        )
        scatter_df = scatter_df.dropna(subset=["elo"])

        fig_sc = px.scatter(
            scatter_df,
            x="elo",
            y="p_champion_pct",
            text="team",
            color="confederation",
            color_discrete_map=CONF_COLORS,
            trendline="ols",
            labels={"elo": "Elo Derecesi", "p_champion_pct": "P(Şampiyon) %",
                    "confederation": "Konfederasyon"},
            title="Elo vs Şampiyonluk Olasılığı (trend üzerindekiler = simülasyonda öne çıkanlar)",
        )
        fig_sc.update_traces(
            textposition="top center",
            selector={"mode": "markers+text"},
        )
        fig_sc.update_layout(
            height=520,
            paper_bgcolor="#0E1117",
            plot_bgcolor="#1a1a2e",
            font_color="white",
            margin={"t": 50, "b": 10, "l": 10, "r": 10},
        )
        st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.info("Elo verisi yüklenemedi; scatter grafiği atlandı.")

    # ── Grafik 3: Top-8 Takım Eleme Yolculuğu ───────────────────────────────
    st.markdown("#### Top 8 Takım — Eleme Turu Yolculuğu")

    _stages = [
        ("p_round32",  "Son 32"),
        ("p_round16",  "Son 16"),
        ("p_quarter",  "Çeyrek"),
        ("p_semi",     "Yarı Final"),
        ("p_finalist", "Final"),
        ("p_champion", "Şampiyon"),
    ]
    stage_cols = [col for col, _ in _stages if col in sim_df.columns]
    stage_labels = [lbl for col, lbl in _stages if col in sim_df.columns]

    top8 = sim_df.head(8).copy()
    top8_long = top8.melt(
        id_vars="team",
        value_vars=stage_cols,
        var_name="round_col",
        value_name="probability",
    )
    label_map = {col: lbl for col, lbl in _stages}
    top8_long["round_label"] = top8_long["round_col"].map(label_map)
    top8_long["probability_pct"] = (top8_long["probability"] * 100).round(1)
    top8_long["round_order"] = top8_long["round_col"].map(
        {col: i for i, (col, _) in enumerate(_stages)}
    )
    top8_long = top8_long.sort_values("round_order")

    top3_teams = sim_df.head(3)["team"].tolist()

    fig_line = px.line(
        top8_long,
        x="round_label",
        y="probability_pct",
        color="team",
        markers=True,
        labels={"round_label": "Tur", "probability_pct": "Olasılık %", "team": "Takım"},
        title="Top 8 Takım — Tur Bazlı Eleme Olasılıkları",
        category_orders={"round_label": stage_labels},
    )
    for trace in fig_line.data:
        if trace.name in top3_teams:
            trace.line.width = 3
        else:
            trace.line.width = 1.5
            trace.opacity = 0.6
    fig_line.update_layout(
        height=420,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#1a1a2e",
        font_color="white",
        margin={"t": 50, "b": 10, "l": 10, "r": 10},
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # ── Kura Zorluğu Analizi ──────────────────────────────────────────────────
    with st.expander("🎲 Kura Zorluğu Analizi"):
        if elo_map and fixtures is not None:
            # Her takım için grup rakiplerinin listesini çıkar
            team_opponents: dict = {}
            team_group: dict = {}
            for _, row in fixtures.iterrows():
                ht = str(row["home_team"])
                at = str(row["away_team"])
                grp = str(row["group"])
                team_group[ht] = grp
                team_group[at] = grp
                team_opponents.setdefault(ht, []).append(at)
                team_opponents.setdefault(at, []).append(ht)

            # Ortalama rakip Elo
            diff_records = []
            for team, opps in team_opponents.items():
                avg_elo = sum(elo_map.get(o, 1700) for o in opps) / len(opps)
                diff_records.append({
                    "team":         team,
                    "group":        team_group.get(team, "?"),
                    "avg_opp_elo":  round(avg_elo, 1),
                })

            diff_df = (
                pd.DataFrame(diff_records)
                .sort_values("avg_opp_elo", ascending=False)
                .reset_index(drop=True)
            )

            st.markdown(
                "Grup rakiplerinin **ortalama Elo'su** — yüksek = zor kura, "
                "düşük = kolay kura. Renk skalası: 🟢 kolay → 🔴 zor."
            )

            fig_diff = px.bar(
                diff_df,
                x="avg_opp_elo",
                y="team",
                orientation="h",
                color="avg_opp_elo",
                color_continuous_scale="RdYlGn_r",
                text=diff_df["avg_opp_elo"].astype(str),
                hover_data=["group"],
                labels={"avg_opp_elo": "Ort. Rakip Elo", "team": "Takım", "group": "Grup"},
            )
            fig_diff.update_layout(
                height=max(600, len(diff_df) * 18),
                yaxis={"autorange": "reversed"},
                showlegend=False,
                coloraxis_showscale=True,
                margin={"t": 20, "b": 10, "l": 10, "r": 80},
                paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117",
                font_color="white",
                xaxis_title="Ortalama Rakip Elo (Yüksek = Zor Kura)",
            )
            fig_diff.update_traces(textposition="outside")
            st.plotly_chart(fig_diff, use_container_width=True)

            # Grup bazlı güçlük özeti
            st.markdown("**Grup Bazlı Ortalama Güçlük** (tüm takımların rakip Elo ortalaması)")
            grp_summary = (
                diff_df.groupby("group")["avg_opp_elo"]
                .mean()
                .round(1)
                .reset_index()
                .sort_values("avg_opp_elo", ascending=False)
                .rename(columns={"group": "Grup", "avg_opp_elo": "Ort. Rakip Elo"})
            )
            st.dataframe(grp_summary, use_container_width=True, hide_index=True)
        else:
            st.info("Elo verisi veya fikstür verisi yüklenemedi.")

    # ── Faktörler expander ────────────────────────────────────────────────────
    with st.expander("⚙️ Simülasyon Faktörleri"):
        st.markdown("""
**0. Maç Bazlı Elo Gürültüsü** — Her maçta her takımın Elo'suna ~N(0, 100) rastgele gürültü eklenir.
Bu, gerçek futbolun tahmin edilemezliğini yansıtır; 200 puan favorinin bile her maçta sürpriz riski vardır.
Ayrıca Elo ölçeği 400→500'e genişletildi — 200 Elo fark artık %76 değil %72 olasılık.

**1. WC Form Boost** — Grup aşamasında kazanan takımlar eleme turunda +Elo bonusu alır.
Her puan farkı ±8 Elo (maks ±50).

**2. Ev Sahibi Ülke Kalabalık Etkisi** — ABD (+40), Meksika (+50), Kanada (+35) Elo bonusu,
kendi stadyumlarında oynarken.

**3. Penaltı İstatistikleri** — Çok yakın maçlarda (win prob 0.40–0.60),
tarihsel penaltı kazanma oranı kazananı hafifçe etkiler (ağırlık: %15).

**4. Dinlenme Günü / Yorgunluk** — 4 günden az dinlenme = her gün için -8 Elo (maks -24).

**5. İrtifa Adaptasyonu** — Azteca (2240m) ve Denver (1600m) gibi yüksek rakımlı stadlarda,
yüksek rakımlı ülkelerden (Meksika, Kolombiya, Ekvador vb.) gelen takımlar avantaj sağlar.

**6. Sıcaklık Stresi** — 26°C üzerinde oynanan maçlarda (Houston 32°C, Dallas 32°C vb.),
serin iklimli Avrupalı takımlar her 1°C için -3 Elo cezası alır (maks -24 Elo).
Brezilya, Arjantin, Fas, Senegal gibi sıcak iklimli takımlar etkilenmez.

**7. Seyahat Yorgunluğu** — 9,000 km'den uzun uçuş gerektiren maçlarda takım performansı düşer.
Her 1,000 km (eşik üzerinde) = -4 Elo (maks -20 Elo). Japonya/Güney Kore ABD'de dezavantajlı.
        """)

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
