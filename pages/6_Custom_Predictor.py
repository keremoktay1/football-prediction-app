"""
6_Custom_Predictor.py — Geçişli Tahmin & Derinlemesine Maç Analizi

Paneller:
  1. Doğrudan Tahmin (model veya Elo tabanlı)
  2. Ortak Rakip Zinciri (geçişli analiz)
  3. H2H Geçmiş (son 10 karşılaşma)
  4. Kadro Karşılaştırması
  5. Olasılıklı Sonuç (ağırlıklı final tahmin)
"""
from __future__ import annotations

import ast
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(APP_DIR, "src"))

from config import FILES, TEAM_NAME_MAP
from data_loader import (
    load_fixtures,
    load_predictions,
    load_elo_ratings,
    build_elo_map,
)
from prediction_engine import _elo_to_3way, _estimate_xg, poisson_score_table

# ── Sayfa yapılandırması ─────────────────────────────────────────────────────
st.set_page_config(page_title="Custom Predictor", page_icon="🔭", layout="wide")
st.title("🔭 Özel Maç Tahmini")
st.caption("Herhangi iki takım arasında geçişli analiz, H2H ve kadro karşılaştırması.")

PROCESSED_DIR = os.path.join(APP_DIR, "data", "processed")

# ── Veri yükleme ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_results() -> pd.DataFrame:
    path = FILES.get("results", "")
    if not os.path.isfile(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"], low_memory=False)
    for col in ["home_team", "away_team"]:
        df[col] = df[col].replace(TEAM_NAME_MAP)
    return df

@st.cache_data(ttl=600)
def load_squad_stats() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "squad_stats.csv")
    if os.path.isfile(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data(ttl=600)
def load_clusters() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "team_clusters.csv")
    if os.path.isfile(path):
        return pd.read_csv(path)
    return pd.DataFrame()

try:
    fixtures    = load_fixtures()
    predictions = load_predictions()
    elo_df      = load_elo_ratings()
    elo_map     = build_elo_map(elo_df)
    results_df  = load_results()
    squad_df    = load_squad_stats()
    clusters_df = load_clusters()
except Exception as exc:
    st.error(f"Veri yükleme hatası: {exc}")
    st.stop()

# ── Takım listesi ─────────────────────────────────────────────────────────────
fixture_teams: list[str] = []
if fixtures is not None:
    fixture_teams = sorted(
        t for t in set(fixtures["home_team"]).union(fixtures["away_team"])
        if not str(t).startswith(("UEFA", "FIFA", "Winner", "Runner", "Best"))
    )

elo_teams = sorted(elo_map.keys())
all_teams = sorted(set(fixture_teams) | set(elo_teams))

if not all_teams:
    st.error("Takım listesi oluşturulamadı.")
    st.stop()

# ── Takım seçimi ─────────────────────────────────────────────────────────────
col_a, col_vs, col_b = st.columns([5, 1, 5])
with col_a:
    default_a = all_teams.index("Turkey") if "Turkey" in all_teams else 0
    team_a = st.selectbox("Takım A", all_teams, index=default_a, key="team_a_sel")
with col_vs:
    st.markdown("<br><h3 style='text-align:center'>vs</h3>", unsafe_allow_html=True)
with col_b:
    default_b = all_teams.index("USA") if "USA" in all_teams else min(1, len(all_teams)-1)
    team_b = st.selectbox("Takım B", all_teams, index=default_b, key="team_b_sel")

if team_a == team_b:
    st.warning("Lütfen farklı iki takım seçin.")
    st.stop()

st.markdown("---")

# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

def avg_gd_vs_opp(team: str, opp: str, df: pd.DataFrame) -> float:
    """Takımın belirtilen rakibe karşı ortalama gol farkı."""
    home = df[(df["home_team"] == team) & (df["away_team"] == opp)].copy()
    away = df[(df["away_team"] == team) & (df["home_team"] == opp)].copy()

    diffs = []
    for _, r in home.iterrows():
        try:
            diffs.append(int(r["home_score"]) - int(r["away_score"]))
        except Exception:
            pass
    for _, r in away.iterrows():
        try:
            diffs.append(int(r["away_score"]) - int(r["home_score"]))
        except Exception:
            pass

    return float(np.mean(diffs)) if diffs else 0.0


def get_cluster_label(team: str) -> str:
    if clusters_df.empty:
        return "—"
    row = clusters_df[clusters_df["team"] == team]
    if row.empty:
        return "—"
    return str(row.iloc[0]["cluster_label"])


def get_squad_row(team: str) -> dict:
    if squad_df.empty:
        return {}
    row = squad_df[squad_df["team"] == team]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def get_direct_prediction(ta: str, tb: str) -> dict:
    """Model veya ELO tabanlı tahmin döner."""
    if predictions is not None and not predictions.empty:
        mask = (
            (predictions["home_team"] == ta) & (predictions["away_team"] == tb)
        ) | (
            (predictions["home_team"] == tb) & (predictions["away_team"] == ta)
        )
        rows = predictions[mask]
        if not rows.empty:
            r = rows.iloc[0]
            flipped = r["home_team"] != ta
            ph = float(r["p_away"]) if flipped else float(r["p_home"])
            pd_ = float(r["p_draw"])
            pa = float(r["p_home"]) if flipped else float(r["p_away"])
            lh = float(r.get("lambda_home", np.nan) or np.nan)
            la = float(r.get("lambda_away", np.nan) or np.nan)
            if flipped:
                lh, la = la, lh
            return {
                "p_home": ph, "p_draw": pd_, "p_away": pa,
                "lambda_home": lh, "lambda_away": la,
                "upset_risk": float(r.get("upset_risk", 0.5) or 0.5),
                "source": "model",
            }

    # ELO fallback
    elo_a = elo_map.get(ta, 1700.0)
    elo_b = elo_map.get(tb, 1700.0)
    ph, pd_, pa = _elo_to_3way(elo_a, elo_b, neutral=True)
    lh, la = _estimate_xg(elo_a, elo_b)
    return {
        "p_home": ph, "p_draw": pd_, "p_away": pa,
        "lambda_home": lh, "lambda_away": la,
        "upset_risk": 0.5,
        "source": "elo",
    }

# ── Recent results (son 4 yıl) ────────────────────────────────────────────────
if not results_df.empty:
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=4)
    recent = results_df[results_df["date"] >= cutoff].copy()
else:
    recent = pd.DataFrame()

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 1 — Doğrudan Tahmin
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("1️⃣ Doğrudan Tahmin")

direct = get_direct_prediction(team_a, team_b)
elo_a  = elo_map.get(team_a, 1700)
elo_b  = elo_map.get(team_b, 1700)
cl_a   = get_cluster_label(team_a)
cl_b   = get_cluster_label(team_b)

p1, p2, p3, p4, p5, p6 = st.columns(6)
p1.metric(f"ELO {team_a}", f"{int(elo_a)}")
p2.metric(f"ELO {team_b}", f"{int(elo_b)}")
p3.metric(f"{team_a} Kazanır", f"{direct['p_home']*100:.1f}%")
p4.metric("Beraberlik", f"{direct['p_draw']*100:.1f}%")
p5.metric(f"{team_b} Kazanır", f"{direct['p_away']*100:.1f}%")
p6.metric("Kaynak", direct["source"].upper())

info_parts = []
if cl_a != "—":
    info_parts.append(f"**{team_a}**: {cl_a}")
if cl_b != "—":
    info_parts.append(f"**{team_b}**: {cl_b}")
if info_parts:
    st.caption(" | ".join(info_parts))

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 2 — Ortak Rakip Zinciri (Geçişli Analiz)
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("2️⃣ Ortak Rakip Zinciri (Geçişli Analiz)")

transitive_diff = 0.0
transitive_count = 0

if recent.empty:
    st.info("Geçmiş maç verisi yüklenemedi.")
else:
    def get_opponents(team: str) -> set:
        h_opps = set(recent[recent["home_team"] == team]["away_team"].dropna())
        a_opps = set(recent[recent["away_team"] == team]["home_team"].dropna())
        return h_opps | a_opps

    opps_a = get_opponents(team_a)
    opps_b = get_opponents(team_b)
    common_opps = opps_a & opps_b
    # Çok fazla ortak rakip olursa son 4 yılda en az 2 kez oynanan rakipleri al
    if len(common_opps) > 20:
        def played_vs(team: str, opp: str) -> int:
            return len(recent[
                ((recent["home_team"]==team) & (recent["away_team"]==opp)) |
                ((recent["away_team"]==team) & (recent["home_team"]==opp))
            ])
        common_opps = {o for o in common_opps if played_vs(team_a, o) >= 2 or played_vs(team_b, o) >= 2}

    if not common_opps:
        st.info(f"Son 4 yılda {team_a} ve {team_b}'nin ortak rakibi bulunamadı.")
    else:
        chain_rows = []
        for opp in sorted(common_opps):
            if opp in (team_a, team_b):
                continue
            a_gd = avg_gd_vs_opp(team_a, opp, recent)
            b_gd = avg_gd_vs_opp(team_b, opp, recent)
            diff = round(a_gd - b_gd, 2)
            chain_rows.append({
                "Ortak Rakip": opp,
                f"{team_a} GD": f"{a_gd:+.2f}",
                f"{team_b} GD": f"{b_gd:+.2f}",
                f"Fark ({team_a} lehine)": f"{diff:+.2f}",
                "_diff_val": diff,
            })

        if chain_rows:
            chain_df = pd.DataFrame(chain_rows)
            avg_diff = float(chain_df["_diff_val"].mean())
            transitive_diff  = avg_diff
            transitive_count = len(chain_df)

            display_df = chain_df.drop(columns=["_diff_val"])
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            if avg_diff > 0.2:
                verdict = f"**{team_a}** ortak rakiplere karşı daha iyi performans → Hafif avantaj"
            elif avg_diff < -0.2:
                verdict = f"**{team_b}** ortak rakiplere karşı daha iyi performans → Hafif avantaj"
            else:
                verdict = "Ortak rakip analizi yakın sonuç gösteriyor → Belirsiz"

            avg_col = st.columns([3, 1])
            avg_col[0].markdown(f"**Ortalama fark:** `{avg_diff:+.2f}` — {verdict}")
        else:
            st.info("Geçerli ortak rakip bulunamadı.")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 3 — H2H Geçmiş
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("3️⃣ H2H Geçmiş (Son 10 Maç)")

if not results_df.empty:
    h2h = results_df[
        ((results_df["home_team"] == team_a) & (results_df["away_team"] == team_b)) |
        ((results_df["home_team"] == team_b) & (results_df["away_team"] == team_a))
    ].sort_values("date", ascending=False).head(10)

    if h2h.empty:
        st.info(f"{team_a} ile {team_b} arasında geçmiş maç kaydı bulunamadı.")
    else:
        h2h_rows = []
        for _, r in h2h.iterrows():
            is_a_home = r["home_team"] == team_a
            score_a   = int(r["home_score"]) if is_a_home else int(r["away_score"])
            score_b   = int(r["away_score"]) if is_a_home else int(r["home_score"])
            if score_a > score_b:
                result_str = f"✅ {team_a} kazandı"
            elif score_b > score_a:
                result_str = f"✅ {team_b} kazandı"
            else:
                result_str = "🤝 Beraberlik"
            h2h_rows.append({
                "Tarih": str(r["date"])[:10],
                "Turnuva": str(r.get("tournament", "—")),
                team_a: score_a,
                team_b: score_b,
                "Sonuç": result_str,
            })
        st.dataframe(pd.DataFrame(h2h_rows), use_container_width=True, hide_index=True)

        # Özet
        total = len(h2h)
        wins_a = sum(1 for rr in h2h_rows if team_a in rr["Sonuç"] and "kazandı" in rr["Sonuç"])
        wins_b = sum(1 for rr in h2h_rows if team_b in rr["Sonuç"] and "kazandı" in rr["Sonuç"])
        draws  = total - wins_a - wins_b
        hcol1, hcol2, hcol3 = st.columns(3)
        hcol1.metric(f"{team_a} Kazanma", wins_a)
        hcol2.metric("Beraberlik", draws)
        hcol3.metric(f"{team_b} Kazanma", wins_b)
else:
    st.info("Geçmiş maç verisi yüklenemedi.")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 4 — Kadro Karşılaştırması
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("4️⃣ Kadro Karşılaştırması")

sq_a = get_squad_row(team_a)
sq_b = get_squad_row(team_b)

if not sq_a and not sq_b:
    st.info("Kadro verisi mevcut değil. Önce `scripts/enrich_team_data.py` çalıştırın.")
else:
    def mv_stars(val: float) -> str:
        stars = int(round(val / 20))  # 0-100 → 0-5 yıldız
        stars = max(1, min(5, stars))
        return "★" * stars + "☆" * (5 - stars)

    sq_cols = st.columns(2)
    with sq_cols[0]:
        st.markdown(f"**{team_a}**")
        if sq_a:
            st.metric("Ortalama Yaş",   f"{sq_a.get('avg_age', '—'):.1f}" if sq_a.get('avg_age') else "—")
            st.metric("Top 5 Lig Oyuncusu", f"{int(sq_a.get('top5_league_count', 0))}")
            st.metric("Market Değeri Proxy", mv_stars(sq_a.get('market_value_proxy', 0)))
            st.metric("Gol/90",         f"{sq_a.get('goals_per90', 0):.2f}")
            st.metric("Asist/90",       f"{sq_a.get('assists_per90', 0):.2f}")
            st.metric("Forvet Gol/90",  f"{sq_a.get('forward_goals_p90', 0):.2f}")
        else:
            st.info("Kadro verisi yok")

    with sq_cols[1]:
        st.markdown(f"**{team_b}**")
        if sq_b:
            st.metric("Ortalama Yaş",   f"{sq_b.get('avg_age', '—'):.1f}" if sq_b.get('avg_age') else "—")
            st.metric("Top 5 Lig Oyuncusu", f"{int(sq_b.get('top5_league_count', 0))}")
            st.metric("Market Değeri Proxy", mv_stars(sq_b.get('market_value_proxy', 0)))
            st.metric("Gol/90",         f"{sq_b.get('goals_per90', 0):.2f}")
            st.metric("Asist/90",       f"{sq_b.get('assists_per90', 0):.2f}")
            st.metric("Forvet Gol/90",  f"{sq_b.get('forward_goals_p90', 0):.2f}")
        else:
            st.info("Kadro verisi yok")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 5 — Olasılıklı Sonuç (Ağırlıklı Final)
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("5️⃣ Olasılıklı Sonuç")

# Geçişli tahmini uyarla: transitive_diff → olasılık ayarı
# %70 doğrudan model + %30 geçişli etki
w_direct     = 0.70
w_transitive = 0.30

ph_direct = direct["p_home"]
pd_direct = direct["p_draw"]
pa_direct = direct["p_away"]

if transitive_count > 0:
    # Geçişli fark → ELO benzeri dönüşüm (her 1 GD fark ≈ 100 Elo)
    transitive_elo_equiv = transitive_diff * 80
    ph_trans, pd_trans, pa_trans = _elo_to_3way(
        elo_a + transitive_elo_equiv, elo_b, neutral=True
    )
    ph_final = w_direct * ph_direct + w_transitive * ph_trans
    pd_final = w_direct * pd_direct + w_transitive * pd_trans
    pa_final = w_direct * pa_direct + w_transitive * pa_trans
    total_f  = ph_final + pd_final + pa_final
    ph_final /= total_f
    pd_final /= total_f
    pa_final /= total_f
    method_note = f"70% Model + 30% Geçişli Analiz ({transitive_count} ortak rakip)"
else:
    ph_final = ph_direct
    pd_final = pd_direct
    pa_final = pa_direct
    method_note = "100% Doğrudan Model (ortak rakip yok)"

# En muhtemel sonuç
if ph_final >= pd_final and ph_final >= pa_final:
    winner = team_a
    win_p  = ph_final
elif pa_final >= ph_final and pa_final >= pd_final:
    winner = team_b
    win_p  = pa_final
else:
    winner = "Beraberlik"
    win_p  = pd_final

fc1, fc2, fc3, fc4 = st.columns(4)
fc1.metric(f"{team_a} Kazanır", f"{ph_final*100:.1f}%")
fc2.metric("Beraberlik",        f"{pd_final*100:.1f}%")
fc3.metric(f"{team_b} Kazanır", f"{pa_final*100:.1f}%")
fc4.metric("Tahmin",            winner, delta=f"{win_p*100:.1f}%")

st.caption(f"Yöntem: {method_note}")

# Sürpriz riski
upset_risk = float(direct.get("upset_risk", 0.5))
st.progress(min(1.0, upset_risk), text=f"Sürpriz Riski: {upset_risk*100:.0f}%")

# Poisson skor tahmini
lh = direct.get("lambda_home")
la = direct.get("lambda_away")
if lh and la and not np.isnan(lh) and not np.isnan(la):
    with st.expander("📊 Tahmini Skor Olasılıkları (Poisson)", expanded=False):
        score_tbl = poisson_score_table(lh, la)
        score_tbl.columns = ["Skor", "Olasılık %", "_prob"]
        st.dataframe(
            score_tbl[["Skor", "Olasılık %"]],
            use_container_width=False,
            hide_index=True,
        )
        st.caption(f"Beklenen gol: {team_a} {lh:.2f} — {team_b} {la:.2f}")
