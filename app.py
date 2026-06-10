"""
app.py — FIFA 2026 Tahmin Platformu ana Streamlit girişi.
"""
import datetime
import os
import subprocess
import sys

import pandas as pd
import streamlit as st

# src/ modüllerini yola ekle
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(APP_DIR, "src"))

from data_loader import (
    load_fixtures, load_match_updates, load_predictions, load_models,
    load_playoff_overrides, save_playoff_override,
)

# ── Sayfa konfigürasyonu ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("⚽ WC 2026")
st.sidebar.markdown("**Tahmin & Takip Platformu**")
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Sayfalar**
    - 📋 Fixtures Live — Skor gir, tahminlerle karşılaştır
    - 🏆 Knockout Bracket — Eleme turu bracket
    - 🔮 Custom Prediction — İki takım arası tahmin
    - 📊 Model Performance — Canlı doğruluk takibi
    """
)
st.sidebar.markdown("---")

# ── Playoff Takım İsimleri ────────────────────────────────────────────────────
_PO_SLOTS = [
    "UEFA Playoff A", "UEFA Playoff B", "UEFA Playoff C", "UEFA Playoff D",
    "FIFA Playoff 1", "FIFA Playoff 2",
]

with st.sidebar.expander("🏟️ Playoff Takımları", expanded=False):
    _current_overrides = load_playoff_overrides()
    _new_vals = {}
    for _slot in _PO_SLOTS:
        _new_vals[_slot] = st.text_input(
            _slot,
            value=_current_overrides.get(_slot, ""),
            placeholder="Takım adı girin",
            key=f"po_{_slot}",
        )
    if st.button("💾 Kaydet", key="po_save"):
        for _slot, _val in _new_vals.items():
            save_playoff_override(_slot, _val.strip())
        st.success("Playoff isimleri güncellendi!")
        st.rerun()

st.sidebar.markdown("---")

# ── Model Yeniden Eğitim ──────────────────────────────────────────────────────
_updates_for_btn = None
try:
    _updates_for_btn = load_match_updates()
except Exception:
    pass

_has_updates = (
    _updates_for_btn is not None
    and not _updates_for_btn.empty
    and len(_updates_for_btn) > 0
)

_comp_path = os.path.join(APP_DIR, "data", "processed", "model_comparison.csv")

if _has_updates:
    # Son eğitim tarihi
    _last_trained_str = ""
    if os.path.isfile(_comp_path):
        try:
            _mtime   = os.path.getmtime(_comp_path)
            _dt      = datetime.datetime.fromtimestamp(_mtime)
            _days_ago = (datetime.datetime.now() - _dt).days
            _last_trained_str = f"Son eğitim: {_days_ago} gün önce"
        except Exception:
            pass

    st.sidebar.markdown(f"**{len(_updates_for_btn)} skor girildi** — model güncellenebilir")
    if _last_trained_str:
        st.sidebar.caption(_last_trained_str)

    with st.sidebar.expander("📊 Eğitim Detayları", expanded=False):
        if os.path.isfile(_comp_path):
            try:
                _comp = pd.read_csv(_comp_path)
                for _model_name in ["Ensemble", "LR", "Random Forest"]:
                    _row = _comp[(_comp["model"] == _model_name) & (_comp["split"] == "test")]
                    if not _row.empty:
                        r = _row.iloc[0]
                        st.write(
                            f"**{_model_name}** — "
                            f"LL: {r['log_loss']:.4f} | "
                            f"Acc: {float(r['accuracy'])*100:.1f}%"
                        )
            except Exception:
                st.caption("Metrikler okunamadı.")
        else:
            st.caption("model_comparison.csv bulunamadı.")

    if st.sidebar.button("🔄 Modeli Yeniden Eğit", key="retrain_btn"):
        _script = os.path.join(APP_DIR, "scripts", "fast_model_training.py")
        with st.sidebar.spinner("Model eğitiliyor..."):
            _result = subprocess.run(
                [sys.executable, _script],
                capture_output=True, text=True, cwd=APP_DIR,
            )
        if _result.returncode == 0:
            st.sidebar.success(
                f"✅ Model yeniden eğitildi — {len(_updates_for_btn)} WC maçı dahil!"
            )
            try:
                _comp2 = pd.read_csv(_comp_path)
                _row2  = _comp2[(_comp2["model"] == "Ensemble") & (_comp2["split"] == "test")]
                if not _row2.empty:
                    _acc = float(_row2.iloc[0]["accuracy"]) * 100
                    st.sidebar.info(f"Ensemble doğruluğu: **{_acc:.1f}%**")
            except Exception:
                pass
            st.rerun()
        else:
            st.sidebar.error(f"❌ Eğitim hatası:\n```\n{_result.stderr[-600:]}\n```")
else:
    st.sidebar.caption("💡 Skor girin → model yeniden eğitilebilir")

st.sidebar.markdown("---")

# ── Bahis Oranları ────────────────────────────────────────────────────────────
with st.sidebar.expander("📡 Bahis Oranları", expanded=False):
    _api_key = os.environ.get("ODDS_API_KEY", "").strip()
    if not _api_key:
        try:
            _api_key = str(st.secrets.get("ODDS_API_KEY", "")).strip()
        except Exception:
            _api_key = ""

    if _api_key:
        if st.button("📡 Odds Güncelle", key="fetch_odds_btn"):
            _odds_script = os.path.join(APP_DIR, "scripts", "fetch_odds.py")
            with st.spinner("Bahis oranları çekiliyor..."):
                _env = {**os.environ, "ODDS_API_KEY": _api_key}
                _res = subprocess.run(
                    [sys.executable, _odds_script],
                    capture_output=True, text=True, cwd=APP_DIR, env=_env,
                )
            if _res.returncode == 0:
                st.success("✅ Bahis oranları güncellendi!")
            else:
                st.error(f"Hata:\n```\n{_res.stderr[-300:]}\n```")
        # Son güncelleme zamanı
        _odds_cache_path = os.path.join(APP_DIR, "data", "processed", "odds_cache.csv")
        if os.path.isfile(_odds_cache_path):
            try:
                _omtime = datetime.datetime.fromtimestamp(os.path.getmtime(_odds_cache_path))
                st.caption(f"Son güncelleme: {_omtime.strftime('%d.%m.%Y %H:%M')}")
            except Exception:
                pass
    else:
        st.info("`.streamlit/secrets.toml`'a `ODDS_API_KEY = '...'` ekleyin.")
        st.caption("Ücretsiz API: the-odds-api.com (500 istek/ay)")

st.sidebar.markdown("---")

# ── Veri yükle ───────────────────────────────────────────────────────────────
try:
    fixtures    = load_fixtures()
    updates     = load_match_updates()
    predictions = load_predictions()
    models      = load_models()
except Exception as exc:
    st.error(f"Veri yüklenirken hata oluştu: {exc}")
    fixtures = updates = predictions = models = None

# ── Ana başlık ───────────────────────────────────────────────────────────────
st.title("⚽ FIFA Dünya Kupası 2026 — Tahmin Platformu")
st.markdown("---")

# ── Özet metrikler ───────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

total_fixtures = len(fixtures) if fixtures is not None else 0
total_played   = len(updates) if (updates is not None and not updates.empty) else 0
total_pred     = len(predictions) if predictions is not None else 0
pct_complete   = round(total_played / total_fixtures * 100, 1) if total_fixtures > 0 else 0.0

with col1:
    st.metric(
        label="Toplam Grup Maçı",
        value=total_fixtures,
        help="GROUP_FIXTURES.CSV'deki toplam maç sayısı",
    )
with col2:
    st.metric(
        label="Oynanan Maç",
        value=total_played,
        delta=f"+{total_played}" if total_played > 0 else None,
        help="Skor girilen maç sayısı",
    )
with col3:
    st.metric(
        label="Tahmin Edilen",
        value=total_pred,
        help="predictions_latest.csv satır sayısı",
    )
with col4:
    st.metric(
        label="Tamamlanma %",
        value=f"{pct_complete}%",
    )

st.markdown("---")

# ── Durum mesajları ──────────────────────────────────────────────────────────
status_col, info_col = st.columns([1, 1])

with status_col:
    st.markdown("#### Sistem Durumu")

    if predictions is not None:
        st.success(f"✅ Model tahminleri yüklendi ({len(predictions)} maç)")
    else:
        st.warning(
            "⚠️ **predictions_latest.csv bulunamadı.**\n\n"
            "Lütfen `notebooks/03_model_training.ipynb` dosyasını çalıştırın."
        )

    if models is not None:
        st.success(f"✅ Model dosyaları yüklendi ({', '.join(models.keys())})")
    else:
        st.info(
            "ℹ️ Model dosyaları (*.pkl) bulunamadı. "
            "Custom Prediction Elo tabanlı tahmin kullanacak."
        )

    if updates is not None and not updates.empty:
        last_update = updates["updated_at"].max()
        st.info(f"📝 Son skor girişi: `{last_update}`")
    else:
        st.info("📝 Henüz skor girilmedi.")

with info_col:
    st.markdown("#### Nasıl Kullanılır?")
    st.markdown(
        """
        1. **📋 Fixtures Live** sayfasından maç sonuçlarını girin.
        2. **🏆 Knockout Bracket** ile eleme turunun şekillenmesini takip edin.
        3. **🔮 Custom Prediction** ile herhangi iki takım arası tahmin yapın.
        4. **📊 Model Performance** ile modelin doğruluğunu izleyin.

        > Skor girişleri `data/processed/match_updates.csv` dosyasına kaydedilir
        > ve tüm sayfalar bu dosyayı canlı olarak okur.
        """
    )

st.markdown("---")

# ── Son oynanan maçlar ───────────────────────────────────────────────────────
if updates is not None and not updates.empty and fixtures is not None:
    st.markdown("#### Son Girilen Sonuçlar")
    recent = updates.sort_values("updated_at", ascending=False).head(5)
    merged = recent.merge(
        fixtures[["match_id", "group", "home_team", "away_team"]],
        on="match_id",
        how="left",
    )
    for _, row in merged.iterrows():
        hs = int(row["home_score"])
        as_ = int(row["away_score"])
        res = "🟡" if hs == as_ else ("🔵" if hs > as_ else "🔴")
        st.markdown(
            f"{res} **Grup {row.get('group', '?')}** — "
            f"{row.get('home_team', '?')} **{hs}–{as_}** {row.get('away_team', '?')}"
        )
