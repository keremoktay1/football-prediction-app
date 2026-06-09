# Football Prediction App — 2026 FIFA World Cup

Probabilistic match prediction engine for the 2026 FIFA World Cup.
Built with Python, Jupyter Notebooks, Scikit-learn, Poisson modelling and Monte Carlo simulation.

## Project Structure

```
football_prediction_app/
├── data/
│   ├── raw/                  # Ham CSV dosyaları
│   └── processed/            # İşlenmiş çıktılar
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_world_cup_simulation.ipynb
├── src/
│   ├── config.py             # Dosya yolları ve sabitler
│   └── ...                   # Modüler fonksiyonlar (sonraki adım)
├── pages/                    # Streamlit sayfaları (sonraki adım)
├── app.py                    # Streamlit ana uygulama (sonraki adım)
├── requirements.txt
└── README.md
```

## Datasets

| Dosya | Açıklama |
|---|---|
| `results.csv` | Tarihi uluslararası maç sonuçları (1872–günümüz) |
| `elo_ratings_wc2026.csv` | Elo derecelendirme geçmişi |
| `GROUP_FIXTURES.CSV` | 2026 Dünya Kupası grup fikstürü |
| `KNOCKOUT_SLOTS.CSV` | Eleme turu slot şablonu |
| `goalscorers.csv` | Gol ve asist kayıtları |
| `shootouts.csv` | Penaltı atışması sonuçları |
| `players_data-2025_2026.csv` | 2025/26 sezonu oyuncu istatistikleri |

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Roadmap

- [x] Veri keşfi ve şema doğrulama
- [ ] Feature engineering (Elo diff, rolling form, attack/defense strength)
- [ ] Baseline model (Logistic Regression + Poisson)
- [ ] Monte Carlo turnuva simülasyonu
- [ ] Streamlit frontend
