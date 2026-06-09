# Sonraki Oturum — Yapılacaklar

## A · Yeni Feature'lar (Feature Engineering + Tüm Modeller)

Bunların tamamı `fast_feature_engineering.py`'a eklenmeli,
ardından `FEATURE_COLS` güncellenmeli ve modeller yeniden eğitilmeli.

| Feature | Kaynak | Açıklama |
|---|---|---|
| `points_last_5` | results.csv | Son 5 maç toplam puan (ev+dep ayrı) |
| `goal_diff_last_5` | results.csv | Son 5 maç gol farkı |
| `goals_for_last_5` | results.csv | Son 5 maç atılan gol |
| `goals_against_last_5` | results.csv | Son 5 maç yenilen gol |
| `win_streak` | results.csv | Güncel galibiyet serisi |
| `loss_streak` | results.csv | Güncel mağlubiyet serisi |
| `clean_sheet_rate` | results.csv | Son N maçta kalesini koruyan oran |
| `failed_to_score_rate` | results.csv | Son N maçta gol atamayan oran |
| `head_to_head_goal_diff` | results.csv | İki takım arasındaki son 5 doğrudan karşılaşmada gol farkı |
| `common_opponent_score` | results.csv | Ortak rakiplere karşı ortalama performans farkı |

Not: `weighted_form`, `attack_strength`, `defense_weakness`, `gf_last_n`, `ga_last_n`
zaten var — sadece `FEATURE_COLS`'a bazıları eklenmemişti.

---

## B · Frontend İyileştirmeleri

### B1 · UEFA/FIFA Playoff Takım İsimleri (Öncelikli)
- `data/processed/playoff_overrides.csv` yeni dosya: `slot_name,actual_team`
- `data_loader.py`: `load_playoff_overrides()`, `save_playoff_override()` fonksiyonları
- `app.py` sidebar'ına "Playoff Takımları" bölümü ekle
  - UEFA Playoff A/B/C/D + FIFA Playoff 1/2 için metin girişleri
  - Kaydet butonu → CSV'yi günceller
- `load_fixtures()` çıktısını overrides ile filtrele (bütün sayfalar otomatik güncellenir)
- Elo hesaplamasında override edilmiş takım ismini kullan

### B2 · Skor Girişi (Canlı Maçlar)
- `1_Fixtures_Live.py`'daki expander'ı kaldır
- Her maç satırında inline skor girişi yap (expander olmadan)
- Oynanmış maçlarda skor düzenleyebilir olsun (form açık)
- Oynanmamış maçlarda küçük "+ Skor Gir" butonu ile toggle
- Knockout maçlar için de aynı pattern

### B3 · Format Düzeltmeleri
- Tarih formatlarını normalize et (Türkçe ay isimleri)
- Playoff takım isimlerini italik yerine `[TBD]` badge ile göster
- Olasılık sütunlarında %0 / %100 overflow'u engelle
- Mobil uyumlu column ratio'ları

---

## C · Veri Filtresi
- `fast_feature_engineering.py`'da `team_history` dict'ini de 2000+ ile sınırla
  (şu an tarihi feature matrix 2000+'dan başlıyor ama rolling form hesabı
   tüm geçmişi kullanıyor)
- Elo data: sadece 2000 sonrası snapshot'ları kullan

---

## D · Ortak Rakip Tahmini (Common Opponent Strength)
- İki takım hiç karşılaşmamışsa:
  - Her iki takımın ortak rakiplerine karşı sonuçlarını bul
  - `common_opponent_score_diff` = home_team_avg - away_team_avg (o rakilere karşı)
- `head_to_head_goal_diff` = son 5 H2H maçtaki ev sahibi lehine gol farkı (yoksa 0)
- Bunları `fast_feature_engineering.py`'a ekle

---

## E · Git Durumu
Tüm değişiklikler commit edildi. Son hash: ecb395e

## F · App Durumu
- URL: http://localhost:8502
- 6 model eğitildi (XGBoost en iyi: LL=0.9453, Acc=%55.8)
- 72 grup maçı tahmin edildi (Ensemble: LR %50 + Poisson %50)
- Monte Carlo simülasyon: Spain %28, Argentina %20, France %10
