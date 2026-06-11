"""
config.py — Proje genelinde kullanılan sabitler ve yollar.
"""
import os

# Kök dizin: bu dosyanın bulunduğu src/ klasörünün bir üstü
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR       = os.path.join(ROOT_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(ROOT_DIR, 'data', 'processed')

FILES = {
    'results'       : os.path.join(RAW_DIR, 'results.csv'),
    'elo'           : os.path.join(RAW_DIR, 'elo_ratings_wc2026.csv'),
    'group_fixtures': os.path.join(RAW_DIR, 'GROUP_FIXTURES.CSV'),
    'knockout_slots': os.path.join(RAW_DIR, 'KNOCKOUT_SLOTS.CSV'),
    'goalscorers'   : os.path.join(RAW_DIR, 'goalscorers.csv'),
    'shootouts'     : os.path.join(RAW_DIR, 'shootouts.csv'),
    'former_names'  : os.path.join(RAW_DIR, 'former_names.csv'),
    'players'       : os.path.join(RAW_DIR, 'players_data-2025_2026.csv'),
    # Çıktılar
    'predictions'   : os.path.join(PROCESSED_DIR, 'predictions_latest.csv'),
    'group_tables'  : os.path.join(PROCESSED_DIR, 'group_tables_latest.csv'),
    'simulation'    : os.path.join(PROCESSED_DIR, 'simulation_latest.csv'),
}

# Takım ismi standardizasyon haritası
TEAM_NAME_MAP = {
    "Czech Republic"          : "Czechia",
    "Ivory Coast"             : "Côte d'Ivoire",
    "Cape Verde"              : "Cabo Verde",
    "USA"                     : "United States",
    "Korea Republic"          : "South Korea",
    "Republic of Ireland"     : "Ireland",
    "North Ireland"           : "Northern Ireland",
    "Kyrgyz Republic"         : "Kyrgyzstan",
    "Trinidad and Tobago"     : "Trinidad & Tobago",
    "St Kitts and Nevis"      : "Saint Kitts and Nevis",
    "St Lucia"                : "Saint Lucia",
    "St Vincent / Grenadines" : "Saint Vincent and the Grenadines",
}

# Model parametreleri
ROLLING_FORM_WINDOW = 5      # Son N maç
N_SIMULATIONS       = 5_000  # Monte Carlo simülasyon sayısı (run_simulation varsayılanı)
ELO_K_FACTOR        = 40     # Elo K faktörü — WC maçlarında kullanılan değer
BASE_GOAL_RATE      = 1.35   # Ortalama gol/maç tahmini (uluslararası futbol)
