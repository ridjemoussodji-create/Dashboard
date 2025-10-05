"""
Configuration centralisée pour le dashboard NASA Spatial Biology
"""
import os
from pathlib import Path
DATA_DIR = Path("data")
CHUNK_SIZE = 5000  # ou ce que tu veux, mais il te le faut


# Chemins de base
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
INDEX_DIR = BASE_DIR / "search_index"

# Créer les dossiers s'ils n'existent pas
DATA_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

# Configuration de la base de données
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///nasa_spatial_bio.db')

# Configuration du cache
CACHE_TYPE = 'SimpleCache'
CACHE_DEFAULT_TIMEOUT = 300  # 5 minutes

# Configuration de l'application
APP_NAME = "Dashboard Biologie Spatiale"
APP_VERSION = "1.0"
DEBUG = True

DATA_PATH = "data/"
CSV_FILES = ["data.csv"]  # Remplace 'data.csv' si besoin par les noms de tes fichiers
CACHE_TIMEOUT = 600


# Configuration des données
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.json'}

# Configuration de la recherche
SEARCH_RESULTS_LIMIT = 100
FUZZY_SEARCH_THRESHOLD = 80  # Score minimum pour RapidFuzz

# Configuration des visualisations
DEFAULT_PLOT_HEIGHT = 600
DEFAULT_PLOT_WIDTH = 1200
PLOT_TEMPLATE = "plotly_white"  # Theme Plotly

# NASA API Configuration (optionnel)
NASA_API_KEY = os.environ.get('NASA_API_KEY', 'DEMO_KEY')
NASA_API_BASE_URL = "https://api.nasa.gov"

# NCBI API key (E-utilities). Prefer providing this via environment variable NCBI_API_KEY.
# If not set, code will work without a key but may be subject to stricter rate limits.
NCBI_API_KEY = os.environ.get('NCBI_API_KEY')

# Paramètres de performance
CHUNK_SIZE = 10000  # Pour le traitement de gros fichiers
N_WORKERS = 4  # Pour le traitement parallèle

# Couleurs et style
COLOR_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

# Colonnes attendues dans les datasets (exemple)
EXPECTED_COLUMNS = [
    'experiment_id',
    'title',
    'description',
    'organism',
    'tissue',
    'condition',
    'date',
    'measurement_type',
    'value'
]