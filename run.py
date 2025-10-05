"""
Script de démarrage simplifié pour le dashboard NASA
"""
import os
import sys
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Vérifie que toutes les dépendances sont installées"""
    required_packages = [
        'dash', 'plotly', 'pandas', 'flask', 
        'dash_bootstrap_components', 'rapidfuzz'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        logger.error(f"Packages manquants: {', '.join(missing)}")
        logger.info("Installez-les avec: pip install -r requirements.txt")
        return False
    
    return True


def check_directories():
    """Vérifie et crée les dossiers nécessaires"""
    required_dirs = ['data', 'static', 'search_index']
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            logger.info(f"Dossier créé: {dir_name}")
    
    return True


def load_env():
    """Charge les variables d'environnement"""
    try:
        from dotenv import load_dotenv
        if Path('.env').exists():
            load_dotenv()
            logger.info("Variables d'environnement chargées depuis .env")
        else:
            logger.info("Fichier .env non trouvé, utilisation des valeurs par défaut")
    except ImportError:
        logger.warning("python-dotenv non installé, tentative de chargement simple de .env")
        # Simple fallback: parse .env ourselves (KEY=VALUE lines) to avoid adding dependency
        try:
            env_path = Path('.env')
            if env_path.exists():
                with env_path.open('r', encoding='utf-8') as fh:
                    for raw in fh:
                        line = raw.strip()
                        if not line or line.startswith('#'):
                            continue
                        if '=' not in line:
                            continue
                        key, val = line.split('=', 1)
                        key = key.strip()
                        val = val.strip().strip('"').strip("'")
                        # set only if not present to allow env override
                        if key and key not in os.environ:
                            os.environ[key] = val
                logger.info("Variables d'environnement chargées depuis .env (fallback)")
            else:
                logger.info("Fichier .env non trouvé, utilisation des valeurs par défaut")
        except Exception as e:
            logger.exception(f"Erreur lors du chargement simple de .env: {e}")


def check_data():
    """Vérifie la présence de données"""
    data_dir = Path('data')
    csv_files = list(data_dir.glob('*.csv'))
    
    if not csv_files:
        logger.warning("Aucun fichier CSV trouvé dans le dossier data/")
        logger.info("Un dataset d'exemple sera créé automatiquement")
    else:
        logger.info(f"{len(csv_files)} fichier(s) CSV trouvé(s):")
        for csv in csv_files:
            logger.info(f"  - {csv.name}")
    
    return True


def main():
    """Fonction principale de démarrage"""
    logger.info("=" * 60)
    logger.info("NASA Spatial Biology Dashboard - Démarrage")
    logger.info("=" * 60)
    
    # Vérifications
    logger.info("Vérification des dépendances...")
    if not check_dependencies():
        sys.exit(1)
    
    logger.info("Vérification des dossiers...")
    check_directories()
    
    logger.info("Chargement de la configuration...")
    load_env()
    
    logger.info("Vérification des données...")
    check_data()
    
    # Import et lancement de l'application
    logger.info("\n" + "=" * 60)
    logger.info("Démarrage de l'application...")
    logger.info("=" * 60)
    
    try:
        from app import app

        # Récupérer la config
        host = os.environ.get('HOST', '0.0.0.0')
        port = int(os.environ.get('PORT', 8050))
        debug = os.environ.get('DEBUG', 'True').lower() == 'true'

        logger.info(f"\n🚀 Application disponible sur: http://localhost:{port}")
        logger.info(f"📊 Mode Debug: {'Activé' if debug else 'Désactivé'}")
        logger.info(f"🌐 Host: {host}")
        logger.info(f"\nAppuyez sur Ctrl+C pour arrêter\n")

        # Dash v2+ uses app.run
        app.run(debug=debug, host=host, port=port)

    except KeyboardInterrupt:
        logger.info("\n\nArrêt de l'application...")
    except Exception as e:
        logger.error(f"Erreur lors du démarrage: {e}")
        logger.exception("Détails de l'erreur:")
        sys.exit(1)


if __name__ == '__main__':
    main()