"""
Module pour suivre l'historique des recherches effectuées par les utilisateurs.

Ce module fournit un tracker en mémoire simple pour enregistrer la fréquence
et les métadonnées des recherches.
"""
from collections import defaultdict, deque
import time
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

HISTORY_FILE = Path("search_history.json")

class SearchHistory:
    """Classe pour suivre l'historique des recherches en mémoire."""
    def __init__(self):
        # Clé: requête normalisée, Valeur: dict avec 'count' et 'last_meta'
        self.history = defaultdict(lambda: {'count': 0, 'timestamps': deque(maxlen=10)})
        self.history_file = HISTORY_FILE
        self._load_history()
        logger.info(f"Initialisation du SearchHistory. {len(self.history)} entrées chargées.")

    def _load_history(self):
        """Charge l'historique depuis un fichier JSON."""
        if not self.history_file.exists():
            logger.info("Aucun fichier d'historique trouvé. Démarrage avec un historique vide.")
            return
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                loaded_history = json.load(f)
                # Reconstruire le defaultdict avec des deques
                for query, data in loaded_history.items():
                    self.history[query] = {
                        'count': data.get('count', 0),
                        'timestamps': deque(data.get('timestamps', []), maxlen=10),
                        'last_meta': data.get('last_meta')
                    }
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Impossible de charger le fichier d'historique {self.history_file}: {e}")

    def _save_history(self):
        """Sauvegarde l'historique actuel dans un fichier JSON."""
        history_to_save = {query: {**data, 'timestamps': list(data['timestamps'])} for query, data in self.history.items()}
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history_to_save, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error(f"Impossible de sauvegarder le fichier d'historique {self.history_file}: {e}")
            
    def log_search(self, query: str, meta: dict):
        """Enregistre une recherche réussie."""
        normalized_query = (query or "").strip().lower()
        if not normalized_query:
            return

        self.history[normalized_query]['count'] += 1
        self.history[normalized_query]['timestamps'].append(time.time())
        self.history[normalized_query]['last_meta'] = meta
        self._save_history()
        logger.debug(f"Recherche enregistrée et sauvegardée pour '{normalized_query}'. Compte: {self.history[normalized_query]['count']}.")

    def get_stats(self, query: str) -> dict:
        """Récupère les statistiques pour une requête spécifique."""
        normalized_query = (query or "").strip().lower()
        return self.history.get(normalized_query)

# Instance globale du tracker d'historique
search_history_tracker = SearchHistory()