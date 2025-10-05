# cache_manager.py
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


# Try to import flask_caching; if not available provide a minimal in-memory fallback
try:
    from flask_caching import Cache  # type: ignore
except Exception:
    Cache = None  # type: ignore


class _SimpleInMemoryCache:
    def __init__(self):
        self._store: Dict[str, Any] = {}

    def get(self, key: str):
        return self._store.get(key)

    def set(self, key: str, value, timeout: int = 600):
        self._store[key] = value

    def clear(self):
        self._store.clear()


# Module-level cache object to be imported by other modules
cache = None  # type: ignore


def init_cache(server_app=None):
    """Initialise le cache. Si flask_caching est présent, l'utilise, sinon retourne un cache simple.

    Retourne l'instance de cache.
    """
    global cache
    if Cache is None:
        logger.warning('flask_caching not available: using in-memory fallback cache')
        cache = _SimpleInMemoryCache()
        return cache

    try:
        # Si server_app est un wrapper (comme Dash), il peut exposer .server
        flask_app = getattr(server_app, 'server', server_app)
        cache = Cache(flask_app, config={"CACHE_TYPE": "simple"})
        return cache
    except TypeError:
        try:
            cache = Cache(config={"CACHE_TYPE": "simple"})
            cache.init_app(flask_app)
            return cache
        except Exception as e:
            logger.exception(f"Impossible d'initialiser le cache: {e}")
            # fallback to simple cache
            cache = _SimpleInMemoryCache()
            return cache
