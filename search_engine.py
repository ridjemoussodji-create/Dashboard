# search_engine.py
from rapidfuzz import fuzz, process
import logging
from typing import Dict, List, Any
import pandas as pd

logger = logging.getLogger(__name__)


def fuzzy_search(df, query: str, score_cutoff: int = 60):
    """Recherche floue sur l'ensemble du DataFrame en concaténant les colonnes.

    Retourne les lignes correspondant au fuzzy match du `query`.
    """
    if df is None or df.empty:
        return df

    # Construire une Series de 'choix' où chaque ligne est la concaténation des colonnes textuelles
    text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    if text_cols:
        choices = df[text_cols].astype(str).agg(' '.join, axis=1)
    else:
        # fallback si pas de colonnes textuelles: concaténer tout
        choices = df.astype(str).agg(' '.join, axis=1)

    try:
        # token_set_ratio is more robust for partial token overlaps
        results = process.extract(query, choices, scorer=fuzz.token_set_ratio, score_cutoff=score_cutoff)
        # Si process.extract ne renvoie rien (ex: incompatibilité de type), faire un fallback manuel
        if not results:
            manual = []
            q = str(query).lower()
            for idx, choice in choices.items():
                try:
                    score = fuzz.token_set_ratio(q, str(choice).lower())
                except Exception:
                    score = 0
                if score >= score_cutoff:
                    manual.append((choice, score, idx))
            results = manual
    except Exception as e:
        logger.exception(f"Erreur lors du fuzzy search: {e}")
        return df.iloc[0:0]

    # process.extract renvoie des tuples (match, score, key) où key correspond à l'index
    indices = [key for (_match, _score, key) in results]

    # Si les indices sont des labels (habituellement 0..n-1), on utilise loc/iloc suivant le type
    try:
        return df.loc[indices]
    except Exception:
        return df.iloc[indices]


class SearchEngine:
    """Petit wrapper pour fournir l'API attendue par le reste de l'application.

    Méthodes exposées:
    - advanced_search(df, query, filters): applique des filtres puis recherche floue
    - get_suggestions(df, query, column, limit): retourne des suggestions pour autocomplétion
    """

    def advanced_search(self, df, query: str = "", filters: Dict[str, Any] = None, score_cutoff: int = 60):
        # Appliquer les filtres si fournis en respectant les types de colonnes
        filtered = df
        if filters:
            for col, val in filters.items():
                if col not in filtered.columns:
                    continue

                if val is None:
                    continue

                series = filtered[col]

                # Si la colonne est numérique et la valeur peut être convertie, faire un filtrage numérique
                if pd.api.types.is_numeric_dtype(series):
                    if isinstance(val, (list, tuple, set)):
                        # convertir chaque élément en float si possible
                        try:
                            numeric_vals = [float(v) for v in val]
                            filtered = filtered[series.isin(numeric_vals)]
                        except Exception:
                            # fallback: comparaison en str
                            filtered = filtered[series.astype(str).isin([str(v) for v in val])]
                    else:
                        try:
                            num = float(val)
                            filtered = filtered[series == num]
                        except Exception:
                            filtered = filtered[series.astype(str) == str(val)]

                # Si la colonne est datetime-like, essayer de parser et comparer
                elif pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_object_dtype(series):
                    # tenter de convertir en datetimes pour la comparaison stricte si possible
                    try:
                        ser_dt = pd.to_datetime(series, errors='coerce')
                        if isinstance(val, (list, tuple, set)):
                            vals_dt = []
                            for v in val:
                                try:
                                    vals_dt.append(pd.to_datetime(v))
                                except Exception:
                                    pass
                            if vals_dt:
                                filtered = filtered[ser_dt.isin(vals_dt)]
                            else:
                                filtered = filtered[series.astype(str).isin([str(v) for v in val])]
                        else:
                            try:
                                vdt = pd.to_datetime(val)
                                filtered = filtered[ser_dt == vdt]
                            except Exception:
                                filtered = filtered[series.astype(str) == str(val)]
                    except Exception:
                        # fallback générique
                        if isinstance(val, (list, tuple, set)):
                            filtered = filtered[series.astype(str).isin([str(v) for v in val])]
                        else:
                            filtered = filtered[series.astype(str) == str(val)]

                else:
                    # Supporter les listes (multi-valeurs) ou une valeur simple pour toutes les autres colonnes
                    if isinstance(val, (list, tuple, set)):
                        filtered = filtered[series.astype(str).isin([str(v) for v in val])]
                    else:
                        # match exact (entier) ou string equivalence
                        filtered = filtered[series.astype(str) == str(val)]

        # Si pas de requête, retourner simplement le DataFrame filtré
        if not query:
            return filtered

        # Sinon appliquer la recherche floue
        return fuzzy_search(filtered, query, score_cutoff=score_cutoff)

    def get_suggestions(self, df, query: str, column: str, limit: int = 10) -> List[str]:
        """Retourne des suggestions basées sur les valeurs uniques d'une colonne.

        Priorité: valeurs commençant par la query (cas-insensible), puis fuzzy matches.
        """
        if df is None or df.empty or column not in df.columns:
            return []

        # Obtenir valeurs uniques non-nulls sous forme de strings
        uniques = df[column].dropna().astype(str).unique().tolist()
        if not query:
            return uniques[:limit]

        q = str(query).lower()

        # 1) matches commençant par la query
        starts = [u for u in uniques if u.lower().startswith(q)]
        if len(starts) >= limit:
            return starts[:limit]

        # 2) fuzzy matching pour compléter
        try:
            results = process.extract(q, uniques, scorer=fuzz.partial_ratio, limit=limit)
        except Exception:
            return starts[:limit]

        suggestions = starts[:]
        for match, score, _key in results:
            if match not in suggestions:
                suggestions.append(match)
            if len(suggestions) >= limit:
                break

        return suggestions[:limit]


# Exposer une instance nommée 'search_engine' pour garder la compatibilité avec les imports
search_engine = SearchEngine()
