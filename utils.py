("""Utilities used by the dashboard: small helpers for formatting and basic stats.

These are minimal implementations to satisfy app callbacks.
""")
from __future__ import annotations

import pandas as pd
from typing import Dict, Any, List
from collections import Counter


def format_number(value, digits=0) -> str:
	"""Format a number with thousands separator and optional decimals."""
	try:
		if digits <= 0:
			return f"{int(value):,}"
		else:
			fmt = f"{{:,.{digits}f}}"
			return fmt.format(float(value))
	except Exception:
		return str(value)


def get_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
	"""Return memory usage summary for a DataFrame."""
	if df is None or df.empty:
		return {"bytes": 0, "formatted": "0 MB"}
	try:
		bytes_used = int(df.memory_usage(deep=True).sum())
		mb = bytes_used / 1024 ** 2
		return {"bytes": bytes_used, "formatted": f"{mb:.2f} MB"}
	except Exception:
		return {"bytes": 0, "formatted": "0 MB"}


def create_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
	"""Create a small summary dict used by the statistics panel."""
	# Always return a dict with the expected keys so the UI callbacks don't KeyError
	defaults = {
		"shape": (0, 0),
		"memory_usage": {"bytes": 0, "formatted": "0 MB"},
		"missing_values": {},
		"missing_percentage": {},
		"numeric_stats": {},
	}
	if df is None or df.empty:
		return defaults

	try:
		shape = df.shape
		mem = get_memory_usage(df)
		missing = df.isnull().sum().to_dict()
		missing_pct = {k: (v / shape[0] * 100 if shape[0] > 0 else 0) for k, v in missing.items()}
		numeric = df.select_dtypes(include=['number']).describe().to_dict()
		return {
			"shape": shape,
			"memory_usage": mem,
			"missing_values": missing,
			"missing_percentage": missing_pct,
			"numeric_stats": numeric,
		}
	except Exception:
		return defaults


def create_advanced_stats(df: pd.DataFrame) -> Dict[str, Any]:
	"""
	Calcule des métriques quantitatives et qualitatives avancées sur le DataFrame.
	"""
	stats = {}

	if df is None or df.empty:
		return {
			"total_articles": 0,
			"distinct_authors": 0,
			"distinct_methodologies": 0,
			"top_subjects": [],
			"top_categories": [],
			"impact_levels": "Non disponible dans les données actuelles",
			"innovative_concepts_frequency": "Non disponible dans les données actuelles",
			"technologies_mentions": "Non disponible dans les données actuelles",
		}

	# 1. Métriques quantitatives
	stats["total_articles"] = len(df)

	# Assuming 'Catégorie' can be a proxy for distinct authors/contributors to topics
	stats["distinct_authors"] = df["Catégorie"].nunique() if "Catégorie" in df.columns else 0

	# Assuming 'Thème principal' can represent different methodologies
	stats["distinct_methodologies"] = df["Thème principal"].nunique() if "Thème principal" in df.columns else 0

	# 2. Statistiques qualitatives et analytiques
	# Principaux sujets ou tendances identifiés
	if "Thème principal" in df.columns:
		top_subjects = df["Thème principal"].value_counts().head(5).to_dict()
		stats["top_subjects"] = [{"name": k, "count": v} for k, v in top_subjects.items()]
	else:
		stats["top_subjects"] = []

	if "Catégorie" in df.columns:
		top_categories = df["Catégorie"].value_counts().head(5).to_dict()
		stats["top_categories"] = [{"name": k, "count": v} for k, v in top_categories.items()]
	else:
		stats["top_categories"] = []

	# Niveaux d'impact ou pertinence scientifique
	stats["impact_levels"] = "Non disponible dans les données actuelles (nécessite des données externes comme le facteur d'impact des revues)."

	# Fréquence des citations de concepts innovants ou émergents
	stats["innovative_concepts_frequency"] = "Non disponible dans les données actuelles (nécessite une analyse de texte plus approfondie ou des données de citation)."

	# Occurrence des technologies, frameworks ou outils mentionnés
	stats["technologies_mentions"] = "Non disponible dans les données actuelles (nécessite une analyse de texte des résumés)."

	return stats
