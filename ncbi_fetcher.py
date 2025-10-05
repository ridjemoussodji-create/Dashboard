"""
Simple NCBI E-utilities helper.

Capabilities:
- esearch -> get list of ids for a query
- efetch  -> retrieve records (summary) for ids
- convert results to a pandas.DataFrame
- basic caching via cache_manager.cache when available

This module purposefully keeps dependencies minimal (requests + stdlib xml parsing).
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional, Dict
import requests
import xml.etree.ElementTree as ET
import pandas as pd

from config import SEARCH_RESULTS_LIMIT, NCBI_API_KEY
import cache_manager

logger = logging.getLogger(__name__)


NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _cache_get(key: str):
    try:
        if getattr(cache_manager, "cache", None):
            return cache_manager.cache.get(key)
    except Exception:
        logger.debug("Cache get failed")
    return None


def _cache_set(key: str, value, timeout: int = 600):
    try:
        if getattr(cache_manager, "cache", None):
            cache_manager.cache.set(key, value, timeout=timeout)
    except Exception:
        logger.debug("Cache set failed")


def esearch(term: str, db: str = "nuccore", retmax: int = 100) -> List[str]:
    """Search NCBI and return list of ids for a term.

    Args:
        term: query term
        db: NCBI database ('nuccore','sra','pubmed',...)
        retmax: maximum number of ids to retrieve
    Returns:
        list of id strings
    """
    key = f"ncbi:esearch:{db}:{term}:{retmax}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    params = {"db": db, "term": term, "retmax": retmax, "retmode": "xml"}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    url = f"{NCBI_BASE}/esearch.fcgi"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    ids = [idn.text for idn in root.findall(".//IdList/Id")]
    _cache_set(key, ids)
    return ids


def efetch_summary(id_list: List[str], db: str = "nuccore") -> List[Dict]:
    """Fetch summaries for a list of ids (using esummary where appropriate).

    Returns a list of dicts with simple fields.
    """
    if not id_list:
        return []

    key = f"ncbi:efetch:{db}:{','.join(id_list)}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    # Use esummary to get compact summaries
    params = {"db": db, "id": ",".join(id_list), "retmode": "xml"}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    url = f"{NCBI_BASE}/esummary.fcgi"
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    results = []
    # Two common esummary formats: DocSum/Item (e.g., nuccore/pubmed) and DocumentSummary (e.g., gene)
    docsum_nodes = root.findall('.//DocSum')
    if docsum_nodes:
        for docsum in docsum_nodes:
            item_map = {}
            uid = docsum.find('Id')
            if uid is not None:
                item_map['uid'] = uid.text
            for item in docsum.findall('Item'):
                name = item.get('Name')
                if name:
                    item_map[name] = item.text
            results.append(item_map)
    else:
        # Try DocumentSummary nodes (used by gene esummary)
        docsum_nodes = root.findall('.//DocumentSummary')
        if not docsum_nodes:
            # no recognized summaries
            _cache_set(key, [])
            return []
        for d in docsum_nodes:
            item_map = {}
            uid = d.get('uid') or d.find('Id')
            if uid is not None:
                item_map['uid'] = uid if isinstance(uid, str) else uid.text
            # children of DocumentSummary are direct tags with text
            for child in list(d):
                # skip nested complex nodes by stringifying if needed
                tag = child.tag
                # get text content; for nested lists, join inner texts
                if child.text and child.text.strip():
                    item_map[tag] = child.text.strip()
                else:
                    # try to extract nested ints or inner text
                    inner_texts = []
                    for sub in list(child):
                        if sub.text and sub.text.strip():
                            inner_texts.append(sub.text.strip())
                    if inner_texts:
                        item_map[tag] = ','.join(inner_texts)
            results.append(item_map)

    _cache_set(key, results)
    return results


def search_and_fetch(term: str, db: str = "nuccore", retmax: int = 100) -> pd.DataFrame:
    """High-level helper: search then fetch summaries and return a DataFrame.

    Caches intermediate results.
    """
    term = (term or "").strip()
    if not term:
        return pd.DataFrame()

    if retmax is None:
        retmax = SEARCH_RESULTS_LIMIT or 100

    ids = esearch(term, db=db, retmax=retmax)
    if not ids:
        return pd.DataFrame()

    summaries = efetch_summary(ids, db=db)
    if not summaries:
        return pd.DataFrame()

    df = pd.DataFrame(summaries)
    # add a search term column for traceability
    df['_query'] = term
    return df


if __name__ == "__main__":
    # petit test local
    logging.basicConfig(level=logging.INFO)
    df = search_and_fetch("BRCA1[Gene] AND human[Organism]", db="gene", retmax=10)
    print(df.head())
