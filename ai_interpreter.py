from __future__ import annotations

"""AI interpreter that converts a free-text user query into an NCBI Entrez search
using an LLM (OpenRouter) when available, with a safe fallback.

Exports:
- interpret_query_with_llm(query: str) -> dict
- interpret_and_fetch(query: str) -> (pd.DataFrame, meta: dict)
"""
import os
import json
import logging
from typing import Tuple, Dict, Any

import requests
import pandas as pd
import time
from collections import Counter
from typing import List

# lazy placeholder for tests to monkeypatch; real import is done inside interpret_and_fetch
search_and_fetch = None

logger = logging.getLogger(__name__)

# OpenRouter config via env
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "deepseek")
OPENROUTER_ENDPOINT = os.environ.get(
    "OPENROUTER_ENDPOINT", "https://api.openrouter.ai/v1/chat/completions"
)

# Circuit breaker configuration (env override)
CB_FAILURE_THRESHOLD = int(os.environ.get('AI_CB_FAILURE_THRESHOLD', '3'))
CB_COOLDOWN_SECONDS = int(os.environ.get('AI_CB_COOLDOWN_SECONDS', '300'))  # 5 minutes

# in-memory circuit breaker state
_cb_fail_count = 0
_cb_last_failure = 0.0


def _call_openrouter(prompt_messages: list, max_tokens: int = 512, temperature: float = 0.0) -> str:
    """Call OpenRouter chat completions and return assistant content.

    Returns an empty string on any failure or if the API key is missing.
    """
    # feature flag: allow disabling AI behavior
    if os.environ.get('ENABLE_AI_NCBI', 'true').lower() not in ('1', 'true', 'yes'):
        logger.info('AI calls disabled via ENABLE_AI_NCBI')
        return ""

    if not OPENROUTER_API_KEY:
        logger.debug("OpenRouter API key not set; skipping LLM call")
        return ""

    # circuit-breaker: if too many recent failures, skip LLM
    global _cb_fail_count, _cb_last_failure
    now = time.time()
    if _cb_fail_count >= CB_FAILURE_THRESHOLD and (now - _cb_last_failure) < CB_COOLDOWN_SECONDS:
        logger.warning('Circuit breaker open: skipping LLM call')
        return ""

    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": prompt_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # retry loop with backoff for transient DNS/connection issues
    backoff = 1.0
    for attempt in range(3):
        try:
            resp = requests.post(OPENROUTER_ENDPOINT, headers=headers, json=payload, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            # Chat-style response: choices -> message -> content
            choices = data.get("choices") or []
            if choices and isinstance(choices, list):
                first = choices[0]
                msg = first.get("message") or {}
                if isinstance(msg, dict):
                    return msg.get("content", "") or ""
                return first.get("text", "") or ""
            return data.get("text", "") or ""
        except requests.exceptions.RequestException as e:
            # network/HTTP error -> retry a few times then give up gracefully
            logger.warning("OpenRouter request attempt %d failed: %s", attempt + 1, e)
            _cb_fail_count += 1
            _cb_last_failure = time.time()
            time.sleep(backoff)
            backoff *= 2
        except Exception as e:
            logger.exception("OpenRouter unexpected error: %s", e)
            return ""
    logger.error("OpenRouter unavailable after retries")
    return ""


def interpret_query_with_llm(query: str) -> Dict[str, Any]:
    """Return a dict with keys 'db', 'term', 'retmax'. Uses OpenRouter when available.

    If the LLM is unavailable or parsing fails, return a safe fallback:
    {'db': 'gene', 'term': query, 'retmax': 50}
    """
    q = (query or "").strip()
    if not q:
        return {"db": "gene", "term": "", "retmax": 0}

    # Ask for a structured JSON with intent, entities, db, term, retmax, fields
    system_msg = {
        "role": "system",
        "content": (
            "You are a careful bioinformatics assistant. Given a user's free-text search request, respond ONLY"
            " with a JSON object (no surrounding text) matching the schema:\n"
            '{"intent":"recherche-gene|recherche-article|recherche-sra|recherche-sequence",'
            ' "entities": {"gene": "BRCA1", "organism": "Homo sapiens", "from": "YYYY-MM-DD", "to": "YYYY-MM-DD", "type": "mRNA"},'
            ' "db": "gene|pubmed|sra|nuccore", "term": "<entrez query>", "retmax": <int<=200>, "fields": ["Name","Description"] }\n'
            "If the user didn't specify some fields, supply reasonable defaults. Keep retmax <= 200."
        )
    }
    user_msg = {"role": "user", "content": f"User query: {q}\nReturn JSON."}

    raw = _call_openrouter([system_msg, user_msg], max_tokens=400, temperature=0.0)
    if not raw:
        logger.debug("LLM returned empty response; using heuristic fallback")
        # heuristic fallback: try to parse NL to Entrez term
        def nl_to_entrez(s: str) -> str:
            s = (s or "").strip()
            if not s:
                return s
            parts = s.split()
            gene = None
            organism = None
            others = []
            for p in parts:
                lp = p.lower().strip(',;.')
                if lp in ('human', 'homo', 'homo sapiens', 'homo_sapiens'):
                    organism = 'Homo sapiens'
                elif lp in ('mouse', 'mus', 'mus musculus', 'mus_musculus'):
                    organism = 'Mus musculus'
                elif p.isupper() and len(p) <= 8 and any(c.isalpha() for c in p):
                    gene = p
                else:
                    others.append(p)
            pieces = []
            if gene:
                pieces.append(f"{gene}[Gene]")
            if organism:
                pieces.append(f"{organism}[Organism]")
            # append remaining words as a plain text search
            if others:
                pieces.append(' '.join(others))
            return ' AND '.join(pieces) if pieces else s

        term_guess = nl_to_entrez(q)
        tokens = q.split()
        is_gene = any(t.isupper() and len(t) <= 8 for t in tokens) or len(tokens) <= 3
        db = "gene" if is_gene else "pubmed"
        default_fields = ["Name", "Description"] if db == "gene" else ["Title", "Source", "PubDate"]
        return {"intent": "recherche-gene" if db == "gene" else "recherche-article", "entities": {}, "db": db, "term": term_guess, "retmax": 50, "fields": default_fields}

    # Try to parse JSON block from raw LLM output
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("no json block")
        block = raw[start : end + 1]
        parsed = json.loads(block)
        db = parsed.get("db") or parsed.get("database") or "gene"
        term = parsed.get("term") or q
        retmax = int(parsed.get("retmax", 50))
        if retmax > 200:
            retmax = 200
        intent = parsed.get("intent", "recherche-gene" if db == "gene" else "recherche-article")
        entities = parsed.get("entities", {}) or {}
        fields = parsed.get("fields", []) or []
        # normalize
        return {"intent": intent, "entities": entities, "db": db, "term": term, "retmax": retmax, "fields": fields}
    except Exception:
        logger.exception("Failed to parse structured LLM output; using fallback")
        tokens = q.split()
        is_gene = any(t.isupper() and len(t) <= 8 for t in tokens) or len(tokens) <= 2
        db = "gene" if is_gene else "pubmed"
        default_fields = ["Name", "Description"] if db == "gene" else ["Title", "Source", "PubDate"]
        return {"intent": "recherche-gene" if db == "gene" else "recherche-article", "entities": {}, "db": db, "term": q, "retmax": 50, "fields": default_fields}


def make_summary_from_df(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "Aucun résultat trouvé."
    try:
        n = len(df)
        cols = df.columns.tolist()
        sample = df.head(3).to_dict(orient="records")
        return f"{n} résultats. Colonnes: {', '.join(cols[:6])}. Exemples: {json.dumps(sample, ensure_ascii=False)}"
    except Exception:
        return "Résultats disponibles."


def interpret_and_fetch(query: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Interpret a free-text query and fetch NCBI results.

    Returns (df, meta) where meta contains 'summary', 'db', 'term', 'retmax'.
    """
    q = (query or "").strip()
    if not q:
        return pd.DataFrame(), {"summary": "Aucune requête fournie."}

    params = interpret_query_with_llm(q)
    db = params.get("db", "gene")
    term = params.get("term") or ""
    retmax = int(params.get("retmax", 50) or 50)
    fields: List[str] = params.get("fields") or []
    intent = params.get("intent")
    entities = params.get("entities") or {}

    # If the LLM returned entities but no explicit term, build an Entrez term
    if not term or not term.strip():
        pieces = []
        if entities.get("gene"):
            pieces.append(f"{entities['gene']}[Gene]")
        if entities.get("organism"):
            pieces.append(f"{entities['organism']}[Organism]")
        # date range
        if entities.get("from") and entities.get("to"):
            # pubmed date example
            pieces.append(f"({entities['from']} : {entities['to']})")
        term = " AND ".join(pieces) if pieces else q

    # Call esearch then efetch_summary explicitly with DB fallback
    try:
        from ncbi_fetcher import esearch, efetch_summary

        def try_search_with_fallback(initial_db: str, term: str, retmax: int):
            tried = []
            # heuristic: if term contains 16S or rRNA, prefer nuccore
            term_lower = term.lower() if term else ''
            prefer_nuccore = '16s' in term_lower or '16s rrna' in term_lower or '16s rna' in term_lower or '16s' in query.lower()

            db_order = [initial_db]
            # build fallback order
            if initial_db == 'gene':
                if prefer_nuccore:
                    db_order = ['nuccore', 'pubmed', 'gene']
                else:
                    db_order = ['gene', 'pubmed', 'nuccore']
            elif initial_db == 'pubmed':
                db_order = ['pubmed', 'gene', 'nuccore']
            elif initial_db in ('nuccore', 'sra'):
                db_order = [initial_db, 'gene', 'pubmed']
            else:
                db_order = [initial_db, 'pubmed', 'gene']

            for d in db_order:
                tried.append(d)
                ids = esearch(term, db=d, retmax=retmax)
                if ids:
                    summaries = efetch_summary(ids, db=d)
                    df_local = pd.DataFrame(summaries) if summaries else pd.DataFrame()
                    return df_local, tried, d
            # nothing found
            return pd.DataFrame(), tried, None

        df, tried_dbs, final_db = try_search_with_fallback(db, term, retmax)
        if df is None or df.empty:
            return pd.DataFrame(), {"summary": "Aucun identifiant trouvé.", "db": db, "term": term, "retmax": retmax, "tried_dbs": tried_dbs}
    except Exception as e:
        logger.exception("NCBI fetch failed: %s", e)
        return pd.DataFrame(), {"summary": f"Erreur lors de la récupération: {e}", "db": db, "term": term, "retmax": retmax}

    # Post-processing / enrichment
    try:
        # add query provenance
        df['_query'] = term
        # try to normalize common date columns
        date_cols = [c for c in df.columns if 'date' in c.lower() or 'pub' in c.lower()]
        for c in date_cols:
            try:
                df[c] = pd.to_datetime(df[c], errors='coerce')
            except Exception:
                pass

        # extract organism if present
        org_col = None
        for cand in ['Organism', 'organism', 'OrganismName', 'Source']:
            if cand in df.columns:
                org_col = cand
                break
        top_organisms = []
        if org_col:
            top_organisms = [t for t, _ in Counter(df[org_col].dropna().astype(str)).most_common(3)]

        # determine suggested charts based on column types
        suggested = []
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if datetime_cols:
            suggested.append('timeseries')
        if numeric_cols:
            suggested.append('histogram')
        if categorical_cols:
            suggested.append('bar')

        # fields: if not provided, guess a few useful columns
        if not fields:
            guessed = []
            for pref in ['Name', 'Title', 'Description', 'Chr', 'Source', 'PubDate']:
                if pref in df.columns and len(guessed) < 4:
                    guessed.append(pref)
            fields = guessed

        # sample rows
        sample = df.head(3).to_dict(orient='records')

        meta = {
            'summary': f"{len(df)} résultats trouvés pour le terme '{term}' dans la base '{final_db or db}'.",
            'db': final_db or db,
            'term': term,
            'retmax': retmax,
            'fields': fields,
            'intent': intent,
            'entities': entities,
            'suggested_charts': suggested,
            'top_organisms': top_organisms,
            'sample': sample,
            'tried_dbs': tried_dbs,
        }
        return df, meta
    except Exception as e:
        logger.exception('Post-processing failed: %s', e)
        return df, {'summary': make_summary_from_df(df), 'db': db, 'term': term, 'retmax': retmax}
