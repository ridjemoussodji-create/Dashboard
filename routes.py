# routes.py
from dash import Input, Output, callback
from app import app
from data_loader import load_all_csv, filter_by_query
from plotting import generate_graph
from cache_manager import init_cache

cache = init_cache(app)
df = load_all_csv()

@app.callback(
    Output("result-graph", "figure"),
    Output("info-msg", "children"),
    Input("submit", "n_clicks"),
    Input("search", "value")
)
@cache.memoize(timeout=600)
def update_graph(n_clicks, query):
    if not query:
        return {}, "Veuillez entrer un sujet."
    filtered = filter_by_query(df, query)
    if filtered.empty:
        return {}, f"Aucune donnée trouvée pour '{query}'"
    figure = generate_graph(filtered, query)
    return figure, f"{len(filtered)} enregistrements retrouvés."
