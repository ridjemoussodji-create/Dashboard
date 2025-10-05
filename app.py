"""
Application principale Dash pour le dashboard NASA Spatial Biology
Contient :
- Layout (sidebar + main area)
- Recherche globale, filtres avancés, visualisations
- Endpoint d'administration pour vider le cache
Ce fichier a été nettoyé pour enlever les doublons et consolider les callbacks.
"""

import json
import logging
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd

# Imports locaux
from data_loader import data_loader
from ncbi_fetcher import search_and_fetch
import ai_interpreter
from search_engine import search_engine
from search_history import search_history_tracker
from plotting import plot_generator
from cache_manager import init_cache
import cache_manager
import utils
from utils import create_advanced_stats # Import the new function
from config import APP_NAME, APP_VERSION, DEBUG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialisation de l'application Dash
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    title=APP_NAME,
)

# Initialiser le cache et récupérer l'instance
cache = init_cache(app)

# Inject custom assets (static folder) and fonts to match the provided template
app.index_string = f"""<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{APP_NAME}</title>
        {{%favicon%}}
        {{%css%}}
        <!-- Google Fonts (adjust as needed) -->
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap" rel="stylesheet">
        <!-- Custom styles and scripts -->
        <link rel="stylesheet" href="/static/custom.css">
        <script src="/static/custom.js"></script>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>"""


# Layout: sidebar + main area
app.layout = dbc.Container(
    [
        dcc.Location(id="url", refresh=False),
        dbc.Row(
            [
                # Sidebar
                dbc.Col(
                    html.Div(
                        [
                            html.Div([
                                # Zone pour le logo. Assurez-vous d'avoir un fichier 'logo.png' dans votre dossier 'assets'.
                                html.Img(src=app.get_asset_url('logo.png'), className="sidebar-logo"),
                                html.Div([
                                    html.H2("DASHBOARD", className="sidebar-title"), 
                                    html.P(f"v{APP_VERSION}", className="sidebar-version")
                                ], className="sidebar-title-container")
                            ], className="sidebar-header"),
                            html.Div(
                                [
                                    dbc.NavLink([html.I(className="fas fa-th-large me-2"), "Tableau de bord"], href="/dashboard", id={"type": "sidebar-link", "index": "dashboard"}, className="nav-link sidebar-item", active=False),
                                    dbc.NavLink([html.I(className="fas fa-cube me-2"), "Artefact 3D"], href="/artefact-3d", id={"type": "sidebar-link", "index": "artefact-3d"}, className="nav-link sidebar-item", active=False),
                                    dbc.NavLink([html.I(className="fas fa-question-circle me-2"), "Quiz"], href="/quiz", id={"type": "sidebar-link", "index": "quiz"}, className="nav-link sidebar-item", active=False),
                                    dbc.NavLink([html.I(className="fas fa-newspaper me-2"), "Actualité"], href="/actualites", id={"type": "sidebar-link", "index": "actualites"}, className="nav-link sidebar-item", active=False),
                                    dbc.NavLink([html.I(className="fas fa-graduation-cap me-2"), "Apprentissage"], href="/apprentissage", id={"type": "sidebar-link", "index": "apprentissage"}, className="nav-link sidebar-item", active=False),
                                ],
                                className="sidebar-nav",
                            ),
                            html.Div(className="sidebar-footer", children=[html.Small("Spatial Biology Dashboard"), html.Div([html.Button("Vider le cache", id="clear-cache-btn", className="btn btn-sm btn-link mt-2"), html.Span(id="clear-cache-result", style={"display": "block", "color": "var(--muted)", "fontSize": "12px", "marginTop": "6px"})])]),
                        ],
                        className="sidebar",
                    ),
                    width=2,
                    className="sidebar-col",
                ),

                # Main column
                dbc.Col(
                    [
                        # Global search + toggle filters
                        dbc.Row(dbc.Col(html.Div([html.Div([dcc.Input(id="search-input", type="text", placeholder="Quelles données cherchez-Vous. Tapez @ pour mentions et / pour raccourcis.", debounce=True, className="global-search-input"), dbc.Button(html.I(className="fas fa-search"), id="search-button", color="link", className="global-search-btn")], className="d-flex justify-content-center align-items-center w-100"), html.Button("Filtres avancés", id="toggle-filters", className="mt-3 muted btn btn-link")], className="global-search-container"), width=12), className="mb-3"),
                        dbc.Row(dbc.Col(html.Div(id='ai-summary', className='mt-2 mb-3'))),

                        # Advanced filters card (hidden by style initially)
                        dbc.Row(
                            dbc.Col(
                                html.Div(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H5([html.I(className="fas fa-filter me-2"), "Recherche et Filtres"], className="card-title"),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.Label("Dataset:", className="fw-bold"),
                                                                dcc.Dropdown(
                                                                    id="dataset-dropdown", # Les options seront mises à jour dynamiquement
                                                                    options=[{"label": "NCBI (requête libre)", "value": "ncbi:"}],
                                                                    value="ncbi:",
                                                                    placeholder="Sélectionner un dataset",
                                                                    clearable=False,
                                                                ),
                                                            ],
                                                            md=6,
                                                        ),
                                                        dbc.Col([html.Label("Recherche actuelle:", className="fw-bold"), dbc.Input(id="search-preview", type="text", placeholder="Votre recherche", disabled=True)], md=6),
                                                    ],
                                                    className="mb-3",
                                                ),
                                                html.Div(id="dynamic-filters"),
                                                dbc.Row(dbc.Col([
                                                    dbc.Button([html.I(className="fas fa-sync-alt me-2"), "Rafraîchir"], id="refresh-button", color="primary", className="me-2"),
                                                    dbc.Button([html.I(className="fas fa-download me-2"), "Exporter CSV"], id="export-button", color="success"),
                                                    dbc.Button("Charger les données locales", id="load-local-data-btn", color="secondary", className="ms-2", style={'display': 'none'})                                                ])),
                                            ]
                                        )
                                    ),
                                    id="filters-collapse",
                                    style={"display": "none"},
                                ),
                                width=12,
                            ),
                            className="mb-3",
                        ),

                        # Quick stats
                        dbc.Row([
                            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Total Entrées", className="text-muted"), html.H3(id="stat-total", children="0", className="text-primary")]), className="shadow-sm text-center"), md=3),
                            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Colonnes", className="text-muted"), html.H3(id="stat-columns", children="0", className="text-info")]), className="shadow-sm text-center"), md=3),
                            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Mémoire", className="text-muted"), html.H3(id="stat-memory", children="0 MB", className="text-warning")]), className="shadow-sm text-center"), md=3),
                            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Résultats Filtrés", className="text-muted"), html.H3(id="stat-filtered", children="0", className="text-success")]), className="shadow-sm text-center"), md=3),
                        ], className="mb-4"),

                        # Tabs: table, stats, viz, about
                        dbc.Row(dbc.Col(dbc.Tabs([
                            dbc.Tab(dcc.Loading(id="loading-table", children=[html.Div(id="data-table", className="mt-3")]), label="📋 Tableau de Données", tab_id="table"),
                            dbc.Tab(html.Div(id="statistics-panel", className="mt-3"), label="📈 Statistiques", tab_id="stats"),
                            dbc.Tab([dbc.Row([dbc.Col([html.Label("Type de graphique:", className="fw-bold mt-3"), dcc.Dropdown(id="plot-type", options=[{"label": "Nuage de points", "value": "scatter"}, {"label": "Graphique en barres", "value": "bar"}, {"label": "Ligne (série temporelle)", "value": "line"}, {"label": "Histogramme", "value": "histogram"}, {"label": "Box Plot", "value": "box"}, {"label": "Camembert", "value": "pie"}, {"label": "Heatmap", "value": "heatmap"}, {"label": "3D Scatter", "value": "3d_scatter"}, {"label": "Matrice de corrélation", "value": "correlation"}], value="scatter")], md=4), dbc.Col([html.Label("Axe X:", className="fw-bold mt-3"), dcc.Dropdown(id="x-axis", placeholder="Sélectionner X")], md=4), dbc.Col([html.Label("Axe Y:", className="fw-bold mt-3"), dcc.Dropdown(id="y-axis", placeholder="Sélectionner Y")], md=4)]), dbc.Row([dbc.Col([html.Label("Couleur (optionnel):", className="fw-bold mt-3"), dcc.Dropdown(id="color-axis", placeholder="Sélectionner couleur")], md=6), dbc.Col([html.Label("Taille (optionnel):", className="fw-bold mt-3"), dcc.Dropdown(id="size-axis", placeholder="Sélectionner taille")], md=6)]), dcc.Loading(id="loading-viz", type="default", children=[dcc.Graph(id="main-plot", style={"height": "600px"})])], label="📊 Visualisation", tab_id="viz"),
                            dbc.Tab(dbc.Card(dbc.CardBody([html.H5("À propos", className="card-title"), html.P(["Dashboard de visualisation et modélisation des données NASA en biologie spatiale.", html.Br(), "Technologies: Dash, Plotly, Pandas, Flask, PostgreSQL", html.Br(), html.Br(), html.Strong("Fonctionnalités:")]), html.Ul([html.Li("Recherche intelligente avec fuzzy matching"), html.Li("Visualisations interactives multiples"), html.Li("Filtres dynamiques"), html.Li("Export de données"), html.Li("Analyse statistique avancée"), html.Li("Support de gros volumes de données")])])), label="ℹ️ À propos", tab_id="about"),
                        ], id="tabs", active_tab="table")), ),

                        dcc.Download(id="download-dataframe-csv"),
                        dcc.Store(id="filtered-data-store"),
                        dcc.Store(id="ai-meta-store"),
                        html.Button(id="search-submit-button", style={"display": "none"}),
                    ],
                    width=10,
                    className="main-col",
                ),
            ],
        ),
    ],
    fluid=True,
    className="p-4",
)


@app.callback(
    [Output("dynamic-filters", "children"), Output("x-axis", "options"), Output("y-axis", "options"), Output("color-axis", "options"), Output("size-axis", "options"), Output("dataset-dropdown", "options")],
    Input("dataset-dropdown", "value"),
)
def update_filters_and_axes(dataset_name):
    # For NCBI we do not have pre-known columns until after a search is run.
    if not dataset_name:
        return html.Div(), [], [], [], [], [{"label": "NCBI (requête libre)", "value": "ncbi:"}]
    if dataset_name == "ncbi:":
        # no pre-built filters/axes for remote source
        local_datasets_options = [{"label": name, "value": name} for name in data_loader.get_dataset_names() if name != 'ncbi:']
        return html.Div("Sélectionnez 'NCBI (requête libre)' et lancez une recherche."), [], [], [], [], [{"label": "NCBI (requête libre)", "value": "ncbi:"}] + local_datasets_options
        
    # Fallback: if other datasets are present, get their columns
    df = data_loader.get_dataset(dataset_name)
    if df is None or df.empty:
        local_datasets_options = [{"label": name, "value": name} for name in data_loader.get_dataset_names() if name != 'ncbi:']
        return html.Div("Aucune donnée disponible"), [], [], [], [], [{"label": "NCBI (requête libre)", "value": "ncbi:"}] + local_datasets_options

    all_columns = [{"label": col, "value": col} for col in df.columns]
    numeric_columns = [{"label": col, "value": col} for col in df.select_dtypes(include=["number"]).columns]
    categorical_columns = [{"label": col, "value": col} for col in df.select_dtypes(include=["object"]).columns]

    filters = []
    for col in categorical_columns[:5]:
        unique_values = data_loader.get_unique_values(dataset_name, col["value"])
        if len(unique_values) <= 50:
            filters.append(dbc.Col([html.Label(f"{col['label']}:", className="fw-bold"), dcc.Dropdown(id={"type": "filter-dropdown", "index": col["value"]}, options=[{"label": v, "value": v} for v in unique_values], multi=True, placeholder=f"Filtrer par {col['label']}")], md=6, className="mb-2"))

    filter_row = dbc.Row(filters) if filters else html.Div()
    local_datasets_options = [{"label": name, "value": name} for name in data_loader.get_dataset_names() if name != 'ncbi:']
    return filter_row, all_columns, numeric_columns, all_columns, numeric_columns, [{"label": "NCBI (requête libre)", "value": "ncbi:"}] + local_datasets_options


@app.callback(Output("filters-collapse", "style"), Input("toggle-filters", "n_clicks"), State("filters-collapse", "style"))
def toggle_filters(n_clicks, current_style):
    logger.info(f"toggle_filters called: n_clicks={n_clicks} current_style={current_style}")
    if not n_clicks:
        return current_style or {"display": "none"}
    if current_style and current_style.get("display") == "block":
        return {"display": "none"}
    return {"display": "block"}


@app.callback(Output("search-preview", "value"), Input("search-input", "value"))
def update_search_preview(value):
    # Mirror the main search input into the preview field inside advanced filters
    return value or ""


@app.callback(
    Output("tabs", "active_tab"),
    [Input("url", "pathname"), Input({"type": "sidebar-link", "index": dash.dependencies.ALL}, "n_clicks")]
)
def navigate(pathname, sidebar_clicks):
    """Unified navigation: react to either URL changes or sidebar clicks.

    Uses callback_context to determine which input triggered the callback.
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    trigger = ctx.triggered[0]["prop_id"]

    # If a sidebar-link was clicked, its prop_id is a JSON-like dict before the '.'
    try:
        prefix = trigger.split('.', 1)[0]
        comp = json.loads(prefix.replace("'", '"'))
        if isinstance(comp, dict) and comp.get("type") == "sidebar-link":
            idx = comp.get("index")
            mapping = {"dashboard": "viz", "artefact-3d": "table", "quiz": "stats", "actualites": "about", "apprentissage": "about"}
            return mapping.get(idx, "viz")
    except Exception:
        # Not a sidebar click; fall through to pathname handling
        pass

    # Fallback: map pathname to tab
    if pathname:
        mapping = {"/dashboard": "viz", "/artefact-3d": "table", "/quiz": "stats", "/actualites": "about", "/apprentissage": "about"}
        return mapping.get(pathname, "viz")

    return dash.no_update


@app.callback(Output({"type": "sidebar-link", "index": dash.dependencies.ALL}, "active"), Input("tabs", "active_tab"))
def tab_to_sidebar(active_tab):
    reverse = {"viz": "dashboard", "table": "artefact-3d", "stats": "quiz", "about": "actualites"}
    desired = reverse.get(active_tab, "dashboard")
    ids = ["dashboard", "artefact-3d", "quiz", "actualites", "apprentissage"]
    return [i == desired for i in ids]


@app.callback(
    [
        Output("ai-meta-store", "data"),
        Output("filtered-data-store", "data"),
        Output("stat-total", "children"),
        Output("stat-columns", "children"),
        Output("stat-memory", "children"),
        Output("stat-filtered", "children"),
        Output("load-local-data-btn", "style"),
    ],
    [
        Input("dataset-dropdown", "value"),
        Input("search-input", "value"),
        Input({"type": "filter-dropdown", "index": dash.dependencies.ALL}, "value"),
        Input("refresh-button", "n_clicks"),
        Input("search-button", "n_clicks"),
        Input("search-submit-button", "n_clicks"),
        Input("load-local-data-btn", "n_clicks"),
    ],
    [State({"type": "filter-dropdown", "index": dash.dependencies.ALL}, "id")],
)
def filter_data(dataset_name, search_query, filter_values, refresh_nclicks, searchbtn_nclicks, submit_nclicks, load_local_nclicks, filter_ids):
    """Perform search and filtering.

    Returns: meta (dict) , data_records, total_rows, total_columns, memory_str, filtered_rows
    """
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Default empty response
    empty_meta = {"summary": "Aucune donnée."}
    hide_button = {'display': 'none'}

    if triggered_id == 'load-local-data-btn' and load_local_nclicks:
        data_loader.load_all_datasets()

    if not dataset_name:
        return empty_meta, None, "0", "0", "0 MB", "0", hide_button

    # NCBI source (AI-assisted)
    if dataset_name == "ncbi:":
        if not search_query or not search_query.strip():
            return {"summary": "Aucune requête  pour le moment."}, None, "0", "0", "0 MB", "0", hide_button

        try:
            df, meta = ai_interpreter.interpret_and_fetch(search_query)
        except Exception as e:
            logger.exception(f"Erreur lors de la recherche NCBI: {e}")
            return {"summary": f"Erreur lors de la recherche: {e}"}, None, "0", "0", "0 MB", "0", {'display': 'inline-block'}

        if df is None or df.empty:
            summary = meta.get("summary") if isinstance(meta, dict) else str(meta)
            return {"summary": summary}, [], "0", "0", "0 MB", "0", {'display': 'inline-block'}
        
        # Enregistrer la recherche réussie dans l'historique
        search_history_tracker.log_search(search_query, meta)

        total_rows = len(df)
        total_cols = len(df.columns)
        memory = utils.get_memory_usage(df)["formatted"]
        data_dict = df.head(1000).to_dict("records")
        filtered_rows = total_rows

        return (meta if isinstance(meta, dict) else {"summary": str(meta)}, data_dict, utils.format_number(total_rows, 0), str(total_cols), memory, utils.format_number(filtered_rows, 0), hide_button)

    # Fallback for local datasets (defensive)
    df = data_loader.get_dataset(dataset_name)
    if df is None or df.empty:
        return empty_meta, None, "0", "0", "0 MB", "0", hide_button

    total_rows = len(df)
    total_cols = len(df.columns)
    memory = utils.get_memory_usage(df)["formatted"]

    filters = {}
    if filter_ids and filter_values:
        for fid, fval in zip(filter_ids, filter_values):
            if fval:
                filters[fid["index"]] = fval

    filtered_df = search_engine.advanced_search(df, query=search_query or "", filters=filters)
    filtered_rows = len(filtered_df)
    data_dict = filtered_df.head(1000).to_dict("records")

    meta = {"summary": f"Données locales: {filter_dataset}"} if False else {"summary": "Données locales chargées."}
    return meta, data_dict, utils.format_number(total_rows, 0), str(total_cols), memory, utils.format_number(filtered_rows, 0), hide_button


@app.callback(
    Output("ai-summary", "children"),
    Input("ai-meta-store", "data"),
    State("search-input", "value")
)
def generate_contextual_summary(meta, search_query):
    """Génère un résumé contextuel en utilisant l'historique et l'IA."""
    if not meta:
        return ""
    
    initial_summary = meta.get("summary", "Aucun résumé disponible.")
    history_stats = search_history_tracker.get_stats(search_query)

    # Si pas d'historique ou pas de clé API, on affiche le résumé de base
    if not history_stats or not getattr(ai_interpreter, 'OPENROUTER_API_KEY', None):
        return dbc.Card(dbc.CardBody([html.H5("Résumé IA"), html.P(initial_summary)]), className="mb-3")

    # Construire un prompt pour l'IA avec le contexte de l'historique
    try:
        count = history_stats.get('count', 1)
        last_meta = history_stats.get('last_meta', {})
        top_organisms = last_meta.get('top_organisms', [])
        sample_data = last_meta.get('sample', [])
        
        system_prompt = (
            "Tu es un assistant de recherche scientifique passionné et éloquent, spécialisé en biologie spatiale. "
            "Ta mission est de présenter les données de manière captivante, comme si tu faisais découvrir un nouveau monde à l'utilisateur. "
            "Réponds en français, dans un style narratif et engageant."
        )
        user_prompt = (
            f"Je viens de rechercher le terme '{search_query}'. "
            f"Ce sujet a été exploré plus de {count}e fois par de grand nom du domaine de la biologie. "
            f"Les principaux organismes trouvés sont : {', '.join(top_organisms) if top_organisms else 'non spécifiés'}. "
            f"Voici un aperçu des données : {json.dumps(sample_data, ensure_ascii=False, indent=2)}. "
            "Rédige un court paragraphe (3-4 phrases) pour introduire ces résultats. Mentionne un ou deux scientifiques qui auraient pu travailler sur ce sujet (réels ou plausibles) et ajoute un 'fun fact' scientifique pertinent et surprenant lié à la recherche. Sois créatif et inspirant."
        )
        
        prompt_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        ai_summary = ai_interpreter._call_openrouter(prompt_messages, max_tokens=150, temperature=0.2)
        
        final_summary = ai_summary if ai_summary else initial_summary
        return dbc.Card(dbc.CardBody([html.H5("Résumé IA"), html.P(final_summary)]), className="mb-3")
    except Exception as e:
        logger.error(f"Erreur lors de la génération du résumé contextuel par l'IA : {e}")
        return dbc.Card(dbc.CardBody([html.H5("Résumé IA"), html.P(initial_summary)]), className="mb-3")


@app.callback(Output("main-plot", "figure"), [Input("filtered-data-store", "data"), Input("plot-type", "value"), Input("x-axis", "value"), Input("y-axis", "value"), Input("color-axis", "value"), Input("size-axis", "value")])
@cache.memoize(timeout=300)
def update_plot(data, plot_type, x, y, color, size):
    if not data:
        return plot_generator._empty_plot("Aucune donnée à afficher")
    df = pd.DataFrame(data)
    if df.empty:
        return plot_generator._empty_plot("Aucune donnée disponible")
    try:
        if plot_type == "scatter":
            if not x or not y:
                return plot_generator._empty_plot("Sélectionnez X et Y")
            return plot_generator.create_scatter_plot(df, x, y, color, size)
        if plot_type == "bar":
            if not x or not y:
                return plot_generator._empty_plot("Sélectionnez X et Y")
            return plot_generator.create_bar_chart(df, x, y, color)
        if plot_type == "line":
            if not x or not y:
                return plot_generator._empty_plot("Sélectionnez X et Y")
            return plot_generator.create_line_chart(df, x, y, color)
        if plot_type == "histogram":
            if not x:
                return plot_generator._empty_plot("Sélectionnez une colonne")
            return plot_generator.create_histogram(df, x)
        if plot_type == "box":
            if not y:
                return plot_generator._empty_plot("Sélectionnez Y")
            return plot_generator.create_box_plot(df, y, x, color)
        if plot_type == "pie":
            if not x or not y:
                return plot_generator._empty_plot("Sélectionnez X (noms) et Y (valeurs)")
            return plot_generator.create_pie_chart(df, x, y)
        if plot_type == "heatmap":
            if not x or not y:
                return plot_generator._empty_plot("Sélectionnez X, Y et Z (couleur)")
            z = color if color else y
            return plot_generator.create_heatmap(df, x, y, z)
        if plot_type == "3d_scatter":
            if not x or not y or not color:
                return plot_generator._empty_plot("Sélectionnez X, Y et Z (couleur)")
            return plot_generator.create_3d_scatter(df, x, y, color)
        if plot_type == "correlation":
            return plot_generator.create_correlation_matrix(df)
        return plot_generator._empty_plot("Type de graphique non supporté")
    except Exception as e:
        logger.error(f"Erreur lors de la création du graphique: {e}")
        return plot_generator._empty_plot(f"Erreur: {str(e)}")


@app.callback(Output("data-table", "children"), Input("filtered-data-store", "data"))
def update_table(data):
    if not data:
        return html.P("Aucune donnée à afficher", className="text-muted")
    df = pd.DataFrame(data)
    if df.empty:
        return html.P("Aucune donnée disponible", className="text-muted")

    # Améliorer les noms des colonnes pour une meilleure lisibilité
    df_display = df.head(100).copy()
    df_display.columns = [col.replace('_', ' ').replace('-', ' ').title() for col in df_display.columns]

    # Ajouter une classe CSS personnalisée pour un style plus agréable
    table = dbc.Table.from_dataframe(df_display, striped=True, bordered=False, hover=True, responsive=True, class_name="custom-table")
    info = html.P(f"Affichage de {min(100, len(df))} lignes sur {len(df)} total", className="text-muted mt-2")
    return html.Div([table, info])


@app.callback(Output("statistics-panel", "children"), Input("filtered-data-store", "data"))
def update_statistics(data):
    if not data:
        return html.P("Aucune donnée à afficher", className="text-muted")
    df = pd.DataFrame(data)
    if df.empty:
        return html.P("Aucune donnée disponible", className="text-muted")

    # Basic summary stats
    summary_stats = utils.create_summary_stats(df)
    content = [
        dbc.Card([
            dbc.CardHeader(html.H5("Informations Générales")),
            dbc.CardBody([
                html.P(f"Nombre de lignes: {summary_stats['shape'][0]}"),
                html.P(f"Nombre de colonnes: {summary_stats['shape'][1]}"),
                html.P(f"Utilisation mémoire: {summary_stats['memory_usage']['formatted']}")
            ])
        ], className="mb-3")
    ]

    missing_df = pd.DataFrame({
        "Colonne": list(summary_stats["missing_values"].keys()),
        "Valeurs Manquantes": list(summary_stats["missing_values"].values()),
        "Pourcentage": [f"{v:.1f}%" for v in summary_stats["missing_percentage"].values()]
    })
    if missing_df["Valeurs Manquantes"].sum() > 0:
        content.append(dbc.Card([
            dbc.CardHeader(html.H5("Valeurs Manquantes")),
            dbc.CardBody([dbc.Table.from_dataframe(missing_df[missing_df['Valeurs Manquantes'] > 0], striped=True, bordered=True, hover=True)])
        ], className="mb-3"))

    # Advanced statistics
    advanced_stats = create_advanced_stats(df)
    content.append(
        dbc.Card([
            dbc.CardHeader(html.H5("Métriques Quantitatives")),
            dbc.CardBody([
                html.P(f"Nombre total d'articles publiés: {advanced_stats['total_articles']}"),
                html.P(f"Nombre d'auteurs distincts (par Catégorie): {advanced_stats['distinct_authors']}"),
                html.P(f"Nombre de méthodologies différentes (par Thème principal): {advanced_stats['distinct_methodologies']}"),
            ])
        ], className="mb-3")
    )

    # Chart options
    chart_options = [
        {"label": "Diagramme en bâtonnets", "value": "bar"},
        {"label": "Diagramme circulaire", "value": "pie"},
    ]

    # Principaux sujets ou tendances
    subjects_chart_card = dbc.Card([
        dbc.CardHeader(html.H5("Principaux Sujets")),
        dbc.CardBody([
            dcc.Dropdown(
                id="subjects-chart-type",
                options=chart_options,
                value="bar",
                clearable=False,
                className="mb-3"
            ),
            dcc.Graph(id="subjects-chart", config={'displayModeBar': False})
        ])
    ], className="mb-3")

    # Principales catégories
    categories_chart_card = dbc.Card([
        dbc.CardHeader(html.H5("Principales Catégories")),
        dbc.CardBody([
            dcc.Dropdown(
                id="categories-chart-type",
                options=chart_options,
                value="bar",
                clearable=False,
                className="mb-3"
            ),
            dcc.Graph(id="categories-chart", config={'displayModeBar': False})
        ])
    ], className="mb-3")

    # Time series chart placeholder
    time_series_card = dbc.Card([
        dbc.CardHeader(html.H5("Variations des publications au cours du temps")),
        dbc.CardBody([
            html.P("Cette visualisation nécessite une colonne de date dans les données pour suivre les variations au cours du temps."),
            html.P(f"Niveaux d'impact ou pertinence scientifique: {advanced_stats['impact_levels']}"),
            html.P(f"Fréquence des citations de concepts innovants ou émergents: {advanced_stats['innovative_concepts_frequency']}"),
            html.P(f"Occurrence des technologies, frameworks ou outils mentionnés: {advanced_stats['technologies_mentions']}"),
        ])
    ], className="mb-3")

    # Arrange charts in a grid
    content.append(
        dbc.Row([
            dbc.Col(subjects_chart_card, md=6),
            dbc.Col(categories_chart_card, md=6),
        ], className="mb-3")
    )
    content.append(
        dbc.Row([
            dbc.Col(time_series_card, md=6), # Changed to md=6 to align with the 2x2 grid idea, if there were 4 charts, it would be 2x2. For 3, it will be 2 on top, 1 on bottom left.
        ], className="mb-3 justify-content-center") # Centering the last card if it's alone
    )

    return html.Div(content)


@app.callback(Output("download-dataframe-csv", "data"), Input("export-button", "n_clicks"), State("filtered-data-store", "data"), prevent_initial_call=True)
def export_data(n_clicks, data):
    if not data:
        return None
    df = pd.DataFrame(data)
    return dcc.send_data_frame(df.to_csv, "nasa_data_export.csv", index=False)


# Note: navigation is handled by the unified `navigate` callback above.


# Admin endpoint to clear cache (single definition)
@app.server.route("/clear-cache", methods=["POST", "GET"])
def clear_cache_route():
    try:
        if getattr(cache_manager, "cache", None):
            cache_manager.cache.clear()
            logger.info("Cache vidé via /clear-cache endpoint")
            return "Cache vidé", 200
        else:
            logger.warning("Demande clear-cache, mais cache non initialisé")
            return "Cache non initialisé", 500
    except Exception as e:
        logger.exception(f"Erreur lors du vidage du cache: {e}")
        return f"Erreur: {e}", 500


@app.callback(Output("clear-cache-result", "children"), Input("clear-cache-btn", "n_clicks"), prevent_initial_call=True)
def clear_cache_button(n_clicks):
    try:
        if getattr(cache_manager, "cache", None):
            cache_manager.cache.clear()
            logger.info("Cache vidé via UI button")
            return "Cache vidé"
        else:
            logger.warning("Tentative de vider le cache via UI mais cache non initialisé")
            return "Cache non initialisé"
    except Exception as e:
        logger.exception(f"Erreur lors du vidage du cache via UI: {e}")
        return f"Erreur: {e}"


@app.callback(
    Output("subjects-chart", "figure"),
    [Input("filtered-data-store", "data"), Input("subjects-chart-type", "value")]
)
def update_subjects_chart(data, chart_type):
    if not data:
        return plot_generator._empty_plot("Aucune donnée à afficher")
    df = pd.DataFrame(data)
    if df.empty or "Thème principal" not in df.columns:
        return plot_generator._empty_plot("Aucune donnée disponible pour les sujets")

    advanced_stats = create_advanced_stats(df)
    if not advanced_stats['top_subjects']:
        return plot_generator._empty_plot("Aucun sujet principal à afficher")

    subjects_df = pd.DataFrame(advanced_stats['top_subjects'])

    if chart_type == "bar":
        fig = plot_generator.create_bar_chart(subjects_df, x='name', y='count', color='name')
    elif chart_type == "pie":
        fig = plot_generator.create_pie_chart(subjects_df, names='name', values='count')
    else:
        fig = plot_generator._empty_plot("Type de graphique non supporté")

    fig.update_layout(title_text="Principaux sujets ou tendances (par Thème principal)", showlegend=False)
    return fig


@app.callback(
    Output("categories-chart", "figure"),
    [Input("filtered-data-store", "data"), Input("categories-chart-type", "value")]
)
def update_categories_chart(data, chart_type):
    if not data:
        return plot_generator._empty_plot("Aucune donnée à afficher")
    df = pd.DataFrame(data)
    if df.empty or "Catégorie" not in df.columns:
        return plot_generator._empty_plot("Aucune donnée disponible pour les catégories")

    advanced_stats = create_advanced_stats(df)
    if not advanced_stats['top_categories']:
        return plot_generator._empty_plot("Aucune catégorie principale à afficher")

    categories_df = pd.DataFrame(advanced_stats['top_categories'])

    if chart_type == "bar":
        fig = plot_generator.create_bar_chart(categories_df, x='name', y='count', color='name')
    elif chart_type == "pie":
        fig = plot_generator.create_pie_chart(categories_df, names='name', values='count')
    else:
        fig = plot_generator._empty_plot("Type de graphique non supporté")

    fig.update_layout(title_text="Principales Catégories", showlegend=False)
    return fig

# Exposer le serveur Flask pour Gunicorn
server = app.server


if __name__ == "__main__":
    logger.info(f"Démarrage de {APP_NAME} v{APP_VERSION}")
    app.run_server(debug=DEBUG, host="0.0.0.0", port=8050)
