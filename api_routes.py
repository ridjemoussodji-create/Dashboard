"""
API REST pour le dashboard NASA - Routes Flask
Permet d'accéder aux données via des endpoints REST
"""
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import logging
from pathlib import Path

from data_loader import data_loader
from search_engine import search_engine
from plotting import plot_generator
import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Créer l'application Flask API
api_app = Flask(__name__)
CORS(api_app)  # Activer CORS pour les requêtes cross-origin


# ============================================================================
# ENDPOINTS - Datasets
# ============================================================================

@api_app.route('/api/v1/datasets', methods=['GET'])
def get_datasets():
    """
    Liste tous les datasets disponibles
    
    Returns:
        JSON: Liste des datasets avec leurs métadonnées
    """
    try:
        datasets = data_loader.get_dataset_names()
        
        datasets_info = []
        for name in datasets:
            df = data_loader.get_dataset(name)
            if df is not None:
                datasets_info.append({
                    'name': name,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': df.columns.tolist(),
                    'memory_mb': round(utils.get_memory_usage(df)['mb'], 2)
                })
        
        return jsonify({
            'success': True,
            'count': len(datasets_info),
            'datasets': datasets_info
        }), 200
        
    except Exception as e:
        logger.error(f"Erreur dans get_datasets: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@api_app.route('/api/v1/datasets/<dataset_name>', methods=['GET'])
def get_dataset_data(dataset_name):
    """
    Récupère les données d'un dataset spécifique
    
    Args:
        dataset_name: Nom du dataset
    
    Query params:
        limit: Nombre max de lignes (défaut: 100)
        offset: Décalage (défaut: 0)
        columns: Colonnes à retourner (séparées par virgule)
    
    Returns:
        JSON: Données du dataset
    """
    try:
        df = data_loader.get_dataset(dataset_name)
        
        if df is None:
            return jsonify({'success': False, 'error': 'Dataset not found'}), 404
        
        # Paramètres de pagination
        limit = min(int(request.args.get('limit', 100)), 1000)
        offset = int(request.args.get('offset', 0))
        
        # Colonnes spécifiques
        columns = request.args.get('columns')
        if columns:
            columns = columns.split(',')
            df = df[columns]
        
        # Appliquer pagination
        df_page = df.iloc[offset:offset+limit]
        
        return jsonify({
            'success': True,
            'dataset': dataset_name,
            'total_rows': len(df),
            'returned_rows': len(df_page),
            'offset': offset,
            'limit': limit,
            'data': df_page.to_dict('records')
        }), 200
        
    except Exception as e:
        logger.error(f"Erreur dans get_dataset_data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@api_app.route('/api/v1/datasets/<dataset_name>/statistics', methods=['GET'])
def get_dataset_statistics(dataset_name):
    """
    Obtient les statistiques d'un dataset
    
    Returns:
        JSON: Statistiques descriptives
    """
    try:
        df = data_loader.get_dataset(dataset_name)
        
        if df is None:
            return jsonify({'success': False, 'error': 'Dataset not found'}), 404
        
        stats = data_loader.get_statistics(dataset_name)
        
        return jsonify({
            'success': True,
            'dataset': dataset_name,
            'statistics': stats
        }), 200
        
    except Exception as e:
        logger.error(f"Erreur dans get_dataset_statistics: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# ENDPOINTS - Recherche
# ============================================================================

@api_app.route('/api/v1/search', methods=['POST'])
def search_data():
    """
    Recherche dans un dataset
    
    Body JSON:
        {
            "dataset": "nom_dataset",
            "query": "terme de recherche",
            "filters": {"colonne": "valeur"},
            "limit": 100
        }
    
    Returns:
        JSON: Résultats de recherche
    """
    try:
        data = request.get_json()
        
        dataset_name = data.get('dataset')
        query = data.get('query', '')
        filters = data.get('filters', {})
        limit = min(int(data.get('limit', 100)), 1000)
        
        if not dataset_name:
            return jsonify({'success': False, 'error': 'dataset required'}), 400
        
        df = data_loader.get_dataset(dataset_name)
        
        if df is None:
            return jsonify({'success': False, 'error': 'Dataset not found'}), 404
        
        # Recherche avancée
        results = search_engine.advanced_search(df, query=query, filters=filters)
        
        # Limiter les résultats
        results = results.head(limit)
        
        return jsonify({
            'success': True,
            'query': query,
            'filters': filters,
            'total_results': len(results),
            'results': results.to_dict('records')
        }), 200
        
    except Exception as e:
        logger.error(f"Erreur dans search_data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@api_app.route('/api/v1/suggest', methods=['GET'])
def get_suggestions():
    """
    Obtient des suggestions d'autocomplétion
    
    Query params:
        dataset: Nom du dataset
        column: Colonne à suggérer
        query: Début du terme
        limit: Nombre de suggestions (défaut: 10)
    
    Returns:
        JSON: Liste de suggestions
    """
    try:
        dataset_name = request.args.get('dataset')
        column = request.args.get('column')
        query = request.args.get('query', '')
        limit = int(request.args.get('limit', 10))
        
        if not dataset_name or not column:
            return jsonify({'success': False, 'error': 'dataset and column required'}), 400
        
        df = data_loader.get_dataset(dataset_name)
        
        if df is None:
            return jsonify({'success': False, 'error': 'Dataset not found'}), 404
        
        suggestions = search_engine.get_suggestions(df, query, column, limit)
        
        return jsonify({
            'success': True,
            'column': column,
            'query': query,
            'suggestions': suggestions
        }), 200
        
    except Exception as e:
        logger.error(f"Erreur dans get_suggestions: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# ENDPOINTS - Filtrage
# ============================================================================

@api_app.route('/api/v1/filter', methods=['POST'])
def filter_dataset():
    """
    Filtre un dataset selon des critères
    
    Body JSON:
        {
            "dataset": "nom_dataset",
            "filters": {"colonne": "valeur" ou ["val1", "val2"]},
            "limit": 100
        }
    
    Returns:
        JSON: Données filtrées
    """
    try:
        data = request.get_json()
        
        dataset_name = data.get('dataset')
        filters = data.get('filters', {})
        limit = min(int(data.get('limit', 100)), 1000)
        
        if not dataset_name:
            return jsonify({'success': False, 'error': 'dataset required'}), 400
        
        filtered_df = data_loader.filter_data(dataset_name, filters)
        
        if filtered_df.empty:
            return jsonify({
                'success': True,
                'message': 'No results found',
                'total_results': 0,
                'results': []
            }), 200
        
        # Limiter les résultats
        filtered_df = filtered_df.head(limit)
        
        return jsonify({
            'success': True,
            'filters': filters,
            'total_results': len(filtered_df),
            'results': filtered_df.to_dict('records')
        }), 200
        
    except Exception as e:
        logger.error(f"Erreur dans filter_dataset: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# ENDPOINTS - Colonnes
# ============================================================================

@api_app.route('/api/v1/datasets/<dataset_name>/columns', methods=['GET'])
def get_columns(dataset_name):
    """
    Liste les colonnes d'un dataset avec leurs types
    
    Returns:
        JSON: Liste des colonnes
    """
    try:
        df = data_loader.get_dataset(dataset_name)
        
        if df is None:
            return jsonify({'success': False, 'error': 'Dataset not found'}), 404
        
        column_types = utils.detect_column_types(df)
        
        columns_info = []
        for col in df.columns:
            columns_info.append({
                'name': col,
                'type': column_types.get(col, 'unknown'),
                'unique_values': int(df[col].nunique()),
                'null_count': int(df[col].isnull().sum())
            })
        
        return jsonify({
            'success': True,
            'dataset': dataset_name,
            'columns': columns_info
        }), 200
        
    except Exception as e:
        logger.error(f"Erreur dans get_columns: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@api_app.route('/api/v1/datasets/<dataset_name>/columns/<column_name>/values', methods=['GET'])
def get_unique_values(dataset_name, column_name):
    """
    Obtient les valeurs uniques d'une colonne
    
    Returns:
        JSON: Valeurs uniques
    """
    try:
        unique_values = data_loader.get_unique_values(dataset_name, column_name)
        
        return jsonify({
            'success': True,
            'dataset': dataset_name,
            'column': column_name,
            'count': len(unique_values),
            'values': unique_values
        }), 200
        
    except Exception as e:
        logger.error(f"Erreur dans get_unique_values: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# ENDPOINTS - Export
# ============================================================================

@api_app.route('/api/v1/export', methods=['POST'])
def export_data():
    """
    Exporte des données filtrées en CSV
    
    Body JSON:
        {
            "dataset": "nom_dataset",
            "filters": {},
            "format": "csv"
        }
    
    Returns:
        Fichier CSV à télécharger
    """
    try:
        data = request.get_json()
        
        dataset_name = data.get('dataset')
        filters = data.get('filters', {})
        file_format = data.get('format', 'csv')
        
        if not dataset_name:
            return jsonify({'success': False, 'error': 'dataset required'}), 400
        
        df = data_loader.get_dataset(dataset_name)
        
        if df is None:
            return jsonify({'success': False, 'error': 'Dataset not found'}), 404
        
        # Appliquer les filtres
        if filters:
            df = data_loader.filter_data(dataset_name, filters)
        
        # Export
        output_path = Path('temp') / f'export_{dataset_name}.{file_format}'
        output_path.parent.mkdir(exist_ok=True)
        
        if file_format == 'csv':
            df.to_csv(output_path, index=False)
        elif file_format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        elif file_format == 'xlsx':
            df.to_excel(output_path, index=False)
        else:
            return jsonify({'success': False, 'error': 'Unsupported format'}), 400
        
        return send_file(output_path, as_attachment=True)
        
    except Exception as e:
        logger.error(f"Erreur dans export_data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# ENDPOINTS - Agrégation
# ============================================================================

@api_app.route('/api/v1/aggregate', methods=['POST'])
def aggregate_data():
    """
    Agrège des données
    
    Body JSON:
        {
            "dataset": "nom_dataset",
            "group_by": ["colonne1", "colonne2"],
            "aggregations": {
                "colonne_value": "mean"
            }
        }
    
    Returns:
        JSON: Données agrégées
    """
    try:
        data = request.get_json()
        
        dataset_name = data.get('dataset')
        group_by = data.get('group_by', [])
        agg_dict = data.get('aggregations', {})
        
        if not dataset_name or not group_by or not agg_dict:
            return jsonify({'success': False, 'error': 'Missing parameters'}), 400
        
        aggregated = data_loader.aggregate_data(dataset_name, group_by, agg_dict)
        
        return jsonify({
            'success': True,
            'group_by': group_by,
            'aggregations': agg_dict,
            'results': aggregated.to_dict('records')
        }), 200
        
    except Exception as e:
        logger.error(f"Erreur dans aggregate_data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# ENDPOINTS - Santé & Info
# ============================================================================

@api_app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Endpoint de santé pour monitoring"""
    return jsonify({
        'status': 'healthy',
        'service': 'NASA Dashboard API',
        'version': '1.0.0'
    }), 200


@api_app.route('/api/v1/info', methods=['GET'])
def get_info():
    """Informations sur l'API"""
    return jsonify({
        'service': 'NASA Spatial Biology Dashboard API',
        'version': '1.0.0',
        'endpoints': {
            'datasets': '/api/v1/datasets',
            'dataset_data': '/api/v1/datasets/<name>',
            'statistics': '/api/v1/datasets/<name>/statistics',
            'search': '/api/v1/search [POST]',
            'filter': '/api/v1/filter [POST]',
            'export': '/api/v1/export [POST]',
            'aggregate': '/api/v1/aggregate [POST]',
            'health': '/api/v1/health'
        }
    }), 200


# ============================================================================
# Gestion des erreurs
# ============================================================================

@api_app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404


@api_app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


# ============================================================================
# Lancement de l'API standalone
# ============================================================================

if __name__ == '__main__':
    logger.info("🚀 Démarrage de l'API REST...")
    logger.info("Documentation: http://localhost:5000/api/v1/info")
    api_app.run(debug=True, host='0.0.0.0', port=5000)