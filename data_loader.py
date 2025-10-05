"""
Module de chargement et manipulation des données NASA
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging
from config import DATA_DIR, CHUNK_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Classe pour charger et manipuler les données NASA"""
    
    def __init__(self):
        self.data_dir = DATA_DIR
        self.datasets = {}
        # self._load_all_datasets() # Ne pas charger au démarrage
    
    def load_all_datasets(self):
        """Charge tous les datasets CSV du répertoire spécifié."""
        for file_path in Path(self.data_dir).glob("*.csv"):
            try:
                df = pd.read_csv(file_path)
                self.datasets[file_path.stem] = df
                logger.info(f"Dataset '{file_path.stem}' chargé avec {len(df)} lignes à la demande.")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du fichier {file_path}: {e}")
    
    def get_dataset(self, name: str) -> Optional[pd.DataFrame]:
        """Récupère un dataset par son nom"""
        # Support virtual NCBI datasets using the prefix 'ncbi:' e.g. 'ncbi:BRCA1'
        if not name:
            return None
        if isinstance(name, str) and name.startswith("ncbi:"):
            # lazy import to avoid importing requests at module load if unused
            try:
                from ncbi_fetcher import search_and_fetch
                term = name.split(":", 1)[1]
                return search_and_fetch(term, retmax=100)
            except Exception as e:
                logger.exception(f"Erreur lors de la récupération NCBI pour {name}: {e}")
                return pd.DataFrame()
        return self.datasets.get(name)
    
    def get_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Retourne tous les datasets chargés"""
        return self.datasets
    
    def get_dataset_names(self) -> List[str]:
        """Retourne la liste des noms de datasets"""
        # Expose both local and virtual NCBI source
        return list(self.datasets.keys()) + ["ncbi:"]
    
    def filter_data(self, dataset_name: str, filters: Dict) -> pd.DataFrame:
        """
        Filtre un dataset selon les critères fournis
        
        Args:
            dataset_name: Nom du dataset
            filters: Dict avec les colonnes et valeurs à filtrer
        """
        df = self.get_dataset(dataset_name)
        if df is None:
            return pd.DataFrame()
        
        filtered_df = df.copy()
        
        for column, value in filters.items():
            if column in filtered_df.columns:
                if isinstance(value, list):
                    filtered_df = filtered_df[filtered_df[column].isin(value)]
                else:
                    filtered_df = filtered_df[filtered_df[column] == value]
        
        return filtered_df
    
    def get_unique_values(self, dataset_name: str, column: str) -> List:
        """Retourne les valeurs uniques d'une colonne"""
        df = self.get_dataset(dataset_name)
        if df is None or column not in df.columns:
            return []
        return sorted(df[column].dropna().unique().tolist())
    
    def get_statistics(self, dataset_name: str) -> Dict:
        """Calcule des statistiques de base sur un dataset"""
        df = self.get_dataset(dataset_name)
        if df is None:
            return {}
        
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        # Statistiques pour colonnes numériques
        numeric_stats = {}
        for col in stats['numeric_columns']:
            numeric_stats[col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max())
            }
        
        stats['numeric_stats'] = numeric_stats
        
        return stats
    
    def merge_datasets(self, dataset_names: List[str], 
                      on: str = None, how: str = 'inner') -> pd.DataFrame:
        """Fusionne plusieurs datasets"""
        if len(dataset_names) < 2:
            return self.get_dataset(dataset_names[0])
        
        result = self.get_dataset(dataset_names[0])
        
        for name in dataset_names[1:]:
            df = self.get_dataset(name)
            if df is not None:
                result = pd.merge(result, df, on=on, how=how)
        
        return result
    
    def aggregate_data(self, dataset_name: str, 
                      group_by: List[str], 
                      agg_dict: Dict) -> pd.DataFrame:
        """Agrège les données selon les critères fournis"""
        df = self.get_dataset(dataset_name)
        if df is None:
            return pd.DataFrame()
        
        return df.groupby(group_by).agg(agg_dict).reset_index()
    
    def export_to_csv(self, df: pd.DataFrame, filename: str):
        """Exporte un DataFrame vers un fichier CSV"""
        output_path = self.data_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Données exportées vers {output_path}")
        return output_path


# Instance globale
data_loader = DataLoader()
