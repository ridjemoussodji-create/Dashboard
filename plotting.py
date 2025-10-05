# plotting.py
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Optional


def _empty_plot(message: str = "Aucune donnée"):
    fig = go.Figure()
    fig.add_annotation(text=message, xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False,
                       font=dict(size=16, color='grey'))
    fig.update_layout(xaxis={'visible': False}, yaxis={'visible': False}, template='plotly_white')
    return fig


def create_scatter_plot(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None, size: Optional[str] = None):
    return px.scatter(df, x=x, y=y, color=color, size=size, template='plotly_white')


def create_bar_chart(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None):
    return px.bar(df, x=x, y=y, color=color, template='plotly_white')


def create_line_chart(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None):
    return px.line(df, x=x, y=y, color=color, template='plotly_white')


def create_histogram(df: pd.DataFrame, x: str):
    return px.histogram(df, x=x, template='plotly_white')


def create_box_plot(df: pd.DataFrame, y: str, x: Optional[str] = None, color: Optional[str] = None):
    return px.box(df, x=x, y=y, color=color, template='plotly_white')


def create_pie_chart(df: pd.DataFrame, names: str, values: str):
    return px.pie(df, names=names, values=values, template='plotly_white')


def create_heatmap(df: pd.DataFrame, x: str, y: str, z: Optional[str] = None):
    # Si z fourni, on pivote pour créer une heatmap des valeurs moyennes
    try:
        if z and z in df.columns and pd.api.types.is_numeric_dtype(df[z]):
            pivot = df.pivot_table(index=y, columns=x, values=z, aggfunc='mean')
            return go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns.astype(str), y=pivot.index.astype(str)),
                             layout=go.Layout(template='plotly_white'))
        else:
            # densité 2D
            return px.density_heatmap(df, x=x, y=y, template='plotly_white')
    except Exception:
        return _empty_plot("Impossible de créer la heatmap")


def create_3d_scatter(df: pd.DataFrame, x: str, y: str, z: str):
    return px.scatter_3d(df, x=x, y=y, z=z, color=z, template='plotly_white')


def create_correlation_matrix(df: pd.DataFrame):
    numeric = df.select_dtypes(include=['number'])
    if numeric.shape[1] == 0:
        return _empty_plot("Aucune colonne numérique pour la corrélation")
    corr = numeric.corr()
    return px.imshow(corr, text_auto=True, color_continuous_scale='RdBu', zmin=-1, zmax=1, template='plotly_white')


# Exposer un objet plot_generator pour compatibilité avec l'application
class _PlotGenerator:
    _empty_plot = staticmethod(_empty_plot)
    create_scatter_plot = staticmethod(create_scatter_plot)
    create_bar_chart = staticmethod(create_bar_chart)
    create_line_chart = staticmethod(create_line_chart)
    create_histogram = staticmethod(create_histogram)
    create_box_plot = staticmethod(create_box_plot)
    create_pie_chart = staticmethod(create_pie_chart)
    create_heatmap = staticmethod(create_heatmap)
    create_3d_scatter = staticmethod(create_3d_scatter)
    create_correlation_matrix = staticmethod(create_correlation_matrix)


plot_generator = _PlotGenerator()
