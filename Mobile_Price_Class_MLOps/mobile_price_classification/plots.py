"""
Gráficos de EDA y evaluación de modelos.

Todas las figuras se guardan automáticamente en `reports/figures/` para
mantener trazabilidad, además de mostrarse en el notebook.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from mobile_price_classification.config import FIGURES_DIR, TARGET_COLUMN, logger

sns.set_style("whitegrid")


@dataclass
class EDAPlotter:
    """Agrupa todos los gráficos usados en el EDA y la evaluación de modelos."""

    figures_dir: Path = field(default_factory=lambda: FIGURES_DIR)
    save: bool = True

    def _save(self, fig: plt.Figure, name: str) -> None:
        if self.save:
            path = self.figures_dir / f"{name}.png"
            fig.savefig(path, bbox_inches="tight", dpi=120)
            logger.info(f"Figura guardada en {path}")

    def plot_target_distribution(self, df: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.countplot(x=TARGET_COLUMN, data=df, palette="viridis", ax=ax)
        ax.set_title("Distribución de price_range")
        ax.set_xlabel("Rango de precio (0=bajo, 1=medio, 2=alto, 3=muy alto)")
        ax.set_ylabel("Cantidad de móviles")
        self._save(fig, "target_distribution")
        plt.show()

    def plot_numeric_distributions(self, df: pd.DataFrame, exclude: Sequence[str] = ()) -> None:
        cols = [c for c in df.columns if c not in exclude]
        fig = df[cols].hist(bins=25, figsize=(16, 12))[0][0].figure
        plt.suptitle("Distribución de variables numéricas", y=1.02)
        plt.tight_layout()
        self._save(fig, "numeric_distributions")
        plt.show()

    def plot_binary_features(self, df: pd.DataFrame, binary_cols: Sequence[str]) -> None:
        n = len(binary_cols)
        ncols = 3
        nrows = -(-n // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
        axes = axes.flatten()
        for i, col in enumerate(binary_cols):
            sns.countplot(x=col, hue=TARGET_COLUMN, data=df, palette="viridis", ax=axes[i])
            axes[i].set_title(col)
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        self._save(fig, "binary_features")
        plt.show()

    def plot_correlation_matrix(self, df: pd.DataFrame) -> pd.Series:
        fig, ax = plt.subplots(figsize=(14, 10))
        corr = df.corr()
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Matriz de correlación")
        self._save(fig, "correlation_matrix")
        plt.show()
        return corr[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(ascending=False)

    def plot_correlation_with_target(self, corr_target: pd.Series) -> None:
        fig, ax = plt.subplots(figsize=(8, 8))
        corr_target.plot(kind="barh", color="teal", ax=ax)
        ax.set_title(f"Correlación de las variables con {TARGET_COLUMN}")
        ax.set_xlabel("Coeficiente de correlación")
        self._save(fig, "correlation_with_target")
        plt.show()

    def plot_boxplot_vs_target(self, df: pd.DataFrame, column: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(x=TARGET_COLUMN, y=column, data=df, palette="viridis", ax=ax)
        ax.set_title(f"{column} vs {TARGET_COLUMN}")
        self._save(fig, f"boxplot_{column}_vs_target")
        plt.show()

    def plot_scatter(self, df: pd.DataFrame, x: str, y: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(x=x, y=y, hue=TARGET_COLUMN, data=df, palette="viridis", alpha=0.6, ax=ax)
        ax.set_title(f"{x} vs {y} por {TARGET_COLUMN}")
        self._save(fig, f"scatter_{x}_vs_{y}")
        plt.show()

    def plot_outliers(self, df: pd.DataFrame, columns: Sequence[str]) -> None:
        n = len(columns)
        ncols = 3
        nrows = -(-n // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
        axes = axes.flatten()
        for i, col in enumerate(columns):
            sns.boxplot(y=df[col], ax=axes[i], color="skyblue")
            axes[i].set_title(col)
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        self._save(fig, "outliers")
        plt.show()

    def plot_model_comparison(self, results_df: pd.DataFrame, metric: str = "CV Accuracy (media)") -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=results_df, x=metric, y="Modelo", palette="viridis", ax=ax)
        ax.set_title(f"Comparación de modelos ({metric})")
        ax.set_xlim(0, 1)
        self._save(fig, "model_comparison")
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, labels: Optional[Sequence[int]] = None) -> None:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Matriz de confusión - Modelo final")
        self._save(fig, "confusion_matrix")
        plt.show()

    def plot_feature_importance(self, importances: pd.Series) -> None:
        fig, ax = plt.subplots(figsize=(8, 8))
        importances.sort_values().plot(kind="barh", color="darkgreen", ax=ax)
        ax.set_title("Importancia de variables (Random Forest)")
        self._save(fig, "feature_importance")
        plt.show()
