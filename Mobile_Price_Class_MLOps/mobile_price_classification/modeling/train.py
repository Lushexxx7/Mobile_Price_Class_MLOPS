"""
Entrenamiento, comparación y ajuste de modelos de clasificación.

Los modelos de scikit-learn se serializan con `joblib` (formato .pkl),
que es el estándar para este tipo de estimadores. El formato .h5 se
reserva para redes neuronales (Keras/TensorFlow); si en el futuro se
incorpora una, se recomienda un módulo aparte (p. ej. `train_nn.py`)
que use `model.save(path.with_suffix(".h5"))`.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from mobile_price_classification.config import (
    BEST_MODEL_PATH,
    BEST_PARAMS_PATH,
    CV_FOLDS,
    MODELS_COMPARISON_PATH,
    RANDOM_STATE,
    logger,
)


def default_models(random_state: int = RANDOM_STATE) -> Dict[str, ClassifierMixin]:
    """Diccionario de modelos candidatos por defecto."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=9),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "SVM": SVC(kernel="rbf", random_state=random_state),
    }


@dataclass
class ModelTrainer:
    """Entrena y compara varios modelos de clasificación, y ajusta el mejor.

    Ejemplo
    -------
    >>> trainer = ModelTrainer()
    >>> results_df = trainer.compare_models(X_train_scaled, y_train, X_val_scaled, y_val, X_full_scaled, y)
    >>> best_model = trainer.tune_best_model(X_full_scaled, y, model_name="SVM", param_grid={...})
    >>> trainer.save_model(best_model)
    """

    models: Dict[str, ClassifierMixin] = field(default_factory=default_models)
    cv_folds: int = CV_FOLDS
    random_state: int = RANDOM_STATE
    results_: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    def compare_models(
        self,
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        X_full_scaled=None,
        y_full=None,
    ) -> pd.DataFrame:
        """Entrena cada modelo, evalúa en validación y corre CV sobre el set completo."""
        from sklearn.metrics import accuracy_score

        if X_full_scaled is None:
            X_full_scaled, y_full = X_train_scaled, y_train

        rows = []
        for name, model in self.models.items():
            logger.info(f"Entrenando {name}...")
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            val_acc = accuracy_score(y_val, y_pred)

            cv_scores = cross_val_score(
                model, X_full_scaled, y_full, cv=self.cv_folds, scoring="accuracy"
            )

            rows.append(
                {
                    "Modelo": name,
                    "Accuracy (validación)": val_acc,
                    "CV Accuracy (media)": cv_scores.mean(),
                    "CV Accuracy (std)": cv_scores.std(),
                }
            )

        self.results_ = (
            pd.DataFrame(rows).sort_values("CV Accuracy (media)", ascending=False).reset_index(drop=True)
        )
        self._save_comparison(self.results_)
        return self.results_

    def tune_best_model(
        self,
        X_scaled,
        y,
        model_name: str,
        param_grid: dict,
        base_estimator: Optional[ClassifierMixin] = None,
    ) -> ClassifierMixin:
        """Ajuste fino de hiperparámetros (GridSearchCV) sobre el modelo indicado."""
        estimator = base_estimator or self.models[model_name].__class__(random_state=self.random_state) \
            if "random_state" in self.models[model_name].get_params() else self.models[model_name]

        logger.info(f"Ejecutando GridSearchCV para {model_name}...")
        grid = GridSearchCV(estimator, param_grid, cv=self.cv_folds, scoring="accuracy", n_jobs=-1)
        grid.fit(X_scaled, y)

        logger.info(f"Mejores parámetros: {grid.best_params_}")
        logger.info(f"Mejor accuracy (CV): {grid.best_score_:.4f}")

        BEST_PARAMS_PATH.write_text(
            json.dumps({"model": model_name, "best_params": grid.best_params_,
                        "best_cv_score": grid.best_score_}, indent=2)
        )
        return grid.best_estimator_

    def save_model(self, model: ClassifierMixin, path: Path = BEST_MODEL_PATH) -> None:
        """Serializa el modelo entrenado con joblib (formato .pkl)."""
        joblib.dump(model, path)
        logger.info(f"Modelo guardado en {path}")

    def _save_comparison(self, results_df: pd.DataFrame) -> None:
        results_df.to_json(MODELS_COMPARISON_PATH, orient="records", indent=2)
        logger.info(f"Comparación de modelos guardada en {MODELS_COMPARISON_PATH}")
