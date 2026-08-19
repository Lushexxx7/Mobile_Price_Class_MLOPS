from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

class EvaluadorModelo:
    def evaluar(self, y_real, y_pred) -> dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_real, y_pred)),
            "precision": float(
                precision_score(y_real, y_pred, average="weighted", zero_division=0)
            ),
            "recall": float(
                recall_score(y_real, y_pred, average="weighted", zero_division=0)
            ),
            "f1": float(
                f1_score(y_real, y_pred, average="weighted", zero_division=0)
            ),
        }

    def diagnostico(self, y_real, y_pred) -> dict[str, Any]:
        return {
            "matriz_confusion": confusion_matrix(y_real, y_pred),
            "reporte": classification_report(
                y_real, y_pred, output_dict=True, zero_division=0
            ),
        }
