import json
from pathlib import Path
from typing import Any

import pandas as pd

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
            "recall": float(recall_score(y_real, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_real, y_pred, average="weighted", zero_division=0)),
        }

    def diagnostico(self, y_real, y_pred) -> dict[str, Any]:
        return {
            "matriz_confusion": confusion_matrix(y_real, y_pred),
            "reporte": classification_report(y_real, y_pred, output_dict=True, zero_division=0),
        }

    @staticmethod
    def guardar_comparacion(resultados: pd.DataFrame, ruta: str | Path, metrica: str) -> Path:
        destino = Path(ruta)
        destino.parent.mkdir(parents=True, exist_ok=True)
        contenido = {
            "metrica_seleccion": metrica,
            "mejor_modelo": str(resultados.iloc[0]["modelo"]),
            "modelos": {
                str(fila["modelo"]): {
                    nombre: float(fila[nombre])
                    for nombre in ("accuracy", "precision", "recall", "f1")
                }
                for _, fila in resultados.iterrows()
            },
        }
        # newline="\n" explicito: en Windows write_text traduce a CRLF, y como
        # DVC hashea los bytes del fichero, la misma etapa producia un hash en
        # Windows y otro en Linux. El pipeline se daba por obsoleto solo por
        # cambiar de maquina.
        destino.write_text(
            json.dumps(contenido, ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )
        return destino
