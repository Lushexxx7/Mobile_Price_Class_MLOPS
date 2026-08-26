import json

from src.config import TARGET, TEST_PATH, TRAIN_PATH, VALIDATION_PATH
from src.data.load_data import CargadorDatos


def main() -> None:
    train = CargadorDatos(TRAIN_PATH).cargar()
    test = CargadorDatos(TEST_PATH).cargar()
    if TARGET not in train.columns:
        raise ValueError(f"Falta la variable objetivo {TARGET} en train.csv")
    faltantes = sorted(set(train.columns) - {TARGET} - set(test.columns))
    if faltantes:
        raise ValueError(f"Faltan columnas en test.csv: {faltantes}")
    reporte = {
        "train": {"filas": len(train), "columnas": len(train.columns), "nulos": int(train.isna().sum().sum()), "duplicados": int(train.duplicated().sum())},
        "test": {"filas": len(test), "columnas": len(test.columns), "nulos": int(test.isna().sum().sum()), "duplicados": int(test.duplicated().sum())},
    }
    VALIDATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": sin esto Windows escribe CRLF y el hash que calcula DVC
    # deja de coincidir con el que produce la misma etapa en Linux.
    VALIDATION_PATH.write_text(
        json.dumps(reporte, indent=2), encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
