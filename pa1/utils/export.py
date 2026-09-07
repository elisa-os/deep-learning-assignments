"""Exportação de métricas de avaliação para CSV.

Classes
--------
PerImageMetricsWriter
    Coleta registros durante a avaliação e grava um CSV por imagem
    com IoU, Dice, mAP, erro de contagem e todos os tp / fp / fn / ap
    por limiar de IoU (0.50 .. 0.95, passo 0.05).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

# Limiares padrão usados pelo compute_map do PA1
_DEFAULT_THRESHOLD_PERCENTAGES = [
    int(round(t * 100)) for t in __import__("numpy").arange(0.50, 1.00, 0.05)
]


def _threshold_columns(threshold_percentages: list[int] | None = None) -> list[str]:
    """Nomes das colunas por limiar (tp, fp, fn, ap)."""
    cols: list[str] = []
    for p in (threshold_percentages or _DEFAULT_THRESHOLD_PERCENTAGES):
        tag = f"{p:02d}"
        cols.extend((f"tp_{tag}", f"fp_{tag}", f"fn_{tag}", f"ap_{tag}"))
    return cols


class PerImageMetricsWriter:
    """Acumula registros de métricas por imagem e exporta um CSV.

    Parâmetros
    ----------
    output_path : Path
        Diretório onde o CSV será salvo. O diretório é criado automaticamente
        (mkdir -p) na chamada a :meth:`write`.
    threshold_percentages : list[int] | None
        Porcentagem dos limiares de IoU para as colunas extras. Por padrão
        usa 50..95 com passo 5, igual ao compute_map.
    """

    def __init__(
        self,
        output_path: Path,
        threshold_percentages: list[int] | None = None,
    ) -> None:
        self._output_path = Path(output_path)
        self._threshold_percentages = threshold_percentages or _DEFAULT_THRESHOLD_PERCENTAGES
        self._records: list[dict[str, Any]] = []

    @property
    def fieldnames(self) -> list[str]:
        """Ordem garantida das colunas do CSV, sem depender da ordem de inserção."""
        common = ["idx", "n_gt", "n_pred", "count_error", "iou_sem", "dice_sem", "mAP"]
        return common + _threshold_columns(self._threshold_percentages)

    @property
    def record_count(self) -> int:
        """Quantidade de registros acumulados até o momento."""
        return len(self._records)

    def add(
        self,
        idx: int,
        n_gt: int,
        n_pred: int,
        count_error: int,
        iou_sem: float,
        dice_sem: float,
        mAP: float,
        per_threshold_details: dict[float, dict[str, int]],
        ap_per_threshold: dict[float, float],
    ) -> None:
        """Adiciona um registro para uma única imagem."""
        record: dict[str, Any] = {
            "idx": idx,
            "n_gt": n_gt,
            "n_pred": n_pred,
            "count_error": count_error,
            "iou_sem": round(iou_sem, 6),
            "dice_sem": round(dice_sem, 6),
            "mAP": round(mAP, 6),
        }
        for thr in sorted(per_threshold_details.keys()):
            det = per_threshold_details[thr]
            p = int(round(thr * 100))
            tag = f"{p:02d}"
            record[f"tp_{tag}"] = det["tp"]
            record[f"fp_{tag}"] = det["fp"]
            record[f"fn_{tag}"] = det["fn"]
            record[f"ap_{tag}"] = round(ap_per_threshold[thr], 6)
        self._records.append(record)

    def write(self, filename: str = "per_image_instance_metrics.csv") -> Path:
        """Grava o CSV e retorna o caminho do arquivo criado."""
        self._output_path.mkdir(parents=True, exist_ok=True)
        csv_path = self._output_path / filename
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.fieldnames))
            writer.writeheader()
            writer.writerows(self._records)
        return csv_path
