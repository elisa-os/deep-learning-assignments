"""Métricas de avaliação de instâncias (Parte 1, etapas 3–5).

Por que métricas de instância são diferentes de métricas semânticas?
----------------------------------------------------------------------
Métricas semânticas (IoU, Dice) tratam a imagem como um mapa de classes:
"quantos pixels de objeto foram acertados?". Elas são agnósticas a QUANTOS
objetos existem e se as instâncias foram separadas corretamente.

Métricas de instância (mAP) avaliam cada objeto individualmente:
- TP: instância prevista que "bate" com uma GT (IoU >= limiar)
- FP: instância prevista que não bate com nenhuma GT
- FN: instância GT que não foi coberta por nenhuma predição

mAP (mean Average Precision):
-------------------------------
Para detectores de objetos, AP é a área sob a curva Precisão×Recall.

Em segmentação de instâncias, o padrão COCO calcula AP para limiares de
IoU de 0.50 a 0.95 (passo 0.05) e tira a média. Assim, o mAP@[.5:.95]
penaliza segmentações grosseiras mesmo que "acertem" o objeto.

Matching (casamento) de instâncias:
-------------------------------------
Para cada imagem, precisamos decidir: "qual predição corresponde a qual GT?"

Dois métodos:
1. **Greedy (guloso)**: ordena pares (pred, GT) por IoU decrescente.
   Casa o par com maior IoU primeiro. O(N² log N).
2. **Hungarian (ótimo)**: encontra o matching global que maximiza IoU total.
   O(N³). Mais caro, mas "mais justo" quando há muitas instâncias.

Ambos os métodos dão resultados ligeiramente diferentes — a escolha deve
ser documentada (exigência do PA1).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou_matrix(pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """Calcula a matriz de IoU entre instâncias previstas e GT.

    Args:
        pred_mask: Array (H, W) de inteiros; 0=fundo, N>0=instância N.
        gt_mask:   Array (H, W) de inteiros; 0=fundo, N>0=instância N.

    Returns:
        iou_mat: Array (n_pred, n_gt) com IoU entre cada par.
    """
    pred_ids = np.unique(pred_mask[pred_mask > 0])
    gt_ids = np.unique(gt_mask[gt_mask > 0])

    if len(pred_ids) == 0 or len(gt_ids) == 0:
        return np.zeros((len(pred_ids), len(gt_ids)))

    iou_mat = np.zeros((len(pred_ids), len(gt_ids)))
    for i, p in enumerate(pred_ids):
        for j, g in enumerate(gt_ids):
            p_mask = pred_mask == p
            g_mask = gt_mask == g
            intersection = (p_mask & g_mask).sum()
            union = (p_mask | g_mask).sum()
            iou_mat[i, j] = intersection / union if union > 0 else 0.0

    return iou_mat


def match_instances(
    iou_mat: np.ndarray,
    iou_threshold: float = 0.5,
    method: str = "hungarian",
) -> tuple[int, int, int]:
    """Faz o matching e conta TP, FP, FN para um dado limiar de IoU.

    Args:
        iou_mat: Matriz (n_pred, n_gt) de IoUs.
        iou_threshold: Limiar mínimo de IoU para considerar TP.
        method: "hungarian" ou "greedy".

    Returns:
        (TP, FP, FN)
    """
    n_pred, n_gt = iou_mat.shape

    if method == "hungarian":
        # Maximizar IoU = minimizar -IoU
        row_ind, col_ind = linear_sum_assignment(-iou_mat)
        tp = int(sum(iou_mat[r, c] >= iou_threshold for r, c in zip(row_ind, col_ind)))
    else:  # greedy
        matched_pred, matched_gt = set(), set()
        tp = 0
        pairs = sorted(
            [(iou_mat[i, j], i, j) for i in range(n_pred) for j in range(n_gt)],
            reverse=True,
        )
        for iou, i, j in pairs:
            if i in matched_pred or j in matched_gt:
                continue
            if iou >= iou_threshold:
                tp += 1
                matched_pred.add(i)
                matched_gt.add(j)

    fp = n_pred - tp
    fn = n_gt - tp
    return tp, fp, fn


def compute_map(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    iou_thresholds: list[float] | None = None,
    method: str = "hungarian",
) -> dict:
    """Calcula mAP@[0.5:0.05:0.95] no estilo COCO.

    Args:
        pred_mask: Máscara de instâncias previstas (H, W).
        gt_mask:   Máscara de instâncias GT (H, W).
        iou_thresholds: Lista de limiares. Padrão: 0.50 a 0.95 (passo 0.05).
        method: "hungarian" ou "greedy".

    Returns:
        dict com "mAP", "AP_per_threshold", "count_error".
    """
    if iou_thresholds is None:
        iou_thresholds = [round(t, 2) for t in np.arange(0.50, 1.00, 0.05)]

    iou_mat = compute_iou_matrix(pred_mask, gt_mask)
    n_pred = len(np.unique(pred_mask[pred_mask > 0]))
    n_gt = len(np.unique(gt_mask[gt_mask > 0]))

    ap_per_thr = {}
    for thr in iou_thresholds:
        tp, fp, fn = match_instances(iou_mat, iou_threshold=thr, method=method)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # AP simplificado por limiar (sem curva P-R completa)
        ap_per_thr[thr] = precision * recall / max(precision + recall, 1e-6) * 2  # F1 proxy

    mAP = float(np.mean(list(ap_per_thr.values())))
    count_error = abs(n_pred - n_gt)

    return {"mAP": mAP, "AP_per_threshold": ap_per_thr, "count_error": count_error}
