"""Visualização e exportação de gráficos de avaliação e instâncias (PA1).

Adicionado para o grid qualitativo da Trilha A (1 × N ou 2 × N.imshow
da camada Interior e Fronteira quando o modelo tem out_channels >= 3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Paleta discreta para as 3 classes da Trilha A:
# 0 = fundo (transparente), 1 = interior, 2 = fronteira
CLASS_COLORS = np.array([
    [0.0, 0.0, 0.0, 0.0],     # 0 fundo
    [1.0, 0.2, 0.2, 0.55],    # 1 interior
    [1.0, 0.85, 0.0, 0.55],   # 2 fronteira
], dtype=np.float32)


def save_figure(fig: plt.Figure, filepath: str | Path, dpi: int = 100) -> Path:
    dest = Path(filepath)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.is_file():
        dest.unlink()
    fig.savefig(dest, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return dest


def show_instances(
    image: np.ndarray,
    instance_mask: np.ndarray,
    title: str = "",
    ax: plt.Axes | None = None,
) -> None:
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
    n_instances = int(instance_mask.max())
    if n_instances > 0:
        rng = np.random.default_rng(seed=0)
        colors = rng.random((n_instances + 1, 4))
        colors[:, 3] = 0.5
        colors[0] = [0, 0, 0, 0]
        ax.imshow(colors[instance_mask])
    ax.set_title(f"{title} ({n_instances} instâncias)" if title else f"{n_instances} instâncias")
    ax.axis("off")


def plot_synthetic_samples(
    dataset,
    save_path: str | Path,
    n_samples: int = 4,
    seed: int = 42,
) -> Path:
    fig, axes = plt.subplots(2, n_samples, figsize=(3.5 * n_samples, 7))
    for i in range(n_samples):
        sample_idx = i * (len(dataset) // max(n_samples, 1))
        batch_item = dataset[sample_idx]
        img_tensor = batch_item["image"]
        if img_tensor.ndim == 3 and img_tensor.shape[0] == 3:
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img = np.clip((img_tensor.numpy() * std + mean).transpose(1, 2, 0), 0, 1)
        else:
            img = img_tensor.squeeze(0).numpy()
        inst = batch_item["mask_instances"].numpy()
        n_inst = len(np.unique(inst[inst > 0]))
        ax_img = axes[0, i] if n_samples > 1 else axes[0]
        ax_inst = axes[1, i] if n_samples > 1 else axes[1]
        if img.ndim == 3:
            ax_img.imshow(img)
        else:
            ax_img.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax_img.set_title(f"Amostra #{sample_idx} ({n_inst} inst.)")
        ax_img.axis("off")
        ax_inst.imshow(inst, cmap="tab20")
        ax_inst.set_title(f"Instâncias #{sample_idx}")
        ax_inst.axis("off")
    plt.suptitle("Exemplos do Dataset Sintético (Imagem + Máscara de Instâncias)", fontsize=13)
    plt.tight_layout()
    return save_figure(fig, save_path)


def plot_training_results(
    history: dict[str, list[float]],
    density_list: Sequence[int],
    map_list: Sequence[float],
    save_path: str | Path,
    title: str = "Parte 0 - Teste Unitário Sintético (Mini U-Net)",
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], "b-o", ms=4, label="Treino")
    axes[0].set_title("Loss de Treinamento")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    val_iou = history.get("val_iou", [])
    val_dice = history.get("val_dice", [])
    if val_iou:
        eval_epochs = [
            int(round(e))
            for e in np.linspace(1, len(history["train_loss"]), len(val_iou))
        ]
        axes[1].plot(eval_epochs, val_iou, "g-o", ms=4, label="IoU Semântico")
        axes[1].plot(eval_epochs, val_dice, "r-s", ms=4, label="Dice Semântico")
    axes[1].set_title("Métricas Semânticas (Validação)")
    axes[1].set_xlabel("Época")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    if len(density_list) > 0 and len(map_list) > 0:
        d_arr = np.array(density_list)
        m_arr = np.array(map_list)
        axes[2].scatter(d_arr, m_arr, alpha=0.5, s=20, c="purple", edgecolors="none")
        if len(np.unique(d_arr)) > 1:
            try:
                z = np.polyfit(d_arr, m_arr, 1)
                x_line = np.linspace(d_arr.min(), d_arr.max(), 100)
                axes[2].plot(x_line, np.poly1d(z)(x_line), "k--", lw=1.5, label="Tendência")
            except Exception:
                pass
        axes[2].set_title("mAP vs. Densidade de Objetos")
        axes[2].set_xlabel("Nº de instâncias GT na imagem")
        axes[2].set_ylabel("mAP@[0.50:0.95]")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    return save_figure(fig, save_path)


def plot_qualitative_results(
    samples: list[dict],
    save_path: str | Path,
    title: str = "Qualitativo - Avaliação de Instâncias",
) -> Path:
    """Grid qualitativo para Trilha A.

    Cada coluna mostra:
      0: Imagem original
      1: GT de instâncias
      2: Predição de instâncias
      3: GT binário (foreground = interior+fronteira)
      4+: (Trilha A) Mapa de probabilidade do canal Interior e Fronteira,
          se disponíveis nos samples.

    Esperado nas amostras:
      - 'image', 'gt_instances', 'pred_instances', 'gt_binary',
        'mAP', 'idx', 'gt_3class', 'prob_map'
    """
    n_cols = len(samples)
    if n_cols == 0:
        return Path(save_path)

    row_labels = ["Imagem Original", "GT Instâncias", "Pred Instâncias", "Binária GT"]
    extra_rows = 0
    for s in samples:
        if s.get("gt_3class") is not None and s.get("prob_map") is not None and np.ndim(s["prob_map"]) == 3:
            extra_rows = 2
            break

    total_rows = 4 + extra_rows
    fig, axes = plt.subplots(total_rows, n_cols, figsize=(3.8 * n_cols, 3.2 * total_rows))

    for col_i, s in enumerate(samples):
        img_np = s["image"]
        inst_gt = s["gt_instances"]
        pred_lab = s["pred_instances"]
        binm_gt = s["gt_binary"]
        map_val = s.get("mAP", 0.0)
        idx_label = s.get("idx", col_i)

        n_gt = len(np.unique(inst_gt[inst_gt > 0]))
        n_pred = len(np.unique(pred_lab[pred_lab > 0]))

        ax0 = axes[0, col_i] if n_cols > 1 else axes[0]
        ax1 = axes[1, col_i] if n_cols > 1 else axes[1]
        ax2 = axes[2, col_i] if n_cols > 1 else axes[2]
        ax3 = axes[3, col_i] if n_cols > 1 else axes[3]

        ax0.imshow(img_np if img_np.ndim == 3 else img_np, cmap="gray", vmin=0, vmax=1)
        ax0.set_title(f"Amostra #{idx_label}")
        ax0.axis("off")

        ax1.imshow(inst_gt, cmap="tab20")
        ax1.set_title(f"GT ({n_gt} instâncias)")
        ax1.axis("off")

        ax2.imshow(pred_lab, cmap="tab20")
        ax2.set_title(f"Pred ({n_pred} inst.) | mAP: {map_val:.3f}")
        ax2.axis("off")

        ax3.imshow(binm_gt, cmap="gray")
        ax3.set_title("Máscara Binária GT")
        ax3.axis("off")

        gt_3class = s.get("gt_3class")
        prob_map = s.get("prob_map")
        if gt_3class is not None and prob_map is not None and extra_rows > 0:
            ax4 = axes[4, col_i] if n_cols > 1 else axes[4]
            ax5 = axes[5, col_i] if n_cols > 1 else axes[5]

            gt_colored = CLASS_COLORS[gt_3class.astype(np.int32)]
            ax4.imshow(gt_colored)
            ax4.set_title("GT 3 classes (int=vermelho, frt=amarelo)")
            ax4.axis("off")

            ax5.imshow(prob_map, cmap="viridis")
            ax5.set_title("Pred Interior (softmax canal 1)")
            ax5.axis("off")

    for r, lbl in enumerate(row_labels):
        target_ax = axes[r, 0] if n_cols > 1 else axes[r]
        target_ax.set_ylabel(lbl, fontsize=11, labelpad=10)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    return save_figure(fig, save_path)
