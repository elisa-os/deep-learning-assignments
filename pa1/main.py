"""Ponto de entrada principal do PA1.

Uso:
    # Rodar com configuração padrão do arquivo configs/pa1.yaml:
    uv run pa1

    # Especificar outro arquivo de configuração:
    uv run pa1 --config configs/outro.yaml

    # Sobrescrever valores via flag se desejar (opcional):
    uv run pa1 --epochs 5 --lr 1e-4
"""

import argparse
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pa1.config import load_config, Config
from pa1.utils import (
    set_seed,
    get_device,
    plot_synthetic_samples,
    plot_training_results,
    plot_qualitative_results,
)
from pa1.data import make_synthetic_loader, make_dsb2018_loaders
from pa1.models import UNet
from pa1.losses import BCEDiceLoss
from pa1.metrics import compute_map
from pa1.postprocessing import semantic_to_instances


# ─────────────────────────────────────────────────────────────────────────────
# Métricas semânticas auxiliares
# ─────────────────────────────────────────────────────────────────────────────

def compute_semantic_metrics(pred_binary: np.ndarray, gt_binary: np.ndarray) -> tuple[float, float]:
    """Calcula IoU e Dice para segmentação semântica binária."""
    p = pred_binary.astype(bool)
    g = gt_binary.astype(bool)
    intersection = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    iou = float(intersection / union) if union > 0 else 1.0
    dice = float(2.0 * intersection / (p.sum() + g.sum())) if (p.sum() + g.sum()) > 0 else 1.0
    return iou, dice


# ─────────────────────────────────────────────────────────────────────────────
# Treino (loop padrão PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Executa uma época de treino e retorna a loss média."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask_semantic"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    max_qualitative_samples: int = 4,
) -> dict:
    """Avalia o modelo calculando mAP, erro de contagem, IoU e Dice semânticos.

    Returns:
        dict com métricas médias, distribuições para gráficos e amostras qualitativas.
    """
    model.eval()
    maps, count_errors = [], []
    ious, dices = [], []
    densities = []
    qualitative_samples = []

    for batch in loader:
        images = batch["image"].to(device)
        gt_instances = batch["mask_instances"].numpy()  # (B, H, W)
        gt_semantics = batch["mask_semantic"].numpy()   # (B, H, W)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # (B, H, W)

        for i in range(images.size(0)):
            prob_map = probs[i]
            pred_inst = semantic_to_instances(prob_map)
            gt_inst = gt_instances[i]
            gt_sem = gt_semantics[i]

            # Métrica de instância (mAP e contagem)
            result = compute_map(pred_inst, gt_inst)
            maps.append(result["mAP"])
            count_errors.append(result["count_error"])

            # Densidade de objetos (número de núcleos/elipses GT)
            n_gt_objects = len(np.unique(gt_inst[gt_inst > 0]))
            densities.append(n_gt_objects)

            # Métricas semânticas (IoU e Dice)
            pred_bin = prob_map >= 0.5
            iou, dice = compute_semantic_metrics(pred_bin, gt_sem)
            ious.append(iou)
            dices.append(dice)

            # Coleta amostras para o grid qualitativo
            if len(qualitative_samples) < max_qualitative_samples:
                raw_img = images[i].cpu().numpy()
                if raw_img.ndim == 3 and raw_img.shape[0] == 3:
                    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                    disp_img = np.clip((raw_img * std + mean).transpose(1, 2, 0), 0, 1)
                else:
                    disp_img = raw_img[0] if raw_img.ndim == 3 else raw_img

                qualitative_samples.append({
                    "image": disp_img,
                    "gt_instances": gt_inst,
                    "pred_instances": pred_inst,
                    "gt_binary": gt_sem.astype(np.uint8),
                    "mAP": result["mAP"],
                    "idx": len(qualitative_samples) + 1,
                })

    return {
        "mean_mAP": float(np.mean(maps)) if maps else 0.0,
        "mean_count_error": float(np.mean(count_errors)) if count_errors else 0.0,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_dice": float(np.mean(dices)) if dices else 0.0,
        "densities": densities,
        "maps": maps,
        "samples": qualitative_samples,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Parsing de argumentos & merge com YAML
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PA1 — Segmentação de Instâncias",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_cfg = "pa1/config.yaml" if Path("pa1/config.yaml").exists() else "config.yaml"
    p.add_argument(
        "--config",
        type=str,
        default=default_cfg,
        help="Caminho do arquivo YAML de configuração.",
    )

    # Flags opcionais para override pontual
    p.add_argument("--synthetic", dest="synthetic", action="store_true", default=None)
    p.add_argument("--no-synthetic", dest="synthetic", action="store_false")
    p.add_argument("--data-dir", dest="data_dir", type=str, default=None)
    p.add_argument("--epochs", dest="epochs", type=int, default=None)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    p.add_argument("--lr", dest="lr", type=float, default=None)
    p.add_argument("--eval-only", dest="eval_only", action="store_true", default=None)
    p.add_argument("--checkpoint", dest="checkpoint", type=str, default=None)
    p.add_argument("--seed", dest="seed", type=int, default=None)

    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    config_path = Path(args.config)
    if not config_path.exists():
        if Path("pa1/config.yaml").exists():
            config_path = Path("pa1/config.yaml")
        elif Path("config.yaml").exists():
            config_path = Path("config.yaml")
        else:
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {args.config}")

    cfg = load_config(config_path)

    # Overrides opcionais
    if args.synthetic is not None:
        cfg.data.synthetic = args.synthetic
    if args.data_dir is not None:
        cfg.data.data_dir = args.data_dir
        if args.synthetic is None:
            cfg.data.synthetic = False
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.eval_only is not None:
        cfg.train.eval_only = args.eval_only
    if args.checkpoint is not None:
        cfg.train.checkpoint = args.checkpoint
    if args.seed is not None:
        cfg.seed = args.seed

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    set_seed(cfg.seed)
    device = get_device()

    output_path = Path(cfg.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ---- Dados ----
    in_channels = cfg.model.in_channels

    if cfg.data.synthetic:
        print("[Modo Sintético] Gerando dataset de elipses...")
        train_loader = make_synthetic_loader(
            n_samples=cfg.data.n_samples,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            seed=cfg.seed,
            split="train",
        )
        val_loader = make_synthetic_loader(
            n_samples=cfg.data.n_samples,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            seed=cfg.seed,
            split="val",
        )

        samples_fig = output_path / "synthetic_samples.png"
        saved_samples_path = plot_synthetic_samples(train_loader.dataset, samples_fig)
        print(f"Amostras do dataset salvas em: {saved_samples_path}")

    elif cfg.data.data_dir:
        print(f"[Modo Real DSB2018] Carregando dataset a partir de {cfg.data.data_dir}...")
        train_loader, val_loader, test_loader = make_dsb2018_loaders(
            data_dir=cfg.data.data_dir,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            seed=cfg.seed,
        )
        in_channels = 3  # Imagens RGB do DSB2018

        samples_fig = output_path / "synthetic_samples.png"
        saved_samples_path = plot_synthetic_samples(train_loader.dataset, samples_fig)
        print(f"Amostras do dataset salvas em: {saved_samples_path}")

    else:
        raise ValueError("Defina `synthetic: true` ou passe um `data_dir` na configuração.")

    # ---- Modelo ----
    model = UNet(
        in_channels=in_channels,
        out_channels=cfg.model.out_channels,
    ).to(device)

    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)

    if cfg.train.checkpoint:
        ckpt_path = Path(cfg.train.checkpoint)
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"Checkpoint carregado: {ckpt_path}")
        else:
            print(f"Checkpoint {ckpt_path} não encontrado, iniciando pesos do zero.")

    history = {
        "train_loss": [],
        "val_iou": [],
        "val_dice": [],
        "val_map": [],
        "val_count_error": [],
    }

    # ---- Treino ----
    if not cfg.train.eval_only:
        print(f"\nTreinando por {cfg.train.epochs} épocas (lr={cfg.train.lr})...\n")
        t0 = time.time()
        for epoch in range(1, cfg.train.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            history["train_loss"].append(train_loss)

            if epoch % 5 == 0 or epoch == cfg.train.epochs:
                metrics = evaluate(model, val_loader, device)
                history["val_iou"].append(metrics["mean_iou"])
                history["val_dice"].append(metrics["mean_dice"])
                history["val_map"].append(metrics["mean_mAP"])
                history["val_count_error"].append(metrics["mean_count_error"])

                elapsed = time.time() - t0
                print(
                    f"Época {epoch:3d}/{cfg.train.epochs} | "
                    f"Loss: {train_loss:.4f} | "
                    f"Val IoU: {metrics['mean_iou']:.3f} | "
                    f"mAP: {metrics['mean_mAP']:.3f} | "
                    f"Count err: {metrics['mean_count_error']:.1f} | "
                    f"Tempo: {elapsed:.1f}s"
                )

        total_time = time.time() - t0
        print(f"\nTreino concluído em {total_time:.1f}s ({total_time/60:.1f} min)")

        if cfg.train.checkpoint:
            ckpt_path = Path(cfg.train.checkpoint)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint salvo em: {ckpt_path}")

    # ---- Avaliação final e salvamento de imagens ----
    print("\n=== Avaliação Final (val) ===")
    final_metrics = evaluate(model, val_loader, device)
    print(f"IoU Semântico Médio: {final_metrics['mean_iou']:.4f}")
    print(f"Dice Semântico Médio: {final_metrics['mean_dice']:.4f}")
    print(f"mAP@[0.50:0.95]: {final_metrics['mean_mAP']:.4f}")
    print(f"Erro de contagem médio: {final_metrics['mean_count_error']:.2f}")

    # Se não rodou treino completo (ex: eval-only), preenche loss fictícia para o plot
    if not history["train_loss"]:
        history["train_loss"] = [0.0]
        history["val_iou"] = [final_metrics["mean_iou"]]
        history["val_dice"] = [final_metrics["mean_dice"]]

    # 1. Salva curvas de treino + mAP vs. Densidade diretamente em outputs/
    results_fig = output_path / "parte0_resultados.png"
    saved_results_path = plot_training_results(
        history=history,
        density_list=final_metrics["densities"],
        map_list=final_metrics["maps"],
        save_path=results_fig,
    )
    print(f"Gráficos de resultados e densidade salvos em: {saved_results_path}")

    # 2. Salva grid qualitativo 4x4 diretamente em outputs/
    qualitative_fig = output_path / "parte0_qualitativo.png"
    saved_qualitative_path = plot_qualitative_results(
        samples=final_metrics["samples"],
        save_path=qualitative_fig,
    )
    print(f"Grid qualitativo salvo em: {saved_qualitative_path}")


if __name__ == "__main__":
    main()
