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
import torch
import torch.nn as nn
import torch.optim as optim

from pa1.config import load_config, Config
from pa1.utils import set_seed, get_device
from pa1.data import make_synthetic_loader
from pa1.models import UNet
from pa1.losses import BCEDiceLoss
from pa1.metrics import compute_map
from pa1.postprocessing import semantic_to_instances


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
    """Executa uma época de treino.

    Loop padrão de treino em PyTorch:
    1. Zera o gradiente acumulado (optimizer.zero_grad)
    2. Forward pass: model(x) → logits
    3. Calcula a loss
    4. Backward pass: loss.backward() → acumula ∂loss/∂θ em cada parâmetro
    5. Atualiza pesos: optimizer.step() → θ ← θ - lr × gradiente

    Returns:
        loss média da época.
    """
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
) -> dict:
    """Avalia o modelo: calcula mAP e erro de contagem médios.

    @torch.no_grad(): desativa o cálculo de gradientes durante a avaliação.
    Isso economiza memória e acelera o forward pass.

    Returns:
        dict com "mean_mAP" e "mean_count_error".
    """
    model.eval()
    maps, count_errors = [], []

    for batch in loader:
        images = batch["image"].to(device)
        gt_instances = batch["mask_instances"].numpy()  # (B, H, W)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # (B, H, W)

        for i in range(images.size(0)):
            pred_inst = semantic_to_instances(probs[i])
            result = compute_map(pred_inst, gt_instances[i])
            maps.append(result["mAP"])
            count_errors.append(result["count_error"])

    return {
        "mean_mAP": float(sum(maps) / len(maps)) if maps else 0.0,
        "mean_count_error": float(sum(count_errors) / len(count_errors)) if count_errors else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Parsing de argumentos & merge com YAML
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PA1 — Segmentação de Instâncias",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Caminho do arquivo YAML de configuração.",
    )

    # Flags opcionais para override pontual
    p.add_argument("--synthetic", dest="synthetic", action="store_true", default=None)
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
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {args.config}")

    cfg = load_config(config_path)

    # Overrides opcionais
    if args.synthetic is not None:
        cfg.data.synthetic = args.synthetic
    if args.data_dir is not None:
        cfg.data.data_dir = args.data_dir
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
    elif cfg.data.data_dir:
        raise NotImplementedError("Integração com dados reais ainda não implementada.")
    else:
        raise ValueError("Defina `synthetic: true` ou passe um `data_dir` na configuração.")

    # ---- Modelo ----
    model = UNet(
        in_channels=cfg.model.in_channels,
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

    # ---- Treino ----
    if not cfg.train.eval_only:
        print(f"\nTreinando por {cfg.train.epochs} épocas (lr={cfg.train.lr})...\n")
        t0 = time.time()
        for epoch in range(1, cfg.train.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            if epoch % 5 == 0 or epoch == cfg.train.epochs:
                metrics = evaluate(model, val_loader, device)
                elapsed = time.time() - t0
                print(
                    f"Época {epoch:3d}/{cfg.train.epochs} | "
                    f"Loss: {train_loss:.4f} | "
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

    # ---- Avaliação final ----
    print("\n=== Avaliação Final (val) ===")
    final_metrics = evaluate(model, val_loader, device)
    print(f"mAP: {final_metrics['mean_mAP']:.4f}")
    print(f"Erro de contagem médio: {final_metrics['mean_count_error']:.2f}")


if __name__ == "__main__":
    main()
