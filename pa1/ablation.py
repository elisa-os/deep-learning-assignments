"""Runner de ablações sistemáticas — Parte 3 do PA1.

Executa as ablações dos dois eixos com múltiplas seeds e salva
os resultados agregados em ``pa1/outputs/ablation/``.

Eixo 1 — Recuperação de Resolução:
    Compara U-Net com skip connections (padrão U-Net) vs. decoder simples
    sem skips (estilo SegNet), usando Focal Loss com γ=2 fixo.

Eixo 2 — Função de Perda (Focal Loss γ):
    Varia o parâmetro γ ∈ {0, 1, 2, 5} com a U-Net completa (com skips).
    γ=0 recupera exatamente a Cross-Entropy ponderada.

Seeds: [42, 123] — média e desvio padrão reportados por configuração.

Nota: as configurações carregam a seção ``parte2`` do config.yaml como
base (DSB2018, 3 classes, Focal Loss) e sobrescrevem apenas os parâmetros
de ablação programaticamente — sem precisar adicionar novas seções ao YAML.

Uso:
    bash run.sh pa1-ablation                  # 20 épocas por run (padrão)
    bash run.sh pa1-ablation --epochs 10      # smoke test rápido
    bash run.sh pa1-ablation --batch-size 4   # batch menor (menos VRAM)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from pa1.config import load_config
from pa1.data import make_dsb2018_loaders
from pa1.losses import BCEDiceLoss, FocalLoss
from pa1.metrics import compute_map
from pa1.models import UNet
from pa1.postprocessing import watershed_to_instances
from pa1.utils import get_device, set_seed

# ─────────────────────────────────────────────────────────────────────────────
# Importa funções reutilizáveis do pipeline principal
# ─────────────────────────────────────────────────────────────────────────────
from pa1.main import train_one_epoch, evaluate

# ─────────────────────────────────────────────────────────────────────────────
# Matriz de ablações
# ─────────────────────────────────────────────────────────────────────────────

SEEDS = [42, 123]

# Cada entrada: (eixo, nome_legível, nome_arquivo, use_skips, gamma)
ABLATIONS = [
    # Eixo 1 — Recuperação de Resolução
    ("eixo1", "Com Skips (U-Net padrão)", "eixo1_com_skips",  True,  2.0),
    ("eixo1", "Sem Skips (decoder simpl.)", "eixo1_sem_skips", False, 2.0),
    # Eixo 2 — Função de Perda
    ("eixo2", "γ=0 (CE ponderada)",        "eixo2_gamma0",    True,  0.0),
    ("eixo2", "γ=1",                        "eixo2_gamma1",    True,  1.0),
    ("eixo2", "γ=2 (baseline Trilha A)",    "eixo2_gamma2",    True,  2.0),
    ("eixo2", "γ=5",                        "eixo2_gamma5",    True,  5.0),
]

# alpha padrão por classe (fundo / interior / fronteira)
DEFAULT_ALPHA = [1.0, 1.0, 2.5]


# ─────────────────────────────────────────────────────────────────────────────
# Runner de uma única configuração
# ─────────────────────────────────────────────────────────────────────────────

def run_single(
    use_skips: bool,
    gamma: float,
    seed: int,
    base_cfg,
    device: torch.device,
    output_path: Path,
    run_name: str,
    print_every: int = 5,
) -> dict:
    """Treina e avalia uma configuração de ablação com uma seed específica."""
    set_seed(seed)

    # ── Dados ────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = make_dsb2018_loaders(
        data_dir=base_cfg.data.data_dir,
        batch_size=base_cfg.data.batch_size,
        num_workers=base_cfg.data.num_workers,
        seed=seed,
    )

    # ── Modelo ───────────────────────────────────────────────────────────────
    model = UNet(
        in_channels=3,
        out_channels=3,          # sempre 3 classes (Trilha A)
        use_skips=use_skips,
    ).to(device)

    # ── Loss ─────────────────────────────────────────────────────────────────
    criterion = FocalLoss(alpha=DEFAULT_ALPHA, gamma=gamma, reduction="mean")

    # ── Otimizador ────────────────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=base_cfg.train.lr)

    # ── Treino ───────────────────────────────────────────────────────────────
    epochs = base_cfg.train.epochs
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        if epoch % print_every == 0 or epoch == epochs:
            metrics = evaluate(model, val_loader, device, max_qualitative_samples=0)
            elapsed = time.time() - t0
            print(
                f"    Época {epoch:3d}/{epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Val mAP: {metrics['mean_mAP']:.3f} | "
                f"Count err: {metrics['mean_count_error']:.1f} | "
                f"Tempo: {elapsed:.0f}s"
            )

    # ── Avaliação final ───────────────────────────────────────────────────────
    final = evaluate(model, val_loader, device, max_qualitative_samples=0)

    # ── Salva checkpoint ──────────────────────────────────────────────────────
    ckpt = output_path / f"{run_name}_seed{seed}.pt"
    torch.save(model.state_dict(), ckpt)

    return {
        "mAP": final["mean_mAP"],
        "count_error": final["mean_count_error"],
        "iou": final["mean_iou"],
        "dice": final["mean_dice"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualizações
# ─────────────────────────────────────────────────────────────────────────────

def _bar_chart(
    labels: list[str],
    means: list[float],
    stds: list[float],
    title: str,
    ylabel: str,
    save_path: Path,
    colors: list[str] | None = None,
) -> None:
    """Gráfico de barras com barras de erro (μ ± σ)."""
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    x = np.arange(len(labels))
    palette = colors or ["#4C72B0"] * len(labels)
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=palette, alpha=0.85, edgecolor="k", linewidth=0.8)

    # Anota o valor em cima de cada barra
    for bar, m, s in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s + 0.005,
            f"{m:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, min(1.05, max(means) + max(stds) + 0.08))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Gráfico salvo em: {save_path}")


def plot_ablation_results(
    results: dict,
    output_path: Path,
) -> None:
    """Gera os gráficos de barras para Eixo 1 e Eixo 2."""

    def _extract(axis: str, metric: str):
        labs, mus, sigs = [], [], []
        for abl in ABLATIONS:
            eixo, label, slug, *_ = abl
            if eixo != axis:
                continue
            seed_vals = [results[slug][f"seed{s}"][metric] for s in SEEDS]
            labs.append(label)
            mus.append(float(np.mean(seed_vals)))
            sigs.append(float(np.std(seed_vals)))
        return labs, mus, sigs

    # ── Eixo 1: mAP ──────────────────────────────────────────────────────────
    labs1, mus1, sigs1 = _extract("eixo1", "mAP")
    colors1 = ["#2196F3", "#FF5722"]  # azul = com skips, laranja = sem skips
    _bar_chart(
        labs1, mus1, sigs1,
        title="Eixo 1 — Skip Connections: mAP@[0.50:0.95] (μ ± σ, 2 seeds)",
        ylabel="mAP@[0.50:0.95]",
        save_path=output_path / "ablation_eixo1_map.png",
        colors=colors1,
    )

    # ── Eixo 2: mAP ──────────────────────────────────────────────────────────
    labs2, mus2, sigs2 = _extract("eixo2", "mAP")
    colors2 = ["#9E9E9E", "#8BC34A", "#2196F3", "#F44336"]  # γ 0,1,2,5
    _bar_chart(
        labs2, mus2, sigs2,
        title="Eixo 2 — Focal Loss γ: mAP@[0.50:0.95] (μ ± σ, 2 seeds)",
        ylabel="mAP@[0.50:0.95]",
        save_path=output_path / "ablation_eixo2_map.png",
        colors=colors2,
    )

    # ── Eixo 2: Erro de contagem ──────────────────────────────────────────────
    labs2c, mus2c, sigs2c = _extract("eixo2", "count_error")
    _bar_chart(
        labs2c, mus2c, sigs2c,
        title="Eixo 2 — Focal Loss γ: Erro de Contagem Médio (μ ± σ, 2 seeds)",
        ylabel="Erro Absoluto Médio de Contagem",
        save_path=output_path / "ablation_eixo2_count.png",
        colors=colors2,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tabela de resumo
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(results: dict) -> None:
    """Imprime a tabela final de ablações no terminal."""
    header = f"{'Config':<30} {'mAP μ':>8} {'mAP σ':>8} {'CE μ':>8} {'IoU μ':>8} {'Dice μ':>8}"
    print("\n" + "=" * 72)
    print(header)
    print("-" * 72)

    current_axis = None
    for eixo, label, slug, *_ in ABLATIONS:
        if eixo != current_axis:
            current_axis = eixo
            print(f"\n  {'Eixo 1 — Recuperação de Resolução' if eixo == 'eixo1' else 'Eixo 2 — Função de Perda (Focal Loss γ)':^70}")
            print("-" * 72)

        seed_data = results[slug]
        mAPs = [seed_data[f"seed{s}"]["mAP"] for s in SEEDS]
        CEs  = [seed_data[f"seed{s}"]["count_error"] for s in SEEDS]
        IoUs = [seed_data[f"seed{s}"]["iou"] for s in SEEDS]
        Dices = [seed_data[f"seed{s}"]["dice"] for s in SEEDS]

        print(
            f"  {label:<28} "
            f"{np.mean(mAPs):>8.3f} "
            f"{np.std(mAPs):>8.3f} "
            f"{np.mean(CEs):>8.2f} "
            f"{np.mean(IoUs):>8.3f} "
            f"{np.mean(Dices):>8.3f}"
        )
    print("=" * 72)
    print("  CE = Erro Absoluto Médio de Contagem | μ = média | σ = desvio padrão")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PA1 — Ablações Sistemáticas (Parte 3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--epochs",     type=int,   default=20,  help="Épocas por run de ablação")
    p.add_argument("--batch-size", type=int,   default=8,   help="Batch size")
    p.add_argument("--lr",         type=float, default=1e-3, help="Learning rate")
    p.add_argument("--config",     type=str,   default=None, help="Caminho do config.yaml (auto-detectado se omitido)")
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help="Rodar só uma config específica (ex: eixo1_sem_skips). Útil para retomar runs interrompidas.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Localiza o config.yaml base ───────────────────────────────────────────
    if args.config:
        cfg_path = Path(args.config)
    elif Path("pa1/config.yaml").exists():
        cfg_path = Path("pa1/config.yaml")
    elif Path("config.yaml").exists():
        cfg_path = Path("config.yaml")
    else:
        raise FileNotFoundError("config.yaml não encontrado. Rode na raiz do repositório.")

    # ── Carrega parte2 como base e aplica overrides CLI ───────────────────────
    base_cfg = load_config(cfg_path, parte="2")
    base_cfg.train.epochs = args.epochs
    base_cfg.data.batch_size = args.batch_size
    base_cfg.train.lr = args.lr

    device = get_device()

    # ── Cria pasta de saída ───────────────────────────────────────────────────
    output_path = Path(base_cfg.output_dir) / "ablation"
    output_path.mkdir(parents=True, exist_ok=True)

    n_total = len(ABLATIONS) * len(SEEDS)
    run_idx = 0
    results: dict = {}

    print(f"\n{'='*72}")
    print(f"  PA1 — Ablações Sistemáticas (Parte 3)")
    print(f"  {len(ABLATIONS)} configs × {len(SEEDS)} seeds = {n_total} runs")
    print(f"  {args.epochs} épocas/run | batch={args.batch_size} | lr={args.lr}")
    print(f"  Device: {device}")
    print(f"  Saídas: {output_path}")
    print(f"{'='*72}\n")

    for eixo, label, slug, use_skips, gamma in ABLATIONS:

        # Suporte a retomada parcial: --only eixo1_sem_skips
        if args.only and slug != args.only:
            continue

        results[slug] = {}

        for seed in SEEDS:
            run_idx += 1
            run_name = f"{slug}_seed{seed}"
            print(f"[{run_idx}/{n_total}] {label}  |  seed={seed}  |  use_skips={use_skips}  γ={gamma}")
            print(f"  ({slug})")

            t0 = time.time()
            metrics = run_single(
                use_skips=use_skips,
                gamma=gamma,
                seed=seed,
                base_cfg=base_cfg,
                device=device,
                output_path=output_path,
                run_name=slug,
                print_every=max(1, args.epochs // 4),
            )
            elapsed = time.time() - t0
            results[slug][f"seed{seed}"] = metrics

            print(
                f"  ✓ Concluído em {elapsed:.0f}s | "
                f"mAP={metrics['mAP']:.3f} | "
                f"Count err={metrics['count_error']:.2f} | "
                f"IoU={metrics['iou']:.3f}\n"
            )

    # ── Salva resultados brutos em JSON ───────────────────────────────────────
    results_json = output_path / "ablation_results.json"
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados brutos salvos em: {results_json}")

    # ── Salva CSV agregado ────────────────────────────────────────────────────
    import csv
    csv_path = output_path / "ablation_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["axis", "config", "label", "seed", "mAP", "count_error", "iou", "dice"])
        for eixo, label, slug, *_ in ABLATIONS:
            if slug not in results:
                continue
            for seed in SEEDS:
                row_key = f"seed{seed}"
                if row_key not in results[slug]:
                    continue
                m = results[slug][row_key]
                writer.writerow([eixo, slug, label, seed, m["mAP"], m["count_error"], m["iou"], m["dice"]])
    print(f"CSV de ablação salvo em: {csv_path}")

    # ── Gráficos ──────────────────────────────────────────────────────────────
    print("\nGerando gráficos de ablação...")
    try:
        plot_ablation_results(results, output_path)
    except Exception as e:
        print(f"  Aviso: falha ao gerar gráficos ({e}). Os JSONs/CSV foram salvos.")

    # ── Tabela final no terminal ───────────────────────────────────────────────
    print_summary_table(results)

    print(f"\n✅ Ablações concluídas. Artefatos em: {output_path}\n")


if __name__ == "__main__":
    main()
