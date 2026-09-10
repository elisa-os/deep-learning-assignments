"""Parte 6: Teste de Estresse (Sensibilidade a Corrupções Sintéticas).

Avalia o modelo congelado (da Parte 2) sob perturbações de blur, ruído e contraste,
em 3 níveis de severidade. 
Gera o gráfico de degradação do mAP@[0.5:0.95].
"""

import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF

from pa1.config import load_config
from pa1.data import make_dsb2018_loaders
from pa1.metrics import compute_map
from pa1.models import UNet
from pa1.postprocessing import watershed_to_instances
from pa1.utils import get_device

# ─────────────────────────────────────────────────────────────────────────────
# Funções de Corrupção (atuam sobre tensores [C, H, W] normalizados)
# ─────────────────────────────────────────────────────────────────────────────

def apply_gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma == 0:
        return img
    # kernel size deve ser ímpar e suficientemente grande
    kernel_size = int(4 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return TF.gaussian_blur(img, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

def apply_gaussian_noise(img: torch.Tensor, std: float) -> torch.Tensor:
    if std == 0:
        return img
    noise = torch.randn_like(img) * std
    return img + noise

def apply_contrast(img: torch.Tensor, alpha: float) -> torch.Tensor:
    """Aplica redução de contraste. alpha=1.0 (original), alpha < 1.0 (menos contraste)."""
    if alpha == 1.0:
        return img
    # Como a imagem está normalizada (ImageNet), podemos apenas escalar em direção à média 0
    return img * alpha


# ─────────────────────────────────────────────────────────────────────────────
# Avaliação de Estresse
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_stress(model, loader, device, corruption_fn, param_val):
    """Avalia o modelo em todo o conjunto de dados sob uma corrupção específica."""
    model.eval()
    maps = []
    
    with torch.no_grad():
        for batch in loader:
            img = batch["image"][0] # batch_size=1
            img_corr = corruption_fn(img, param_val).unsqueeze(0).to(device)
            gt_inst = batch["mask_instances"][0].numpy()
            
            logits = model(img_corr)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_inst = watershed_to_instances(probs, interior_channel=1, marker_threshold=0.6, min_area=15)
            
            metrics = compute_map(pred_inst, gt_inst, method="hungarian")
            maps.append(metrics["mAP"])
            
    return np.mean(maps)

def main():
    print("="*70)
    print(" Parte 6: Teste de Estresse por Corrupções Sintéticas")
    print("="*70)
    
    cfg = load_config("pa1/config.yaml", parte="2")
    device = get_device()
    
    ckpt_path = Path(cfg.output_dir) / "checkpoints" / "parte2_baseline_unet.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}. Rode a Parte 2 primeiro.")
        
    model = UNet(in_channels=3, out_channels=3).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    
    print(f"\\n[1] Modelo carregado: {ckpt_path.name}")
    
    # Loader de teste com batch_size=1
    _, _, test_loader = make_dsb2018_loaders(
        data_dir=cfg.data.data_dir,
        batch_size=1,
        num_workers=0,
        seed=42,
    )
    
    # ── Configuração das Perturbações (conforme Plano de Execução) ──
    corruptions = {
        "Gaussian Blur (sigma)": {
            "func": apply_gaussian_blur,
            "levels": [0.0, 1.0, 2.0, 3.5], # 0.0 é o baseline original
        },
        "Gaussian Noise (std)": {
            "func": apply_gaussian_noise,
            "levels": [0.0, 0.05, 0.15, 0.30],
        },
        "Contrast (alpha)": {
            "func": apply_contrast,
            "levels": [1.0, 0.7, 0.4, 0.2], # 1.0 é o baseline original
        }
    }
    
    results = {}
    print("\\n[2] Iniciando Testes de Estresse no Test Set...")
    t0 = time.time()
    
    for name, config in corruptions.items():
        print(f"\\n--- Testando: {name} ---")
        levels = config["levels"]
        func = config["func"]
        mAPs = []
        
        for val in levels:
            print(f"  Avaliando intensidade {val:.2f}...", end="", flush=True)
            map_val = evaluate_stress(model, test_loader, device, func, val)
            mAPs.append(map_val)
            print(f" mAP = {map_val:.3f}")
            
        results[name] = {"levels": levels, "mAPs": mAPs}
        
    print(f"\\nTestes finalizados em {time.time()-t0:.1f}s")
    
    # ── Geração do Gráfico de Degradação ──
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, (name, data) in zip(axs, results.items()):
        levels = data["levels"]
        maps = data["mAPs"]
        
        # O eixo X varia conforme o parâmetro
        ax.plot(levels, maps, marker='o', linewidth=2, color='darkorange')
        ax.set_title(name)
        ax.set_xlabel("Intensidade")
        ax.set_ylabel("mAP@[0.5:0.95]")
        ax.set_ylim(0, max(maps) * 1.1)
        ax.grid(True, linestyle="--", alpha=0.6)
        
        # Sublinhar o baseline (intensidade 0 ou 1)
        ax.axhline(y=maps[0], color='gray', linestyle='--', label='Baseline', alpha=0.5)
        ax.legend()
        
    plt.tight_layout()
    out_dir = Path(cfg.output_dir)
    save_fig = out_dir / "parte6_stress_test.png"
    fig.savefig(save_fig, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\\n[3] Gráfico de degradação salvo em: {save_fig}")
    print("✅ Parte 6 finalizada com sucesso.")

if __name__ == "__main__":
    main()
