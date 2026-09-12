"""
Parte 4: Mosaico e Janelas Deslizantes (Tiling).

Este script demonstra como aplicar a U-Net em imagens gigantes que não cabem
na memória da GPU de uma só vez, utilizando janelas deslizantes (sliding windows).

Problema evidenciado:
Quando aplicamos o watershed (ou conectamos componentes) tile por tile,
células que ficam na linha de corte são partidas ao meio, gerando
falsos positivos e inflando o erro de contagem.

Solução implementada:
Algoritmo de fusão de instâncias na borda. Analisamos a faixa de sobreposição
entre os tiles. Se uma instância do tile atual tiver forte sobreposição (IoU)
com uma instância do tile vizinho, elas são unificadas sob o mesmo ID.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from matplotlib.patches import Rectangle

from pa1.config import load_config
from pa1.data import make_dsb2018_loaders
from pa1.metrics import compute_map
from pa1.models import UNet
from pa1.postprocessing import watershed_to_instances
from pa1.utils import get_device, set_seed

# ─────────────────────────────────────────────────────────────────────────────
# 1. Criação do Mosaico (Imagem Gigante Sintética)
# ─────────────────────────────────────────────────────────────────────────────

def build_mosaic(dataset, indices: list[int]) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Costura 4 imagens do dataset em um mosaico 2x2 de 512x512."""
    assert len(indices) == 4, "Necessário exatamente 4 índices para um mosaico 2x2"
    
    # Cada imagem no DSB2018 após transformações tem 256x256
    H, W = 256, 256
    mosaic_img = torch.zeros((3, H * 2, W * 2), dtype=torch.float32)
    mosaic_gt = np.zeros((H * 2, W * 2), dtype=np.int32)
    
    current_max_id = 0
    
    positions = [(0, 0), (0, W), (H, 0), (H, W)]
    
    for i, pos in zip(indices, positions):
        y, x = pos
        item = dataset[i]
        img = item["image"]
        inst = item["mask_instances"].numpy()
        
        mosaic_img[:, y:y+H, x:x+W] = img
        
        # Deslocar os IDs para não haver colisão entre imagens distintas
        mask_fg = inst > 0
        inst_shifted = np.where(mask_fg, inst + current_max_id, 0)
        mosaic_gt[y:y+H, x:x+W] = inst_shifted
        
        if mask_fg.any():
            current_max_id = inst_shifted.max()
            
    # Criar uma imagem RGB para plotagem (desnormalizando)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb_vis = (mosaic_img * std + mean).permute(1, 2, 0).numpy()
    rgb_vis = np.clip(rgb_vis, 0, 1)
    
    return mosaic_img, mosaic_gt, rgb_vis


# ─────────────────────────────────────────────────────────────────────────────
# 2. Inferência com Janelas Deslizantes e Fusão
# ─────────────────────────────────────────────────────────────────────────────

def extract_patches_coords(img_h: int, img_w: int, patch_size: int, stride: int) -> list[tuple[int, int, int, int]]:
    """Gera as coordenadas (y1, y2, x1, x2) para as janelas deslizantes."""
    y_starts = list(range(0, img_h - patch_size + 1, stride))
    if y_starts[-1] + patch_size < img_h:
        y_starts.append(img_h - patch_size)
        
    x_starts = list(range(0, img_w - patch_size + 1, stride))
    if x_starts[-1] + patch_size < img_w:
        x_starts.append(img_w - patch_size)
        
    coords = []
    for y in y_starts:
        for x in x_starts:
            coords.append((y, y + patch_size, x, x + patch_size))
            
    return list(dict.fromkeys(coords))


def resolve_equivalence_graph(edges: list[tuple[int, int]]) -> dict[int, int]:
    """
    Resolve componentes conexos de equivalência usando scipy.
    Retorna dicionário mapeando cada ID para seu ID raiz (o menor ID do grupo).
    """
    import scipy.sparse as sparse
    from scipy.sparse.csgraph import connected_components

    if not edges:
        return {}
        
    # Maior ID no grafo
    max_node = max(max(u, v) for u, v in edges)
    
    # Constroi matriz adjacente
    row = np.array([u for u, v in edges] + [v for u, v in edges])
    col = np.array([v for u, v in edges] + [u for u, v in edges])
    data = np.ones(len(row), dtype=bool)
    
    adj = sparse.coo_matrix((data, (row, col)), shape=(max_node + 1, max_node + 1))
    
    n_components, labels = connected_components(csgraph=adj, directed=False, return_labels=True)
    
    mapping = {}
    for comp_id in range(n_components):
        # Todos os nós neste componente
        nodes = np.where(labels == comp_id)[0]
        if len(nodes) > 1:
            root = nodes[0]  # menor nó
            for node in nodes:
                mapping[node] = root
                
    return mapping


def sliding_window_inference(
    model: torch.nn.Module,
    mosaic_img: torch.Tensor,
    device: torch.device,
    patch_size: int = 256,
    overlap: int = 64,
    apply_fusion: bool = False,
    merge_iou_threshold: float = 0.2,
) -> np.ndarray:
    """
    Realiza inferência usando janelas deslizantes e constrói o mapa de instâncias final.
    Se apply_fusion=True, resolve as instâncias cortadas nas bordas.
    """
    model.eval()
    _, img_h, img_w = mosaic_img.shape
    stride = patch_size - overlap
    
    coords = extract_patches_coords(img_h, img_w, patch_size, stride)
    
    global_instances = np.zeros((img_h, img_w), dtype=np.int32)
    current_max_id = 0
    
    # Lista de arestas (id1, id2) que representam a mesma célula na borda
    equivalences = []
    
    with torch.no_grad():
        for (y1, y2, x1, x2) in coords:
            patch = mosaic_img[:, y1:y2, x1:x2].unsqueeze(0).to(device)
            logits = model(patch)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            
            # Watershed no tile
            tile_inst = watershed_to_instances(probs, interior_channel=1, marker_threshold=0.6, min_area=15)
            
            if tile_inst.max() == 0:
                continue
                
            # Desloca os IDs deste tile para serem únicos globalmente
            mask_fg = tile_inst > 0
            tile_inst_shifted = np.where(mask_fg, tile_inst + current_max_id, 0)
            
            # Atualiza o contador
            current_max_id += tile_inst.max()
            
            # Extrair região correspondente no canvas global
            global_region = global_instances[y1:y2, x1:x2]
            
            # Análise de sobreposição para fundir bordas
            if apply_fusion:
                # Onde global já tinha algo E tile tem algo novo
                intersection_mask = (global_region > 0) & (tile_inst_shifted > 0)
                
                if intersection_mask.any():
                    overlap_global_ids = global_region[intersection_mask]
                    overlap_tile_ids = tile_inst_shifted[intersection_mask]
                    
                    # Pares únicos de interseção e contagem
                    pairs = np.column_stack((overlap_global_ids, overlap_tile_ids))
                    unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
                    
                    for (g_id, t_id), count in zip(unique_pairs, counts):
                        area_g = (global_region == g_id).sum()
                        area_t = (tile_inst_shifted == t_id).sum()
                        
                        iou = count / (area_g + area_t - count)
                        
                        if iou > merge_iou_threshold:
                            equivalences.append((g_id, t_id))
            
            # Coloca o patch no canvas global (o mais recente sobrescreve na borda)
            global_instances[y1:y2, x1:x2] = np.where(tile_inst_shifted > 0, tile_inst_shifted, global_region)

    # ── Resolve as fusões de borda ─────────────────────────────────────────────
    if apply_fusion and equivalences:
        mapping = resolve_equivalence_graph(equivalences)
        
        max_possible_id = global_instances.max()
        map_array = np.arange(max_possible_id + 1, dtype=np.int32)
        
        for node, root in mapping.items():
            if node <= max_possible_id:
                map_array[node] = root
                
        global_instances = map_array[global_instances]

    return global_instances


# ─────────────────────────────────────────────────────────────────────────────
# Visualização
# ─────────────────────────────────────────────────────────────────────────────

def plot_mosaic_results(
    rgb_img: np.ndarray,
    gt_inst: np.ndarray,
    pred_naive: np.ndarray,
    pred_fusion: np.ndarray,
    coords: list[tuple],
    save_path: Path,
):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()
    
    # Configura cmap discreto (fundo preto)
    np.random.seed(42)
    colors = np.random.rand(10000, 3)
    colors[0] = [0, 0, 0] 
    cmap = mcolors.ListedColormap(colors)
    
    def format_ax(ax, title, img, is_inst=False):
        if is_inst:
            ax.imshow(img, cmap=cmap, interpolation="nearest")
        else:
            ax.imshow(img)
            
        ax.set_title(title, fontsize=12, pad=10)
        ax.axis("off")
        
        # Desenha grid / overlap (apenas visual)
        for (y1, y2, x1, x2) in coords:
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.2, edgecolor="yellow", facecolor="none", linestyle="--", alpha=0.6)
            ax.add_patch(rect)

    format_ax(axs[0], "1. Mosaico 512x512 (Janelas Deslizantes 256x256)", rgb_img)
    format_ax(axs[1], f"2. Ground Truth (Total: {len(np.unique(gt_inst))-1})", gt_inst, is_inst=True)
    format_ax(axs[2], f"3. Sem Fusão de Borda (Falsos Positivos: {len(np.unique(pred_naive))-1})", pred_naive, is_inst=True)
    format_ax(axs[3], f"4. Com Fusão de Borda (Corrigido: {len(np.unique(pred_fusion))-1})", pred_fusion, is_inst=True)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Mosaic] Gráfico salvo em: {save_path}")


def plot_map_comparison(naive_map: float, fusion_map: float, save_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    categories = ['Sem Fusão', 'Com Fusão']
    values = [naive_map, fusion_map]
    bars = ax.bar(categories, values, color=['#d9534f', '#5cb85c'], edgecolor='black')
    
    ax.set_ylabel('mAP (Hungarian)', fontsize=12)
    ax.set_title('Impacto da Fusão de Bordas no mAP', fontsize=14, pad=15)
    max_val = max(values)
    if max_val > 0:
        ax.set_ylim(0, max_val * 1.2)
    else:
        ax.set_ylim(0, 1.0)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
                    
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Mosaic] Gráfico de mAP salvo em: {save_path}")


def plot_zoom_comparison(rgb_img: np.ndarray, pred_naive: np.ndarray, pred_fusion: np.ndarray, save_path: Path):
    # Recorte na interseção central: y de 200 a 312, x de 200 a 312 (112x112)
    y1, y2 = 200, 312
    x1, x2 = 200, 312
    
    zoom_img = rgb_img[y1:y2, x1:x2]
    zoom_naive = pred_naive[y1:y2, x1:x2]
    zoom_fusion = pred_fusion[y1:y2, x1:x2]
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    np.random.seed(42)
    colors = np.random.rand(10000, 3)
    colors[0] = [0, 0, 0] 
    cmap = mcolors.ListedColormap(colors)
    
    axs[0].imshow(zoom_img)
    axs[0].set_title("1. Recorte (Zoom na fronteira)", fontsize=14)
    
    axs[1].imshow(zoom_naive, cmap=cmap, interpolation="nearest")
    axs[1].set_title("2. Sem Fusão (Células cortadas)", fontsize=14)
    
    axs[2].imshow(zoom_fusion, cmap=cmap, interpolation="nearest")
    axs[2].set_title("3. Com Fusão (Células consertadas)", fontsize=14)
    
    for ax in axs:
        ax.axis('off')
        # Desenha as linhas da fronteira (em x=256 e y=256 original -> transladado para x=56, y=56 no zoom)
        ax.axhline(y=256 - y1, color='yellow', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(x=256 - x1, color='yellow', linestyle='--', linewidth=2, alpha=0.7)
        
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Mosaic] Gráfico de Zoom salvo em: {save_path}")


def main():
    print("="*70)
    print(" Parte 4: Mosaico e Tiling (Resolução de Instâncias na Borda)")
    print("="*70)
    
    cfg_path = Path("pa1/config.yaml")
    if not cfg_path.exists():
        cfg_path = Path("config.yaml")
        
    cfg = load_config(cfg_path, parte="2")
    device = get_device()
    
    ckpt_path = Path(cfg.output_dir) / "checkpoints" / "parte2_baseline_unet.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}. Rode 'uv run pa1 2' primeiro.")
        
    model = UNet(in_channels=3, out_channels=3).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    
    print(f"\n[1] Modelo carregado: {ckpt_path.name}")
    
    _, _, test_loader = make_dsb2018_loaders(
        data_dir=cfg.data.data_dir,
        batch_size=1,
        num_workers=0,
        seed=42,
    )
    
    # 4 imagens do teste
    indices = [2, 5, 8, 11]  
    mosaic_tensor, mosaic_gt, rgb_vis = build_mosaic(test_loader.dataset, indices)
    print("[2] Mosaico sintético criado (512x512) com 4 imagens.")
    
    print("\n[3] Inferência com Janelas Deslizantes (Overlap: 64px)...")
    t0 = time.time()
    pred_naive = sliding_window_inference(model, mosaic_tensor, device, patch_size=256, overlap=64, apply_fusion=False)
    print(f"    -> Sem Fusão concluída em {time.time()-t0:.2f}s")
    
    t0 = time.time()
    pred_fusion = sliding_window_inference(model, mosaic_tensor, device, patch_size=256, overlap=64, apply_fusion=True)
    print(f"    -> Com Fusão concluída em {time.time()-t0:.2f}s")
    
    print("\n[4] Avaliação:")
    naive_metrics = compute_map(pred_naive, mosaic_gt, method="hungarian")
    fusion_metrics = compute_map(pred_fusion, mosaic_gt, method="hungarian")
    
    n_gt = len(np.unique(mosaic_gt)) - 1
    n_naive = len(np.unique(pred_naive)) - 1
    n_fusion = len(np.unique(pred_fusion)) - 1
    
    print(f"  Ground Truth   : {n_gt} células")
    print(f"  Sem Fusão      : Encontradas {n_naive} (Erro: {abs(n_naive - n_gt)}) | mAP: {naive_metrics['mAP']:.3f}")
    print(f"  Com Fusão      : Encontradas {n_fusion} (Erro: {abs(n_fusion - n_gt)}) | mAP: {fusion_metrics['mAP']:.3f}")
    
    out_dir = Path(cfg.output_dir)
    coords = extract_patches_coords(512, 512, 256, 192)
    
    # 1. Gráfico Principal (Mosaico Completo)
    save_fig = out_dir / "parte4_mosaic_fusion.png"
    plot_mosaic_results(rgb_vis, mosaic_gt, pred_naive, pred_fusion, coords, save_path=save_fig)
    
    # 2. Gráfico do mAP
    save_map = out_dir / "parte4_map_comparison.png"
    plot_map_comparison(naive_metrics['mAP'], fusion_metrics['mAP'], save_path=save_map)
    
    # 3. Gráfico do Zoom
    save_zoom = out_dir / "parte4_zoom_fusion.png"
    plot_zoom_comparison(rgb_vis, pred_naive, pred_fusion, save_path=save_zoom)
    
    print("\n✅ Parte 4 finalizada com sucesso.")


if __name__ == "__main__":
    main()
