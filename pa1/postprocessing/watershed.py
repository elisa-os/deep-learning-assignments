"""Decodificação de instâncias via Watershed com marcadores (Passo 3).

Entrada: mapa de probabilidades das 3 classes (após softmax).
Saída:   máscara de instâncias (0=fundo, 1..N=instâncias).

Algoritmo:
  1. Extrair mapa de probabilidade da classe Interior.
  2. Gerar marcadores (sementes) a partir do mapa de interior com
     limiar alto e componentes conexos.
  3. Aplicar watershed sobre o negativo da probabilidade de interior,
     usando os marcadores como sementes.
  4. Filtrar componentes espúrios por área mínima.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed


def watershed_to_instances(
    prob_3class: np.ndarray,
    *,
    interior_channel: int = 1,
    marker_threshold: float = 0.6,
    min_area: int = 15,
    prob_cut: float = 0.3,
    distance_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Decodifica predição 3-classes em máscara de instâncias via watershed.

    Args:
        prob_3class: (H, W) ou (3, H, W) de probabilidades pós-softmax.
        interior_channel: índice do canal da classe Interior (0=fundo,1=interior,2=fronteira).
        marker_threshold: limiar de confiança para gerar marcadores (sementes).
        min_area: instâncias com área < min_area são descartadas.
        prob_cut: máscara de foreground para o watershed (prob_interior > prob_cut).
        distance_map: mapa de distância EDT opcional para bacia; se None usa -prob_interior.

    Returns:
        inst_mask: (H, W) int32, 0=fundo, 1..N=instâncias.
    """
    if prob_3class.ndim == 3:
        p_int = prob_3class[interior_channel]
    else:
        p_int = prob_3class

    # --- marcadores: componentes conexos dentro do limiar alto ---
    markers_mask = p_int > marker_threshold
    seeds, n_seeds = ndimage.label(markers_mask)
    
    if n_seeds < 1:
        return np.zeros(p_int.shape, dtype=np.int32)
        
    # Otimização crucial: filtrar sementes microscópicas ANTES do watershed.
    # Quando o modelo está destreinado, ele pode gerar milhares de sementes 
    # de 1 pixel. O algoritmo de watershed (que usa uma Priority Queue) sofre.
    if min_area > 0 and n_seeds > 0:
        seed_areas = np.bincount(seeds.ravel())
        # Sementes podem ser um pouco menores que a área final, então usamos 
        # um critério mais leniente (min_area // 3 ou no mínimo 2 pixels)
        tiny_seeds = np.where((seed_areas < max(2, min_area // 3)) & (seed_areas > 0))[0]
        if tiny_seeds.size > 0:
            seeds[np.isin(seeds, tiny_seeds)] = 0

    # --- bacia: negativo da probabilidade (ou distância negativa) ---
    if distance_map is not None:
        basin = -distance_map.astype(np.float64)
    else:
        # moldura de zeros evita fronteiras artificiais
        basin = -np.pad(p_int, 1, mode="constant", constant_values=1.0)[1:-1, 1:-1].astype(np.float64)

    fg = p_int > prob_cut
    inst = watershed(basin, seeds, mask=fg)

    # --- filtrar ruído por área ---
    if min_area > 0:
        # Vetorização: bincount conta os pixels de cada label instantaneamente.
        # inst vai de 0 até inst.max().
        areas = np.bincount(inst.ravel())
        
        # Filtra instâncias (ignorando o fundo 0) que são menores que min_area
        too_small = np.where((areas < min_area) & (areas > 0))[0]
        
        if too_small.size > 0:
            # Zera todas as instâncias muito pequenas de uma vez
            inst[np.isin(inst, too_small)] = 0
            
    return inst.astype(np.int32)


def distance_transform_fg(mask: np.ndarray) -> np.ndarray:
    """EDT sobre o foreground (1s). Útil como bacia alternativa."""
    from skimage.measure import label
    fg = (mask > 0).astype(np.uint8)
    lbl, n = label(fg, return_num=True)
    if n == 0:
        return np.zeros_like(mask, dtype=np.float32)
    dt = ndimage.distance_transform_edt(lbl > 0)
    return dt.astype(np.float32)
