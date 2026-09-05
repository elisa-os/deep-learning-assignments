"""Geração de targets de 3 classes para a Trilha A (fronteiras + watershed).

Converte máscaras de instâncias individuais em mapa de 3 classes:
  0: fundo
  1: interior (núcleo erodido)
  2: fronteira (borda dilatada entre núcleos)

Referência: slide 83 do curso — representação parasegregação de instâncias
via watershed com marcadores.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from scipy import ndimage
from skimage.morphology import disk
from skimage.morphology.binary import binary_erosion, binary_dilation


def build_3class_mask(
    instance_mask: np.ndarray,
    *,
    interior_radius: int = 1,
    boundary_width: int = 2,
    foreground_value: int = 1,
) -> np.ndarray:
    """Converte máscara de instâncias em mapa 3 classes (0=fundo,1=interior,2=fronteira).

    Args:
        instance_mask: (H, W) int64, instâncias rotuladas 1..N.
        interior_radius: raio em pixels para erosão do interior.
        boundary_width: largura em pixels da banda de fronteira.
        foreground_value: valor que denota foreground nas máscaras de entrada.

    Returns:
        mask_3class: (H, W) int8 com valores {0, 1, 2}.
    """
    sem = (instance_mask > 0).astype(np.uint8)

    # --- Interior: erosão de cada instância individualmente ---
    interior = np.zeros_like(instance_mask, dtype=np.uint8)
    inst_ids = np.unique(instance_mask[instance_mask > 0])
    for iid in inst_ids:
        single = (instance_mask == iid).astype(np.uint8)
        if interior_radius > 0:
            k = disk(interior_radius)
            er = binary_erosion(single, k).astype(np.uint8)
        else:
            er = single
        interior[er > 0] = 1

    # --- Fronteira: região entre o foreground dilatado e o interior ---
    if boundary_width > 0:
        k = disk(boundary_width)
        fg_dilated = binary_dilation(sem, k).astype(np.uint8)
    else:
        fg_dilated = sem
    boundary = (fg_dilated > 0) & (interior == 0)

    mask = np.zeros(instance_mask.shape, dtype=np.int8)
    mask[interior > 0] = 1
    mask[boundary > 0] = 2
    return mask


def make_3class_from_masks(
    instance_masks: list[np.ndarray],
    *,
    interior_radius: int = 1,
    boundary_width: int = 2,
) -> list[np.ndarray]:
    """Aplica `build_3class_mask` a uma lista de máscaras de instâncias."""
    return [
        build_3class_mask(m, interior_radius=interior_radius, boundary_width=boundary_width)
        for m in instance_masks
    ]
