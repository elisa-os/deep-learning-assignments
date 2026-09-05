"""Visualização de máscaras de instância."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def show_instances(
    image: np.ndarray,
    instance_mask: np.ndarray,
    title: str = "",
    ax: plt.Axes | None = None,
) -> None:
    """Plota imagem com instâncias coloridas por ID.

    Args:
        image: Array (H, W) ou (H, W, C) com a imagem original.
        instance_mask: Array (H, W) de inteiros; 0 = fundo, N > 0 = instância N.
        title: Título opcional do gráfico.
        ax: Eixo matplotlib opcional (cria um novo se None).
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Imagem de fundo em escala de cinza
    ax.imshow(image, cmap="gray" if image.ndim == 2 else None)

    # Sobrepõe cada instância com cor aleatória mas reprodutível
    n_instances = instance_mask.max()
    rng = np.random.default_rng(seed=0)
    colors = rng.random((n_instances + 1, 4))
    colors[:, 3] = 0.5   # alpha
    colors[0] = [0, 0, 0, 0]  # fundo transparente

    colored = colors[instance_mask]
    ax.imshow(colored)

    ax.set_title(f"{title} ({n_instances} instâncias)" if title else f"{n_instances} instâncias")
    ax.axis("off")
