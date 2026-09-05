"""Pós-processamento: máscara semântica → instâncias (Parte 1, etapa 2).

Método ingênuo: Limiar + Componentes Conexos
---------------------------------------------
Uma vez que o modelo prevê uma máscara SEMÂNTICA binária (fundo vs. objeto),
precisamos separar os objetos individuais. A abordagem mais simples:

1. Aplica limiar na probabilidade de saída → máscara binária
2. Encontra componentes conexos (regiões conectadas de pixels 1)
3. Cada componente vira uma instância

Limitações:
- Objetos que se tocam formam um único componente → um único "falso" objeto
- Esse é exatamente o problema que a Parte 2 resolve!

Componentes Conexos:
---------------------
Em teoria dos grafos, um componente conexo é um subconjunto de vértices onde
há caminho entre qualquer par. Em imagens binárias:
- Vértices = pixels com valor 1
- Arestas = vizinhança (4-conectado ou 8-conectado)

scipy.ndimage.label implementa isso eficientemente com complexidade O(H×W).
"""

import numpy as np
from scipy.ndimage import label as scipy_label


def semantic_to_instances(
    semantic_prob: np.ndarray,
    threshold: float = 0.5,
    connectivity: int = 2,
    min_area: int = 10,
) -> np.ndarray:
    """Converte mapa de probabilidade semântica em máscara de instâncias.

    Args:
        semantic_prob: Array (H, W) de floats em [0, 1] — probabilidade de "objeto".
        threshold: Limiar de binarização.
        connectivity: 1 = 4-conectado (cruzes), 2 = 8-conectado (diagonal).
        min_area: Remove componentes com menos de min_area pixels (ruído).

    Returns:
        instance_mask: Array (H, W) de inteiros; 0=fundo, N=ID da instância N.
    """
    binary = (semantic_prob >= threshold).astype(np.uint8)

    # Estrutura de conectividade (np.ones((3,3)) = 8-conectado)
    struct = np.ones((3, 3)) if connectivity == 2 else None
    labeled, n_components = scipy_label(binary, structure=struct)

    # Remove componentes pequenos (provavelmente ruído)
    instance_mask = np.zeros_like(labeled)
    new_id = 1
    for comp_id in range(1, n_components + 1):
        comp = labeled == comp_id
        if comp.sum() >= min_area:
            instance_mask[comp] = new_id
            new_id += 1

    return instance_mask
