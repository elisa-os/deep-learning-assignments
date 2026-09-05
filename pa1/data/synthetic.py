"""Dataset sintético de elipses para testes unitários (Parte 0).

Por que dados sintéticos em Deep Learning?
-------------------------------------------
Antes de tocar em dados reais (ruidosos, desbalanceados, mal-anotados),
treinar em dados sintéticos é como um "teste unitário" da sua arquitetura:

1. Você SABE a resposta correta (ground truth perfeito).
2. Você controla a dificuldade (sobreposição, ruído, contraste).
3. Se o modelo não aprende em dados sintéticos simples, há um bug
   na arquitetura, na loss ou no pipeline de dados — não nos dados.

O Dataset aqui gera imagens 128×128 com elipses aleatórias, simulando
o caso de segmentação de instâncias de objetos circulares (células, núcleos, etc.).

Estrutura de um Dataset PyTorch:
---------------------------------
PyTorch usa o protocolo Dataset + DataLoader:
- Dataset: define __len__() e __getitem__(idx) — "como acessar o dado i"
- DataLoader: embaralha, divide em batches, carrega em paralelo (num_workers)

Um "batch" é um conjunto de N amostras processadas juntas pela GPU.
Usar batches > 1 é mais eficiente porque a GPU processa matrizes em paralelo.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.draw import ellipse as sk_ellipse


class SyntheticEllipseDataset(Dataset):
    """Gera imagens 128×128 com 5–20 elipses sobrepostas.

    Cada imagem tem:
    - image: tensor float32 (1, H, W), valores em [0, 1]
    - mask_semantic: tensor long (H, W), 0=fundo, 1=objeto
    - mask_instances: tensor long (H, W), 0=fundo, N=ID da instância N

    Args:
        n_samples: Quantas imagens gerar por "epoch".
        img_size: Tamanho do lado da imagem quadrada.
        min_ellipses: Mínimo de elipses por imagem.
        max_ellipses: Máximo de elipses por imagem.
        seed: Seed para reprodutibilidade (None = aleatório).
    """

    def __init__(
        self,
        n_samples: int = 500,
        img_size: int = 128,
        min_ellipses: int = 5,
        max_ellipses: int = 20,
        seed: int | None = 42,
    ) -> None:
        self.n_samples = n_samples
        self.img_size = img_size
        self.min_ellipses = min_ellipses
        self.max_ellipses = max_ellipses
        self.seed = seed

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        # Seed por amostra: garante que o mesmo idx sempre retorna o mesmo dado
        rng = np.random.default_rng(self.seed + idx if self.seed is not None else None)

        H, W = self.img_size, self.img_size
        image = np.zeros((H, W), dtype=np.float32)
        instance_mask = np.zeros((H, W), dtype=np.int64)

        n_ellipses = int(rng.integers(self.min_ellipses, self.max_ellipses + 1))

        for i in range(1, n_ellipses + 1):
            cx = int(rng.integers(10, W - 10))
            cy = int(rng.integers(10, H - 10))
            rx = int(rng.integers(5, 25))
            ry = int(rng.integers(5, 25))
            rotation = float(rng.uniform(0, np.pi))
            brightness = float(rng.uniform(0.4, 1.0))

            # skimage.draw.ellipse retorna (row_indices, col_indices) dentro da imagem
            rr, cc = sk_ellipse(cy, cx, ry, rx, shape=(H, W), rotation=rotation)
            image[rr, cc] = brightness
            instance_mask[rr, cc] = i

        # Ruído gaussiano
        noise_std = float(rng.uniform(0.02, 0.15))
        image = np.clip(image + rng.normal(0, noise_std, (H, W)).astype(np.float32), 0, 1)

        semantic_mask = (instance_mask > 0).astype(np.int64)

        return {
            "image": torch.from_numpy(image).unsqueeze(0),          # (1, H, W)
            "mask_semantic": torch.from_numpy(semantic_mask),        # (H, W)
            "mask_instances": torch.from_numpy(instance_mask),       # (H, W)
        }


def make_synthetic_loader(
    n_samples: int = 500,
    batch_size: int = 8,
    num_workers: int = 2,
    seed: int = 42,
    split: str = "train",
) -> DataLoader:
    """Cria DataLoader do dataset sintético com split train/val/test.

    Args:
        n_samples: Total de amostras.
        batch_size: Amostras por batch.
        num_workers: Threads paralelas de carregamento.
        seed: Seed para reprodutibilidade.
        split: "train" | "val" | "test"
    """
    splits = {"train": 0, "val": n_samples, "test": n_samples + n_samples // 5}
    offset = splits.get(split, 0)
    size = n_samples if split == "train" else n_samples // 5

    dataset = SyntheticEllipseDataset(
        n_samples=size,
        seed=seed + offset,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
