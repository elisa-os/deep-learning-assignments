"""
Cabeças de saída para segmentação (PA1 — Trilha A).

Fornece cabeças alternativas que podem ser empilhadas sobre o encoder-decoder
de uma U-Net para produzir predições com o número desejado de classes.

Classes
-------
SegmentationHead
    Convolução 1×1 que mapeia o canal de features do decoder para C classes,
    com suporte opcional a ativação (softmax para multiclasse, sigmoid para binário).

U-Net com ``out_channels``
    A U-Net em ``unet.py`` já aceita ``out_channels`` como parâmetro de
    construção. Para 3 classes (0: fundo, 1: interior, 2: fronteira) basta
    instanciar ``UNet(out_channels=3)`` — a cabeça final ``self.head`` já é
    uma ``nn.Conv2d(features[0], out_channels, kernel_size=1)``.

    Esta módulo existe para centralizar a documentação das cabeças e futuras
    variações (cabeça com bottleneck adicional, heads auxiliares, etc.).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    """Cabeça de segmentação genérica: 1×1 conv + ativação opcional.

    Args:
        in_channels: número de canais de features de entrada (geralmente
            ``features[0]`` da U-Net, ex: 32).
        out_channels: número de classes de saída.
            - 2 → segmentação binária (fundo / objeto).
            - 3 → Trilha A: 0=fundo, 1=interior, 2=fronteira.
            - N → segmentação multiclasse arbitrária.
        activation: ``"softmax"`` (multiclasse), ``"sigmoid"`` (binário),
            ou ``None`` (logits crus — recomendado quando a loss aplica
            softmax/sigmoid internamente, ex: CrossEntropy, FocalLoss).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.activation == "softmax":
            return F.softmax(x, dim=1)
        if self.activation == "sigmoid":
            return torch.sigmoid(x)
        return x  # logits


class BoundaryAwareHead(nn.Module):
    """Cabeça alternativa para a Trilha A com 3 classes.

    Mantém a mesma interface de saída (``out_channels=3``) mas adiciona
    um pequeno bottleneck de 3×3 antes da convolução final, para dar à
    cabeça mais capacidade de modelar a transição suave entre interior e
    fronteira. Usar apenas se houver evidência de que a cabeça 1×1 simples
    subperforma em bordas.

    Args:
        in_channels: canais de entrada (features[0] da U-Net).
        inter_channels: canais intermediários do bottleneck (default: in_channels).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        inter_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        inter = inter_channels if inter_channels is not None else in_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, inter, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, out_channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
