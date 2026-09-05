"""Funções de perda para segmentação (Parte 1 e Parte 2).

O que é uma função de perda (loss)?
--------------------------------------
A loss mede o quão errada está a predição do modelo em relação ao ground truth.
O treinamento consiste em MINIMIZAR essa loss via gradiente descendente:

    θ ← θ - lr × ∇_θ L(y_pred, y_true)

onde θ são os pesos da rede, lr é o learning rate e ∇_θ é o gradiente.

Por que não usar só Cross-Entropy (CE)?
-----------------------------------------
CE funciona bem quando as classes são balanceadas. Em segmentação de
instâncias, o fundo (classe 0) domina: pode ter 95% dos pixels como fundo
e apenas 5% como objeto. A CE otimizada por SGD pode simplesmente aprender
a prever "tudo é fundo" e obter 95% de acurácia... mas ser inútil.

Soluções:
- **Dice Loss**: mede sobreposição entre predição e GT, insensível ao
  desbalanceamento (veja abaixo).
- **Focal Loss**: penaliza mais os erros nos exemplos DIFÍCEIS. Reduz o
  peso dos exemplos fáceis (fundo trivial), forçando o modelo a focar
  nas fronteiras e objetos minoritários.

Dice Loss:
-----------
Dice = 2 * |A ∩ B| / (|A| + |B|)

Onde A é o conjunto de pixels preditos como objeto e B o ground truth.
Dice = 1 → predição perfeita. Dice = 0 → sem sobreposição.
DiceLoss = 1 - Dice, logo minimizar DiceLoss maximiza a sobreposição.

Focal Loss (Lin et al., 2017):
--------------------------------
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

O fator (1 - p_t)^γ reduz a contribuição de exemplos fáceis (onde p_t ≈ 1).
- γ = 0 → equivale à CE padrão
- γ = 2 → configuração padrão do artigo original (RetinaNet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss para segmentação binária.

    Args:
        smooth: Constante de suavização para evitar divisão por zero.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Saída bruta da rede, shape (B, C, H, W).
            targets: Máscara inteira (B, H, W) com valores em {0, ..., C-1}.
        """
        probs = F.softmax(logits, dim=1)[:, 1]  # probabilidade da classe 1
        targets_f = targets.float()

        intersection = (probs * targets_f).sum(dim=(-2, -1))
        union = probs.sum(dim=(-2, -1)) + targets_f.sum(dim=(-2, -1))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss para segmentação (lida bem com desbalanceamento).

    Args:
        gamma: Expoente de focalização. 0 = CE padrão, 2 = padrão do artigo.
        alpha: Peso para a classe positiva (objeto). None = sem ponderação.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce)  # probabilidade da classe correta
        focal = ((1 - p_t) ** self.gamma) * ce

        if self.alpha is not None:
            # Peso alpha para classe 1, (1-alpha) para classe 0
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal = alpha_t * focal

        return focal.mean()


class BCEDiceLoss(nn.Module):
    """Combinação de Cross-Entropy + Dice Loss (prática comum).

    Args:
        bce_weight: Peso relativo da CE (1 - bce_weight vai para o Dice).
    """

    def __init__(self, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_w = bce_weight
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets)
        dice = self.dice(logits, targets)
        return self.bce_w * ce + (1 - self.bce_w) * dice
