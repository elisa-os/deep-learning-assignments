"""Extensões para Focal Loss multiclasse ponderada (Passo 5 - Trilha A).

Implementa Focal Loss com suporte a:
  - Múltiplas classes (C entradas no canal de saída)
  - Peso alpha_c por classe (sobremaneira para a classe fronteira)
  - Parâmetro gamma variável para ajuste de foco nos pixels "difíceis"

Fórmula:
  L_focal = -alpha_c * (1 - p_c)^gamma * log(p_c)

Onde p_c é a probabilidade softmax da classe correta.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss multiclasse com peso alpha_c por classe.

    Args:
        alpha: tensor (C,) com pesos por classe, ou float para usar
               o mesmo peso em todas as classes. Default: None (alpha=1 para todas).
        gamma: fator de focusing. gamma=0 recupera Cross-Entropy ponderada.
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        alpha: float | list[float] | torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

        if alpha is None:
            self._alpha = torch.tensor(1.0)
        elif isinstance(alpha, (float, int)):
            self._alpha = torch.tensor(float(alpha))
        elif isinstance(alpha, list):
            self._alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, torch.Tensor):
            self._alpha = alpha.float()
        else:
            raise TypeError(f"alpha type não suportado: {type(alpha)}")

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Calcula Focal Loss.

        Args:
            logits: (B, C, H, W) logits brutos (antes do softmax).
            targets: (B, H, W) com rótulos inteiros ou (B, C, H, W) one-hot.

        Returns:
            loss: escalar (ou tensor com reduction='none').
        """
        C = logits.shape[1]

        if targets.dim() == 3:
            targets_onehot = (
                F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float().to(logits.device)
            )
        elif targets.dim() == 4 and targets.shape[1] == C:
            targets_onehot = targets.float()
        else:
            raise ValueError(
                f"targets shape {tuple(targets.shape)} incompatível com logits {tuple(logits.shape)}"
            )

        probs = F.softmax(logits, dim=1)                                    # (B, C, H, W)
        probs_safe = torch.clamp(probs, min=self.eps, max=1 - self.eps)

        p_t = (probs_safe * targets_onehot).sum(dim=1)                     # (B, H, W)

        alpha = self._alpha
        if alpha.ndim == 0:
            alpha_t = alpha
        else:
            alpha_t = (alpha[None, :, None, None] * targets_onehot).sum(dim=1)  # (B, H, W)

        focal_weight = (1 - p_t) ** self.gamma
        loss_map = -alpha_t * focal_weight * torch.log(p_t + self.eps)

        if self.reduction == "mean":
            return loss_map.mean()
        elif self.reduction == "sum":
            return loss_map.sum()
        return loss_map


class BCEDiceLoss(nn.Module):
    """Perda combinada BCE + Dice para segmentação binária.

    Args:
        bce_weight: peso do termo BCE (1 - Dice).
        eps: epsilon para evitar log(0) e divisão por zero.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Calcula BCEDiceLoss.

        Args:
            logits: (B, 1, H, W) ou (B, C, H, W) com canal 1 = objeto.
            targets: (B, H, W) com valores {0, 1}.
        """
        if logits.shape[1] > 1:
            logits = logits[:, 1:2]

        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.unsqueeze(1).float(), reduction="mean"
        )

        pred_bin = (probs > 0.5).float()
        intersection = (pred_bin * targets.unsqueeze(1).float()).sum()
        union = pred_bin.sum() + targets.unsqueeze(1).float().sum()
        dice = (2.0 * intersection + self.eps) / (union + self.eps)

        return self.bce_weight * bce + (1 - self.bce_weight) * (1 - dice)


class MulticlassDiceLoss(nn.Module):
    """Dice Loss para segmentação multiclasse (uma classe por canal, one-hot).

    Args:
        eps: epsilon para estabilidade numérica.
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        eps: float = 1e-7,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Calcula Dice Loss multiclasse.

        Args:
            logits: (B, C, H, W) logits.
            targets: (B, H, W) com rótulos inteiros em [0, C-1].
        """
        C = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_onehot = (
            F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float().to(logits.device)
        )

        intersection = (probs * targets_onehot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + targets_onehot.sum(dim=(0, 2, 3))
        dice_per_class = (2.0 * intersection + self.eps) / (union + self.eps)

        if self.reduction == "mean":
            return 1 - dice_per_class.mean()
        elif self.reduction == "sum":
            return 1 - dice_per_class.sum()
        return 1 - dice_per_class


class DiceLoss(nn.Module):
    """Dice Loss binário simples. Mantido para compatibilidade com código existente.

    Calcula 1 - Dice Score entre predição e alvo binário.
    """

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Calcula Dice Loss.

        Args:
            logits: (B, 1, H, W) ou (B, H, W).
            targets: (B, H, W) com valores {0, 1}.
        """
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets.unsqueeze(1).float()).sum()
        union = probs.sum() + targets.unsqueeze(1).float().sum()
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        return 1 - dice
