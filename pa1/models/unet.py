"""UNet — arquitetura de segmentação semântica (Parte 1).

O que é segmentação semântica?
--------------------------------
Em vez de classificar a imagem inteira (classificação), ou detectar caixas
(object detection), aqui classificamos CADA PIXEL: fundo ou objeto.

A UNet (Ronneberger et al., 2015) é a arquitetura padrão para isso em
imagens médicas e científicas. Ela tem duas partes simétricas:

┌─────────────────────────────────────────────────────┐
│  ENCODER (contração)          DECODER (expansão)    │
│                                                     │
│  Imagem → Conv → Pool → ...→ UpConv → Conv → Másc  │
│             ↓         skip           ↑              │
│           Pool → ...─────────────── ↑              │
│                                                     │
│  * Encoder: extrai FEATURES (o que há na imagem)    │
│  * Decoder: reconstrói ONDE estão os objetos        │
│  * Skip connections: passam detalhes do encoder     │
│    direto para o decoder (recuperam resolução)      │
└─────────────────────────────────────────────────────┘

Por que perder resolução (pooling) para depois recuperar?
---------------------------------------------------------
O pooling aumenta o campo receptivo: neurônios mais profundos "veem"
regiões maiores da imagem. Isso permite detectar contexto ("há uma célula
aqui porque estou rodeado de outras células"). O decoder recupera a
resolução para dar a resposta pixel a pixel.

Convoluções 3×3:
-----------------
Uma convolução 3×3 aplica um filtro 3×3 deslizante sobre a imagem.
Cada filtro aprende a detectar um padrão (borda, textura, forma).
Com N filtros, temos N "mapas de features" na saída.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    """Bloco Conv→BN→ReLU→Conv→BN→ReLU (bloco básico da UNet).

    Batch Normalization (BN):
    --------------------------
    Normaliza a ativação de cada batch para média≈0, std≈1.
    Isso estabiliza o treino: sem BN, gradientes podem explodir ou
    desaparecer conforme passam pelas camadas (problema do vanishing gradient).

    ReLU:
    ------
    A função de ativação ReLU(x) = max(0, x) introduz não-linearidade.
    Sem ativações não-lineares, empilhar camadas lineares não faz sentido —
    qualquer composição de funções lineares é ainda uma função linear.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """UNet para segmentação binária ou multi-classe.

    Args:
        in_channels: Canais de entrada (1 para escala de cinza, 3 para RGB).
        out_channels: Classes de saída (2 para binário, N para N classes).
        features: Lista com o número de filtros em cada nível do encoder.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        features: list[int] | None = None,
    ) -> None:
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        # ---- Encoder ----
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(_DoubleConv(ch, f))
            self.pools.append(nn.MaxPool2d(2))
            ch = f

        # ---- Bottleneck (fundo do U) ----
        self.bottleneck = _DoubleConv(features[-1], features[-1] * 2)

        # ---- Decoder ----
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        rev = list(reversed(features))
        ch = features[-1] * 2
        for f in rev:
            self.upconvs.append(nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2))
            self.decoders.append(_DoubleConv(f * 2, f))  # *2 por causa do skip
            ch = f

        # ---- Cabeça final ----
        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []

        # Encoder: guarda as ativações para o skip
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        # Decoder: upsampling + concatena skip + convoluções
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            # Alinha tamanhos caso haja diferença de 1 pixel (padding)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.head(x)
