"""Detecção do dispositivo de computação.

CPU vs GPU em Deep Learning:
-----------------------------
GPUs (Graphics Processing Units) foram originalmente feitas para renderizar
pixels em paralelo. Pixels independentes → operações paralelas massivas.
Essa mesma estrutura é perfeita para multiplicações de matrizes em redes neurais.

Uma GPU moderna tem milhares de núcleos simples (CUDA cores), enquanto uma CPU
tem dezenas de núcleos complexos. Para as operações de álgebra linear de DL,
a GPU ganha fácil: treinar uma UNet em CPU pode levar horas; na GPU, minutos.

CUDA é o framework da NVIDIA que permite usar a GPU para computação geral.
PyTorch usa CUDA automaticamente quando disponível.
"""

import torch


def get_device() -> torch.device:
    """Retorna GPU se disponível, senão CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | PyTorch: {torch.__version__}")
    return device
