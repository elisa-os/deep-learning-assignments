"""Carregamento e validação da configuração via YAML.

Usa dataclasses para acesso por atributo (cfg.train.lr) em vez de
dicionário (cfg["train"]["lr"]) — mais legível e com autocomplete.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class DataConfig:
    synthetic: bool = True
    data_dir: str | None = None
    n_samples: int = 500
    batch_size: int = 8
    num_workers: int = 2


@dataclass
class ModelConfig:
    in_channels: int = 1
    out_channels: int = 2


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    checkpoint: str | None = None
    eval_only: bool = False


@dataclass
class Config:
    seed: int = 42
    output_dir: str = "pa1/outputs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path: str | Path = "pa1/config.yaml") -> Config:
    """Lê um arquivo YAML e retorna um Config tipado.

    Campos ausentes no YAML usam os defaults do dataclass.
    Resolve output_dir relativamente à localização do arquivo de configuração.
    """
    path = Path(path)
    if not path.exists():
        if Path("config.yaml").exists():
            path = Path("config.yaml")
        elif Path("pa1/config.yaml").exists():
            path = Path("pa1/config.yaml")
        else:
            raise FileNotFoundError(f"Config não encontrada: {path}")

    with open(path) as f:
        raw: dict = yaml.safe_load(f) or {}

    out_dir_raw = raw.get("output_dir", "outputs")
    out_dir_path = Path(out_dir_raw)
    if not out_dir_path.is_absolute():
        out_dir_path = path.parent / out_dir_path

    return Config(
        seed=raw.get("seed", 42),
        output_dir=str(out_dir_path),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
    )
