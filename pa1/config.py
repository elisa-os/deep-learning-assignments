"""Carregamento e validação da configuração via YAML.

Usa dataclasses para acesso por atributo (cfg.train.lr) em vez de
dicionário (cfg["train"]["lr"]) — mais legível e com autocomplete.

Suporte a configuração por parte (parte0, parte1, parte2, ...):
  - Na seção "parteN" do YAML, cada parte tem seus próprios valores.
  - load_config(path, parte=N) carrega a seção parte{N} do YAML.
  - Se parte não for informado, usa o comportamento original (campos
    de nível superior do YAML, mantendo compatibilidade com configs
    antigas sem seções por parte).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
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
    use_skips: bool = True  # False = decoder sem skip connections (ablação Eixo 1)


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    checkpoint: str | None = None
    eval_only: bool = False
    loss_gamma: float = 2.0   # gamma da FocalLoss (0 = CE ponderada; ablação Eixo 2)
    loss_alpha: list | None = None  # pesos por classe; None → [1.0, 1.0, 2.5]


@dataclass
class Config:
    seed: int = 42
    output_dir: str = "pa1/outputs"
    parte: Optional[int] = None
    mode_tag: str = "parte0"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(
    path: str | Path = "pa1/config.yaml",
    parte: str | int | None = None,
) -> Config:
    """Lê um arquivo YAML e retorna um Config tipado.

    Se `parte` for informado, carrega a seção correspondente do YAML.
    Aceita:
      - int (ex: 0, 1, 2) → carrega seção `parte{parte}`
      - str "0", "1", "2" → converte para int e carrega seção
      - str "parte0", "parte1" → extrai número e carrega seção
      - str "parte0_baseline" → extrai número e carrega seção

    Caso contrário, usa os campos de nível superior (comportamento
    original, compatível com configs antigas sem seções por parte).

    Campos ausentes no YAML usam os defaults do dataclass.
    Resolve output_dir relativamente à localização do arquivo de configuração.
    """
    # Normaliza parte para int ou None
    parte_int: Optional[int] = None
    if parte is not None:
        if isinstance(parte, int):
            parte_int = parte
        elif isinstance(parte, str):
            s = parte.strip()
            if s.startswith("parte"):
                # extrai o número: "parte0", "parte1", "parte1_baseline" → 0, 1, 1
                num_part = s[len("parte"):]
                # pega apenas os dígitos iniciais
                digits = ""
                for ch in num_part:
                    if ch.isdigit():
                        digits += ch
                    else:
                        break
                if digits:
                    parte_int = int(digits)
            elif s.isdigit():
                parte_int = int(s)

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

    seed = raw.get("seed", 42)
    out_dir_raw = raw.get("output_dir", "outputs")
    out_dir_path = Path(out_dir_raw)
    if not out_dir_path.is_absolute():
        out_dir_path = path.parent / out_dir_path

    # Escolhe a seção de configuração: parte{N} ou nível superior
    if parte_int is not None:
        parte_key = f"parte{parte_int}"
        parte_section = raw.get(parte_key, {})

        # Suporte a duas estruturas de YAML:
        #   a) Sub-dicts: parte2: { data: {...}, model: {...}, train: {...} }
        #   b) Flat (atual): parte2: { synthetic: false, out_channels: 3, epochs: 30, ... }
        # Campos flat são mapeados para as sub-seções correspondentes.
        _DATA_FIELDS  = {"synthetic", "data_dir", "n_samples", "batch_size", "num_workers"}
        _MODEL_FIELDS = {"in_channels", "out_channels", "use_skips"}
        _TRAIN_FIELDS = {"epochs", "lr", "checkpoint", "eval_only", "loss_gamma", "loss_alpha"}

        flat_data  = {k: v for k, v in parte_section.items() if k in _DATA_FIELDS}
        flat_model = {k: v for k, v in parte_section.items() if k in _MODEL_FIELDS}
        flat_train = {k: v for k, v in parte_section.items() if k in _TRAIN_FIELDS}

        # Sub-dicts explícitos sobrescrevem flat, que por sua vez sobrescreve o nível raiz
        data_raw  = {**raw.get("data",  {}), **flat_data,  **parte_section.get("data",  {})}
        model_raw = {**raw.get("model", {}), **flat_model, **parte_section.get("model", {})}
        train_raw = {**raw.get("train", {}), **flat_train, **parte_section.get("train", {})}

        seed = parte_section.get("seed", seed)
        out_dir_raw = parte_section.get("output_dir", out_dir_raw)
        if out_dir_raw:
            out_dir_path = Path(out_dir_raw)
            if not out_dir_path.is_absolute():
                out_dir_path = path.parent / out_dir_path
    else:
        data_raw  = raw.get("data",  {})
        model_raw = raw.get("model", {})
        train_raw = raw.get("train", {})
        if "output_dir" in raw:
            out_dir_raw = raw["output_dir"]
            out_dir_path = Path(out_dir_raw)
            if not out_dir_path.is_absolute():
                out_dir_path = path.parent / out_dir_path

    out_dir_str = str(out_dir_path) if out_dir_raw else str(out_dir_path)

    return Config(
        seed=seed,
        output_dir=out_dir_str,
        parte=parte_int,
        data=DataConfig(**data_raw),
        model=ModelConfig(**model_raw),
        train=TrainConfig(**train_raw),
    )
