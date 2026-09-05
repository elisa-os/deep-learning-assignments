"""Dataset do Data Science Bowl 2018 (DSB2018 / BBBC038v1) — Passo 1.

O DSB2018 (`stage1_train`) contém ~670 imagens de microscopia de núcleos
celulares. Cada imagem tem suas anotações em PNGs separados — um por núcleo.

Estrutura esperada dos dados brutos:
    stage1_train/
    └── <image_id>/
        ├── images/
        │   └── <image_id>.png   ← imagem RGB(A), tamanhos variados
        └── masks/
            ├── <hash1>.png      ← máscara binária do núcleo 1
            ├── <hash2>.png      ← máscara binária do núcleo 2
            └── ...

O que este módulo faz:
1. Varre as 670 pastas e lê imagem + conjunto de máscaras de cada amostra.
2. Consolida as máscaras individuais em:
   - mask_semantic (H, W) int64 — {0: fundo, 1: núcleo}
   - mask_instances (H, W) int64 — 0 = fundo, k = instância k (1..N)
3. Classifica cada imagem em uma de três modalidades por estatísticas de cor:
   - 0 = Fluorescência (fundo escuro, núcleos brilhantes — grayscale)
   - 1 = Campo claro (fundo claro, núcleos escuros — grayscale)
   - 2 = Histologia H&E (colorido, tons roxo/rosa)
4. Faz split estratificado 70/15/15 (treino/val/teste) preservando a
   proporção de cada modalidade em cada partição.
5. Expõe um DataLoader PyTorch com data augmentation para treino
   (HorizontalFlip, VerticalFlip, RandomRotate90, RandomCrop 256×256).
"""

from __future__ import annotations

from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Detecção de Modalidade
# ─────────────────────────────────────────────────────────────────────────────

def detect_modality(img_rgb: np.ndarray) -> int:
    """Classifica a imagem em uma de três modalidades biológicas.

    Parâmetros
    ----------
    img_rgb : np.ndarray
        Array (H, W, 3), dtype uint8.

    Retorna
    -------
    int
        0 = Fluorescência (grayscale + fundo escuro)
        1 = Campo claro   (grayscale + fundo claro)
        2 = Histologia H&E (colorida)
    """
    r_mean = img_rgb[:, :, 0].astype(float).mean()
    g_mean = img_rgb[:, :, 1].astype(float).mean()
    b_mean = img_rgb[:, :, 2].astype(float).mean()
    ch_means = np.array([r_mean, g_mean, b_mean])

    # H&E: desbalanceamento de cor entre canais > 10
    if ch_means.max() - ch_means.min() > 10:
        return 2  # H&E

    brightness = ch_means.mean()
    if brightness < 80:
        return 0  # Fluorescência
    return 1      # Campo claro


# ─────────────────────────────────────────────────────────────────────────────
# Varredura e indexação do dataset
# ─────────────────────────────────────────────────────────────────────────────

def scan_dataset(data_dir: str | Path) -> list[dict]:
    """Varre o diretório stage1_train e retorna metadados das amostras."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Diretório de dados não encontrado: {data_dir}\n"
            "Verifique o caminho em config.yaml (chave data.data_dir)."
        )

    records: list[dict] = []
    for sample_dir in sorted(data_dir.iterdir()):
        if not sample_dir.is_dir():
            continue

        img_dir = sample_dir / "images"
        mask_dir = sample_dir / "masks"

        if not img_dir.exists() or not mask_dir.exists():
            continue

        img_files = list(img_dir.iterdir())
        if not img_files:
            continue

        img_path = img_files[0]
        mask_paths = sorted(mask_dir.iterdir())
        if not mask_paths:
            continue

        # Detecta modalidade a partir da imagem RGB
        img_arr = np.array(Image.open(img_path).convert("RGB"))
        modality = detect_modality(img_arr)

        records.append(
            {
                "image_path": str(img_path),
                "mask_paths": [str(p) for p in mask_paths],
                "modality": modality,
            }
        )

    if not records:
        raise RuntimeError(
            f"Nenhuma amostra encontrada em {data_dir}. "
            "Verifique se o stage1_train foi extraído corretamente."
        )
    return records


def stratified_split(
    records: list[dict],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Divide os registros em treino/val/teste com estratificação por modalidade."""
    rng = np.random.default_rng(seed)

    by_modality: dict[int, list[dict]] = {}
    for rec in records:
        m = rec["modality"]
        by_modality.setdefault(m, []).append(rec)

    train_recs: list[dict] = []
    val_recs: list[dict] = []
    test_recs: list[dict] = []

    for mod, group in sorted(by_modality.items()):
        indices = rng.permutation(len(group))
        n = len(group)
        n_val = max(1, int(round(n * val_frac)))
        n_test = max(1, int(round(n * (1.0 - train_frac - val_frac))))
        n_train = n - n_val - n_test

        train_recs.extend([group[i] for i in indices[:n_train]])
        val_recs.extend([group[i] for i in indices[n_train:n_train + n_val]])
        test_recs.extend([group[i] for i in indices[n_train + n_val:]])

    return train_recs, val_recs, test_recs


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento e consolidação de uma amostra
# ─────────────────────────────────────────────────────────────────────────────

def load_sample(
    record: dict,
    target_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carrega uma imagem e consolida suas máscaras individuais."""
    pil_img = Image.open(record["image_path"]).convert("RGB")

    if target_size is not None:
        h, w = pil_img.size[1], pil_img.size[0]
        min_side = min(h, w)
        if min_side < target_size:
            scale = target_size / min_side
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)

    image = np.array(pil_img)  # (H, W, 3) uint8

    H, W = image.shape[:2]
    instance_mask = np.zeros((H, W), dtype=np.int32)
    semantic_mask = np.zeros((H, W), dtype=np.uint8)

    # Consolida máscaras individuais
    for inst_id, mask_path in enumerate(record["mask_paths"], start=1):
        m = np.array(Image.open(mask_path))

        if m.shape[:2] != (H, W):
            m = np.array(Image.fromarray(m).resize((W, H), Image.NEAREST))

        if m.ndim == 3:
            m = m[:, :, 0]
        nucleus_pixels = m > 127

        instance_mask[nucleus_pixels] = inst_id
        semantic_mask[nucleus_pixels] = 1

    return image, semantic_mask, instance_mask


# ─────────────────────────────────────────────────────────────────────────────
# Augmentations
# ─────────────────────────────────────────────────────────────────────────────

def make_train_transform(crop_size: int = 256) -> A.Compose:
    """Pipeline de augmentation para treino."""
    return A.Compose(
        [
            A.RandomCrop(height=crop_size, width=crop_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        additional_targets={
            "mask_semantic": "mask",
            "mask_instances": "mask",
        },
    )


def make_val_transform(crop_size: int = 256) -> A.Compose:
    """Pipeline de validação/teste: crop central e normalização."""
    return A.Compose(
        [
            A.CenterCrop(height=crop_size, width=crop_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        additional_targets={
            "mask_semantic": "mask",
            "mask_instances": "mask",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset PyTorch
# ─────────────────────────────────────────────────────────────────────────────

class DSB2018Dataset(Dataset):
    """Dataset do Data Science Bowl 2018 para segmentação de instâncias."""

    def __init__(
        self,
        records: list[dict],
        transform: A.Compose | None = None,
        crop_size: int = 256,
        target_size_for_small: int = 256,
    ) -> None:
        self.records = records
        self.transform = transform
        self.crop_size = crop_size
        self.target_size_for_small = target_size_for_small

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]

        image, sem_mask, inst_mask = load_sample(
            record, target_size=self.target_size_for_small
        )

        if self.transform is not None:
            augmented = self.transform(
                image=image,
                mask_semantic=sem_mask.astype(np.int32),
                mask_instances=inst_mask.astype(np.int32),
            )
            img_tensor = augmented["image"].float()           # (3, H, W) float32
            sem_tensor = augmented["mask_semantic"].long()    # (H, W) int64
            inst_tensor = augmented["mask_instances"].long()  # (H, W) int64
        else:
            img_tensor = (
                torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            )
            sem_tensor = torch.from_numpy(sem_mask).long()
            inst_tensor = torch.from_numpy(inst_mask).long()

        return {
            "image": img_tensor,
            "mask_semantic": sem_tensor,
            "mask_instances": inst_tensor,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Fábrica pública: make_dsb2018_loaders
# ─────────────────────────────────────────────────────────────────────────────

def make_dsb2018_loaders(
    data_dir: str | Path,
    batch_size: int = 8,
    num_workers: int = 2,
    crop_size: int = 256,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Cria os DataLoaders de treino, validação e teste para o DSB2018."""
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    print(f"[DSB2018] Varrendo {data_dir} ...")
    records = scan_dataset(data_dir)
    print(f"[DSB2018] {len(records)} amostras encontradas.")

    modality_names = {0: "Fluorescência", 1: "Campo Claro", 2: "H&E"}
    for mod, name in modality_names.items():
        count = sum(1 for r in records if r["modality"] == mod)
        print(f"           {name}: {count} imagens")

    train_recs, val_recs, test_recs = stratified_split(
        records, train_frac=train_frac, val_frac=val_frac, seed=seed
    )
    print(
        f"[DSB2018] Split: treino={len(train_recs)} | "
        f"val={len(val_recs)} | teste={len(test_recs)}"
    )

    train_tf = make_train_transform(crop_size=crop_size)
    val_tf = make_val_transform(crop_size=crop_size)

    train_ds = DSB2018Dataset(train_recs, transform=train_tf, crop_size=crop_size)
    val_ds = DSB2018Dataset(val_recs, transform=val_tf, crop_size=crop_size)
    test_ds = DSB2018Dataset(test_recs, transform=val_tf, crop_size=crop_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
