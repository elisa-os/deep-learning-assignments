# PA1 — Segmentação de Instâncias

Aprendizado Profundo — Programming Assignment 1  
Duplas · Entrega: 10/09, 23h59 — B41254@fgv.edu.br

## O problema

O PA pede que as arquiteturas vistas em aula (U-Net, ResUNet, DeepLab etc.) produzam rótulos instance-aware sem detectores baseados em proposta de região (Mask R-CNN etc. proibidos). Três decisões acopladas: representação de saída, perda e pós-processamento.

## Entregáveis

- Repositório Git (histórico distribuído ao longo das duas semanas)
- `README.md` (este arquivo) — ambiente, download dos dados, comandos de treino/avaliação
- `AI_LOG.md` — como usaram IA no assignment
- `inferencia.ipynb` — recebe caminho de uma imagem, devolve máscara de instâncias colorida e contagem, roda sem retreinar
- Checkpoint (pesos do modelo final)

Não há relatório escrito; a avaliação é a apresentação. Toda tabela, curva e imagem mostrada tem que ser reproduzível a partir do repositório.

## Ambiente

### Pré-requisitos

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) para gerenciamento de pacotes e execução

### Instalação

```bash
cd deep-learning-assignments/pa1
uv sync
```

### Dados

**Opção A (padrão) — DSB2018 / BBBC038v1**

Microscopia com máscara individual por núcleo (~670 imagens de treino, cada núcleo em PNG separado, sem sobreposição).

Download direto: https://bbbc.broadinstitute.org/BBBC038  
Ou via Kaggle: `kaggle competitions download -c data-science-bowl-2018` (usar stage1_train)

**Opção B — Edificações em imagem aérea**

AICrowd / CrowdAI Mapping Challenge. Tiles RGB de 300×300 com anotações COCO. Exige conta na plataforma.

### Configuração

Edite `pa1/config.yaml` (ou passe outro arquivo via `--config`). Principais campos:

```yaml
output_dir: outputs          # onde os arquivos de saída vão
seed: 42
data:
  synthetic: false           # true = dataset sintético (Parte 0); false = DSB2018
  data_dir: /caminho/para/dsb2018/stage1_train
  batch_size: 8
  num_workers: 2
train:
  epochs: 20
  lr: 1e-3
  checkpoint: null           # caminho opcional para checkpoint de início
```

### Comandos

O pipeline do PA1 usa um **argumento posicional de parte** que seleciona automaticamente a seção correspondente em `pa1/config.yaml`:

```bash
uv run pa1 0          # Parte 0 — sintético (elipses)
uv run pa1 1          # Parte 1 — baseline binária (DSB2018)
uv run pa1 2          # Parte 2 — Trilha A (3 classes + Watershed)
uv run pa1 parte 2    # equivalente a "2"
uv run pa1 parte2     # equivalente a "2"
uv run pa1 parte2_baseline  # equivalente a "2"
```

#### Parte 0 — Teste unitário sintético

```bash
uv run pa1 0                     # treino completo com config da parte0
uv run pa1 0 --epochs 10         # sobrescreve epochs
uv run pa1 0 --epochs 5 --batch-size 16 --lr 1e-3  # teste rápido
uv run pa1 0 --eval-only --checkpoint outputs/parte0_baseline_unet.pt  # avalia só
```

#### Parte 1 — Baseline binária (DSB2018)

```bash
uv run pa1 1                     # treino completo
uv run pa1 1 --epochs 20         # sobrescreve epochs
uv run pa1 1 --epochs 5 --batch-size 4 --lr 1e-3   # teste rápido
uv run pa1 1 --eval-only --checkpoint outputs/parte1_baseline_unet.pt  # avalia só
```

> **Pré-requisito:** o diretório `pa1/data/stage1_train/` deve existir.

#### Parte 2 — Trilha A (3 classes + Watershed)

```bash
uv run pa1 2                     # treino completo (30 epochs por padrão)
uv run pa1 2 --epochs 30         # sobrescreve epochs
uv run pa1 2 --epochs 5 --batch-size 4 --lr 1e-3   # teste rápido
uv run pa1 2 --eval-only --checkpoint outputs/trilhaA_baseline_unet.pt  # avalia só
```

> **Pré-requisito:** o diretório `pa1/data/stage1_train/` deve existir.

#### Override pontual

Qualquer flag de override funciona com qualquer parte:

```bash
uv run pa1 0 --epochs 10 --lr 1e-4
uv run pa1 2 --eval-only --checkpoint outputs/trilhaA_baseline_unet.pt
uv run pa1 --help
```

## Saída (arquivos gerados)

Todos na pasta `outputs/` (ou onde `output_dir` apontar):

|| Arquivo | Descrição |
|---------|-----------|
| `parte1_baseline_unet.pt` | Pesos do modelo treinado — baseline binária (Parte 1) |
| `parte1_baseline_results.json` | Métricas médias de validação e teste do baseline (IoU, Dice, mAP, erro de contagem) |
| `parte1_per_image_instance_metrics.csv` | Métricas completas por imagem do baseline |
| `trilhaA_baseline_unet.pt` | Pesos do modelo treinado com 3 classes (Parte 2 — Trilha A) |
| `trilhaA_baseline_results.json` | Métricas médias de validação e teste da Trilha A (IoU, Dice, mAP, erro de contagem) |
| `trilhaA_per_image_instance_metrics.csv` | Métricas completas por imagem da Trilha A (Watershed) |
| `parte0_resultados.png` / `parte1_resultados.png` / `trilhaA_resultados.png` | Gráficos: loss, métricas semânticas, mAP vs. densidade |
| `parte0_qualitativo.png` / `parte1_qualitativo.png` / `trilhaA_qualitativo.png` | Grid com as 4 piores imagens pelo mAP |
| `synthetic_samples.png` | Amostras do dataset sintético (só em modo sintético) |

## Métricas por imagem (`parte1_per_image_instance_metrics.csv`)

O CSV contém **uma linha por imagem avaliada**, com:

- `idx` — índice da imagem na ordem de iteração
- `n_gt` — número de instâncias GT na imagem
- `n_pred` — número de instâncias previstas
- `count_error` — erro absoluto de contagem (`abs(n_pred - n_gt)`)
- `iou_sem` — IoU semântico binário para a imagem
- `dice_sem` — Dice semântico binário para a imagem
- `mAP` — mAP@[0.50:0.95] da imagem (média sobre os 10 limiares)
- Para cada limiar `T` em {50, 55, 60, 65, 70, 75, 80, 85, 90, 95}:
  - `tp_T` — verdadeiros positivos (instâncias previstas com IoU ≥ T/100)
  - `fp_T` — falsos positivos
  - `fn_T` — falsos negativos
  - `ap_T` — AP por limiar (F1 proxy usado no cálculo do mAP)

Exemplo de uso para analisar as piores imagens:

```bash
uv run python -c "import pandas as pd; df = pd.read_csv('outputs/parte1_per_image_instance_metrics.csv'); print(df.sort_values('mAP').head(5))"
```

## Estrutura do código

```
pa1/
├── __init__.py
├── main.py                    # ponto de entrada; treino, avaliação, salvamento
├── config.py                  # configuração via dataclass + YAML
├── models/
│   ├── __init__.py
│   └── unet.py                # arquitetura UNet
├── losses/
│   ├── __init__.py
│   └── segmentation.py        # BCEDiceLoss
├── data/
│   ├── __init__.py
│   ├── synthetic.py           # dataset de elipses sintéticas
│   ├── dsb2018.py             # loader DSB2018
│   └── targets.py             # formatação de targets
├── metrics/
│   ├── __init__.py
│   └── instance.py            # mAP, matching (hungarian/greedy), IoU matrix
├── postprocessing/
│   ├── __init__.py
│   ├── connected_components.py
│   └── watershed.py
├── utils/
│   ├── __init__.py
│   ├── seed.py
│   ├── device.py
│   ├── visualize.py           # plotagem de amostras, grids, gráficos
│   └── export.py              # PerImageMetricsWriter (exportação CSV)
└── PA1.pdf                    # enunciado completo
```

## Métodos de avaliação implementados

- `metrics/instance.py`:
  - `compute_map()` — mAP@[0.50:0.95] com 10 limiares, devolve mAP, AP por limiar, erro de contagem, n_pred, n_gt, e `per_threshold_details` (tp/fp/fn para cada limiar).
  - `match_instances()` — suporta `"hungarian"` (ótimo, via `scipy.optimize.linear_sum_assignment`) e `"greedy"` (por IoU decrescente).
  - `compute_iou_matrix()` — matriz IoU entre predições e GT.

## Regras de matching

A documentação do matching usado deve ficar explícita na apresentação (Parte 1, item 4). O código suporta ambos; o padrão no `compute_map` é `"hungarian"`.

## Abordagem de exportação de métricas

As métricas por imagem são exportadas via `PerImageMetricsWriter` (em `pa1/utils/export.py`), acoplado ao loop de avaliação no `evaluate()` do `main.py`. O escritor é opcional — quando não fornecido, a avaliação roda normalmente sem persistência. Isso permite reuso nas partes seguintes com prefixos diferentes (`parte2_...`, `parte3_...`) changeando apenas o nome do arquivo no parâmetro `filename` de `writer.write()`.
