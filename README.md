# Deep Learning Assignments — FGV CDIA

**Disciplina:** Aprendizado Profundo | FGV CDIA  
**Professor:** Dario Oliveira | **Monitor:** Erick Brito  
**Alunos:** Bruno Ferreira & Elisa Soares  

Este repositório centraliza os trabalhos práticos (Programming Assignments) da disciplina de Aprendizado Profundo. Cada assignment é encapsulado em sua respectiva pasta (`pa1/`, e futuramente `pa2/`, etc.), mantendo código, configurações, dados e artefatos organizados de forma independente.

---

## 📁 Estrutura do Repositório

```
deep-learning-assignments/
├── pyproject.toml              # Gerenciador de dependências e comandos uv
├── uv.lock                     # Lockfile determinístico do ambiente
├── README.md                   # Visão geral do repositório
├── .gitignore                  # Arquivos e diretórios ignorados
│
└── pa1/                        # ── PA1: Segmentação de Instâncias ──
    ├── PA1.pdf                 # Enunciado oficial do PA1
    ├── PLANO_DE_EXECUCAO.md    # Plano de execução sequencial detalhado (o que, onde e como)
    ├── config.yaml             # Arquivo central de hiperparâmetros e caminhos
    ├── main.py                 # Ponto de entrada CLI do PA1
    ├── config.py               # Parsing tipado de dataclasses
    ├── data/                   # Datasets (elipses sintéticas e stage1_train do DSB2018)
    │   ├── synthetic.py        # [✅ Concluído] Gerador de elipses (Parte 0)
    │   ├── dsb2018.py          # [✅ Concluído] Loader e split estratificado (Passo 1)
    │   └── targets.py          # [✅ Concluído] Geração do mapa 3 classes (Passo 2)
    ├── models/                 # Arquiteturas de rede
    │   └── unet.py             # [✅ Concluído] U-Net base
    ├── losses/                 # Funções de perda
    │   └── segmentation.py     # [✅ Concluído] BCEDiceLoss, Focal Loss multiclasse
    ├── postprocessing/         # Decodificação de instâncias
    │   ├── connected_components.py # [✅ Concluído] Extração ingênua (Passo 3)
    │   └── watershed.py        # [✅ Concluído] Watershed com marcadores (Passo 3)
    ├── metrics/                # Métricas de avaliação
    │   └── instance.py         # [✅ Concluído] Hungarian/Greedy matching, mAP e erro de contagem
    ├── utils/                  # Plotters, exportação de figuras e utilitários
    │   └── export.py           # [✅ Concluído] PerImageMetricsWriter (exportação CSV por imagem)
    ├── outputs/                # Figuras geradas, métricas e checkpoints salvos
    └── pa1.ipynb               # [Legado] Notebook inicial exploratório
```

---

## 🎯 PA1 — Segmentação de Instâncias

O **Programming Assignment 1 (PA1)** foca em adaptar arquiteturas de segmentação semântica vistas em aula (em especial a **U-Net**) para realizarem **segmentação de instâncias**, sem o uso de detectores com proposta de região (Mask R-CNN, YOLO, SAM, etc.).

### Decisões Técnicas do PA1:
* **Dataset:** Opção A — **Data Science Bowl 2018 / BBBC038v1** (`stage1_train`, com ~670 imagens e máscaras individuais de núcleos).
* **Trilha:** **Trilha A — Fronteiras e Watershed** (representação de 3 classes: Fundo, Interior e Fronteira, decodificada via Watershed com marcadores).
* **Ablações:**
  * *Eixo 1 (Recuperação de Resolução):* U-Net com Skip Connections vs. Decodificador sem skips.
  * *Eixo 2 (Funções de Perda):* Cross-Entropy ponderada vs. Focal Loss variando $\gamma \in \{0, 1, 2, 5\}$.
* **Teste de Estresse:** Corrupções sintéticas (Gaussian Blur, Ruído Gaussiano e Contraste) em 3 intensidades com curvas de degradação do mAP.
* **Métrica Oficial:** mAP@[0.50:0.05:0.95] computado via matching Hungarian e Erro Médio de Contagem.

Consulte o documento completo:  
👉 **[pa1/PLANO_DE_EXECUCAO.md](pa1/PLANO_DE_EXECUCAO.md)**

---

## 🚀 Como Rodar o PA1

O Pipeline do PA1 é controlado pela flag `synthetic` em `pa1/config.yaml` (ou via CLI `--synthetic`/`--no-synthetic`). Os demais parâmetros (`data_dir`, `epochs`, `lr`, etc.) também podem ser sobrescritos diretamente na linha de comando.

### Parte 0 — Teste unitário sintético (elipses)

Treina com dataset sintético de elipses e gera os gráficos + grid qualitativo. Roda em poucos minutos.

```bash
# 1. Com configuração padrão (YAML): configura epochs, lr, batch_size no config.yaml
uv run pa1 --synthetic

# 2. Sobrescrevendo epochs explicitamente na linha de comando:
uv run pa1 --synthetic --epochs 10     # default no YAML: 20; ajuste conforme necessidade

# 3. Teste rápido com menos épocas e batch pequeno:
uv run pa1 --synthetic --epochs 5 --batch-size 16 --lr 1e-3

# 4. Avaliação só (sem retreinar) a partir de checkpoint:
uv run pa1 --synthetic --eval-only --checkpoint pa1/outputs/parte0_checkpoint.pt
```

**Saídas (em `pa1/outputs/`):**
- `pa1/outputs/synthetic_samples.png` — grid 2×4 com imagens + máscara de instâncias sintéticas
- `pa1/outputs/parte0_resultados.png` — curvas de loss/IoU/Dice + dispersão mAP × densidade
- `pa1/outputs/parte0_qualitativo.png` — grid 4×4 comparativo (imagem / GT / predição / binário)

### Parte 1 — Baseline com dados reais (DSB2018)

Treina U-Net binária sobre microscopia de núcleos e reporta IoU/Dice + mAP de instâncias.

**Pré-requisito:** o diretório `pa1/data/stage1_train/` com as imagens e máscaras do DSB2018 deve existir (baixe e extraia conforme o enunciado do PA1).

```bash
# 1. Treino completo com configuração padrão (YAML):
uv run pa1 --no-synthetic

# 2. Sobrescrevendo epochs explicitamente:
uv run pa1 --no-synthetic --epochs 20   # default no YAML: 20; ajuste conforme necessidade

# 3. Execução rápida de teste (reduz epochs e batch):
uv run pa1 --no-synthetic --epochs 5 --batch-size 4 --lr 1e-3

# 4. Avaliação só (sem retreinar) a partir de checkpoint:
uv run pa1 --no-synthetic --eval-only --checkpoint pa1/outputs/parte1_baseline_unet.pt
```

> **Dica de epochs:** o YAML `pa1/config.yaml` define o número de epochs usado quando nenhuma flag `--epochs` é passada. Para rodar com um número específico, passe `--epochs N` na linha de comando (ex: `--epochs 20`). Ambos os modos aceitam a mesma flag.

**Saídas (em `pa1/outputs/`):**
- `pa1/outputs/parte1_baseline_unet.pt` — pesos do modelo treinado (checkpoint)
- `pa1/outputs/parte1_baseline_results.json` — resumo JSON com métricas médias de val e test
- `pa1/outputs/parte1_per_image_instance_metrics.csv` — métricas completas por imagem (ver abaixo)
- `pa1/outputs/parte1_resultados.png` — curvas de loss/IoU/Dice + dispersão mAP × densidade
- `pa1/outputs/parte1_qualitativo.png` — grid 4×4 comparativo sobre dados reais (imagem / GT / predição / binário)

---

## 📦 Saídas do Pipeline

Todas as figuras e artefatos são salvos diretamente em `pa1/outputs/`. Se o arquivo já existir, ele é substituído. Para arquivar execuções anteriores, crie subpastas dentro de `outputs/` (ex: `pa1/outputs/historico/`); o pipeline não lê nem modifica arquivos dentro de subpastas.

### Parte 0 — Sintético

| Arquivo | Quando é gerado | Descrição |
|---------|-----------------|-----------|
| `outputs/synthetic_samples.png` | Parte 0 (sintético) | Grid 2×4: imagens de elipses + máscaras de instâncias coloridas |
| `outputs/parte0_resultados.png` | Parte 0 (sintético) | Painel 3 gráficos: loss, IoU/Dice semânticos, mAP vs. densidade |
| `outputs/parte0_qualitativo.png` | Parte 0 (sintético) | Grid 4×4: imagem original, GT instâncias, predição, GT binário |

### Parte 1 — Baseline (DSB2018)

| Arquivo | Quando é gerado | Descrição |
|---------|-----------------|-----------|
| `outputs/parte1_baseline_unet.pt` | Parte 1 (treino ou eval-only) | Pesos do modelo final (checkpoint) |
| `outputs/parte1_baseline_results.json` | Parte 1 (final da avaliação) | Métricas médias de validação e teste: IoU, Dice, mAP, erro de contagem |
| `outputs/parte1_per_image_instance_metrics.csv` | Parte 1 (final da avaliação) | Métricas completas por imagem (ver seção abaixo) |
| `outputs/parte1_resultados.png` | Parte 1 (DSB2018) | Mesmo formato do parte0_resultados.png, mas sobre dados reais |
| `outputs/parte1_qualitativo.png` | Parte 1 (DSB2018) | Mesmo formato do parte0_qualitativo.png, mas sobre dados reais |

---

## 📊 Métricas por imagem — `parte1_per_image_instance_metrics.csv`

O CSV contém **uma linha por imagem avaliada** no conjunto de validação (e teste, se disponível). É gerado ao final da execução de `uv run pa1 --no-synthetic` (ou `eval-only`).

### Colunas

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `idx` | int | Índice sequencial da imagem na ordem de iteração do loader |
| `n_gt` | int | Número de instâncias GT na imagem |
| `n_pred` | int | Número de instâncias previstas pelo modelo |
| `count_error` | int | Erro absoluto de contagem: `abs(n_pred - n_gt)` |
| `iou_sem` | float | IoU semântico binário para a imagem |
| `dice_sem` | float | Dice semântico binário para a imagem |
| `mAP` | float | mAP@[0.50:0.95] da imagem (média sobre os 10 limiares) |

Para cada limiar `T` em {50, 55, 60, 65, 70, 75, 80, 85, 90, 95} (passo 0.05):
| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `tp_T` | int | Verdadeiros positivos (instâncias previstas com IoU ≥ T/100) |
| `fp_T` | int | Falsos positivos |
| `fn_T` | int | Falsos negativos |
| `ap_T` | float | AP por limiar (F1 proxy usado no cálculo do mAP) |

Exemplo de colunas geradas:

```
idx,n_gt,n_pred,count_error,iou_sem,dice_sem,mAP,tp_50,fp_50,fn_50,ap_50,tp_55,fp_55,fn_55,ap_55,...
0,5,3,2,0.71,0.83,0.62,3,0,2,0.60,3,0,2,0.60,...
1,2,2,0,0.88,0.93,0.85,2,0,0,0.80,2,0,0,0.80,...
```

### Como consultar

```bash
cd pa1

# 5 piores imagens pelo mAP (para galeria de falhas da Parte 5):
uv run python -c "import pandas as pd; df=pd.read_csv('outputs/parte1_per_image_instance_metrics.csv'); print(df.sort_values('mAP').head(5))"

# 5 melhores imagens:
uv run python -c "import pandas as pd; df=pd.read_csv('outputs/parte1_per_image_instance_metrics.csv'); print(df.sort_values('mAP', ascending=False).head(5))"

# Distribuição de erro de contagem:
uv run python -c "import pandas as pd; df=pd.read_csv('outputs/parte1_per_image_instance_metrics.csv'); print(df['count_error'].value_counts().sort_index())"
```

### Reuso nas partes seguintes

O exportador é implementado em `pa1/utils/export.py` (`PerImageMetricsWriter`) e recebe um `filename` no momento da escrita:

```python
from pa1.utils import PerImageMetricsWriter
writer = PerImageMetricsWriter(Path("outputs"))
# ... durante avaliação, writer.add(...) por imagem ...
csv_path = writer.write("parte2_per_image_instance_metrics.csv")
```

Isso permite manter arquivos separados por parte (`parte2_...`, `parte3_...`) sem mudar a lógica de avaliação.

---

## ⚙️ Configuração do Ambiente com `uv`

O projeto utiliza [`uv`](https://docs.astral.sh/uv/) para gerenciar dependências e comandos. O `uv` não exige a pasta `src/` — o pacote `pa1` está configurado diretamente na raiz via `module-root = "."` no `pyproject.toml`.

```bash
# Sincroniza dependências e cria o ambiente virtual isolado (.venv)
uv sync
```

---

## 📂 Configuração via `pa1/config.yaml`

O comportamento do pipeline é controlado por `pa1/config.yaml`:

```yaml
seed: 42
output_dir: outputs

data:
  synthetic: false              # false = dados reais DSB2018, true = elipses sintéticas (Parte 0)
  data_dir: pa1/data/stage1_train  # caminho dos dados reais do DSB2018
  n_samples: 500                # amostras no dataset sintético (quando synthetic: true)
  batch_size: 8
  num_workers: 2

model:
  in_channels: 3                # 3 = RGB (DSB2018), 1 = escala de cinza (sintético)
  out_channels: 2               # 2 = binário (fundo/objeto)

train:
  epochs: 20
  lr: 1.0e-3
  checkpoint: null              # ex: pa1/outputs/parte1_baseline_unet.pt
  eval_only: false              # true = pula treino, só avalia
```

Overrides via linha de comando (ex: `--epochs 5 --lr 1e-3`) sobrescrevem os valores do YAML.

---

## 📄 Notas

- **Regra de histórico:** Se uma imagem com o mesmo nome já existir na raiz de `outputs/`, ela é substituída pela mais recente. Caso deseje arquivar execuções anteriores, crie subpastas dentro de `outputs/` (ex: `outputs/historico/`); o pipeline não mexe nem lê arquivos dentro de subpastas.
- **Checkpoint de avaliação única:** Use `--eval-only --checkpoint pa1/outputs/parte1_baseline_unet.pt` para avaliar sem retreinar. O CSV de métricas por imagem também é gerado nesse modo.
- **API de exportação:** A classe `PerImageMetricsWriter` está disponível em `pa1/utils/export.py` para reuso em partes subsequentes.
