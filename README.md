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
uv run pa1 --no-synthetic --eval-only --checkpoint pa1/outputs/parte1_checkpoint.pt
```

> **Dica de epochs:** o YAML `pa1/config.yaml` define o número de epochs usado quando nenhuma flag `--epochs` é passada. Para rodar com um número específico, passe `--epochs N` na linha de comando (ex: `--epochs 20`). Ambos os modos aceitam a mesma flag.

**Saídas (em `pa1/outputs/`):**
- `pa1/outputs/parte1_resultados.png` — curvas de loss/IoU/Dice + dispersão mAP × densidade
- `pa1/outputs/parte1_qualitativo.png` — grid 4×4 comparativo sobre dados reais (imagem / GT / predição / binário)

## 📦 Saídas do Pipeline

Todas as figuras são salvas diretamente em `outputs/` (raiz do projeto). Se o arquivo já existir, ele é substituído. Para arquivar execuções anteriores, crie subpastas dentro de `outputs/` (ex: `outputs/historico/`); o pipeline não lê nem modifica arquivos dentro de subpastas.

| Arquivo | Quando é gerado | Descrição |
|---------|-----------------|-----------|
| `outputs/synthetic_samples.png` | Parte 0 (sintético) | Grid 2×4: imagens de elipses + máscaras de instâncias coloridas |
| `outputs/parte0_resultados.png` | Parte 0 (sintético) | Painel 3 gráficos: loss, IoU/Dice semânticos, mAP vs. densidade |
| `outputs/parte0_qualitativo.png` | Parte 0 (sintético) | Grid 4×4: imagem original, GT instâncias, predição, GT binário |
| `outputs/parte1_resultados.png` | Parte 1 (DSB2018) | Mesmo formato do parte0_resultados.png, mas sobre dados reais |
| `outputs/parte1_qualitativo.png` | Parte 1 (DSB2018) | Mesmo formato do parte0_qualitativo.png, mas sobre dados reais |

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
  checkpoint: null              # ex: pa1/outputs/checkpoint.pt
  eval_only: false              # true = pula treino, só avalia
```

Overrides via linha de comando (ex: `--epochs 5 --lr 1e-3`) sobrescrevem os valores do YAML.

---

## 📄 Arquivos de Saída

Todas as figuras e artefatos gerados pelo PA1 são salvos diretamente em `outputs/` (raiz do projeto):

- `outputs/synthetic_samples.png`: Grid com imagens e instâncias do dataset sintético (apenas Parte 0).
- `outputs/parte0_resultados.png`: Gráficos de perda, IoU/Dice semânticos e dispersão `mAP vs. Densidade de Objetos` (apenas Parte 0).
- `outputs/parte0_qualitativo.png`: Grid 4×N comparando Imagem Original, GT de Instâncias, Predições e Máscara Binária (Parte 0).
- `outputs/parte1_resultados.png`: Mesmo formato do `parte0_resultados.png`, sobre dados reais DSB2018 (apenas Parte 1).
- `outputs/parte1_qualitativo.png`: Mesmo formato do `parte0_qualitativo.png`, sobre dados reais DSB2018 (apenas Parte 1).

> **Regra de histórico:** Se uma imagem com o mesmo nome já existir na raiz de `outputs/`, ela é substituída pela mais recente. Caso deseje arquivar execuções anteriores, crie subpastas dentro de `outputs/` (ex: `outputs/historico/`); o pipeline não mexe nem lê arquivos dentro de subpastas.
