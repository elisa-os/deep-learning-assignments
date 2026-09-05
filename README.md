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

## 🚀 Como Rodar a Parte 1 do PA1 (Baseline de Segmentação Semântica)

A **Parte 1** estabelece o baseline binário: treina uma U-Net para segmentação semântica do DSB2018 e quantifica o fracasso da abordagem ingênua (componentes conexos) em imagens com alta densidade de núcleos.

### Pré-requisitos

O projeto usa [`uv`](https://docs.astral.sh/uv/) para gerenciar dependências e ambientes virtuais.

```bash
# Instalação do uv (se ainda não tiver)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Ou via pip
pip install uv

# Sincroniza dependências e cria o ambiente virtual isolado (.venv)
uv sync
```

### Dados

Os dados do DSB2018 (`stage1_train`) precisam estar em `pa1/data/stage1_train/`. 
Se ainda não estão disponíveis, baixe e extraia o dataset seguindo as instruções do enunciado do PA1.

### Execução

```bash
# 1. Treinar a U-Net binária (baseline) com configuração padrão:
uv run python -m pa1.main --epochs 20 --lr 1e-3

# 2. Treinar com parâmetros personalizados (útil para testes rápidos):
uv run python -m pa1.main --epochs 5 --lr 1e-3 --batch-size 4

# 3. Avaliar apenas (sem retreinar) usando checkpoint salvo:
uv run python -m pa1.main --eval-only --checkpoint pa1/outputs/checkpoint.pt

# 4. Rodar com dados sintéticos (elipses) para validação rápida do pipeline:
uv run python -m pa1.main --synthetic --epochs 5
```

### Saídas da Parte 1

Após o treino, as seguintes figuras são geradas em `pa1/outputs/`:

| Arquivo | Descrição |
|---------|-----------|
| `parte0_resultados.png` | Gráficos de loss, métricas semânticas (IoU/Dice) e dispersão mAP vs. densidade de objetos |
| `parte0_qualitativo.png` | Grid 4×N comparando Imagem Original, GT de Instâncias, Predições e Máscara Binária |

> **Regra de histórico:** Se uma imagem com o mesmo nome já existir na raiz de `pa1/outputs/`, ela é substituída pela mais recente. Caso deseje arquivar execuções anteriores, crie subpastas dentro de `pa1/outputs/` (ex: `pa1/outputs/historico/`); o pipeline não mexe nem lê arquivos dentro de subpastas.

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

Todas as figuras e artefatos gerados pelo PA1 são salvos diretamente em `pa1/outputs/`:

- `pa1/outputs/synthetic_samples.png`: Grid com imagens e instâncias do dataset sintético.
- `pa1/outputs/parte0_resultados.png`: Gráficos de perda, IoU/Dice semânticos e dispersão `mAP vs. Densidade de Objetos`.
- `pa1/outputs/parte0_qualitativo.png`: Grid 4×N comparando Imagem Original, GT de Instâncias, Predições e Máscara Binária.

> **Regra de histórico:** Se uma imagem com o mesmo nome já existir na raiz de `pa1/outputs/`, ela é substituída pela mais recente. Caso deseje arquivar execuções anteriores, crie subpastas dentro de `pa1/outputs/` (ex: `pa1/outputs/historico/`); o pipeline não mexe nem lê arquivos dentro de subpastas.
