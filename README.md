# PA1 — Segmentação de Instâncias

**Disciplina:** Aprendizado Profundo | FGV CDIA  
**Professor:** Dario Oliveira | **Monitor:** Erick Brito  
**Alunos:** Bruno Ferreira & Elisa Soares  

Este repositório contém o código, experimentos e documentação do **Programming Assignment 1 (PA1)**. O objetivo é adaptar arquiteturas de segmentação semântica vistas em aula (em especial a **U-Net**) para realizarem **segmentação de instâncias**, sem o uso de detectores baseados em propostas de região (proibidos Mask R-CNN, YOLO, SAM, etc.).

---

## 🎯 Filosofia e Diretrizes do Projeto

Conforme estabelecido nas diretrizes internas da equipe:
1. **Foco no Aprendizado e Didática:** Toda decisão de arquitetura, função de perda e decodificação é implementada de forma limpa, explícita e justificada matematicamente e conceitualmente.
2. **Soluções Simples e Funcionais:** Preferência por pipelines claros e modulares em vez de complexidades desnecessárias.
3. **Mapeamento Explícito:** Toda e qualquer parte exigida no [PA1.pdf](PA1.pdf) (Partes 0 a 6) possui seu local no código, sua justificativa e seu comando de reprodução documentados nesta página.
4. **Transparência no Uso de IA:** Qualquer assistência por IA generativa é auditada e documentada no [AI_LOG.md](AI_LOG.md).
5. **Divisão em Dupla sem Bloqueios:** O detalhamento de quem faz o quê e como os módulos conversam está formalizado no [PLANO_DE_EXECUCAO.md](PLANO_DE_EXECUCAO.md).

---

## 📌 Decisões de Projeto

* **Dataset Selecionado:** Opção A — **Data Science Bowl 2018 / BBBC038v1** (exclusivamente `stage1_train`, com ~670 imagens de microscopia e máscaras individuais de núcleos celulares).
* **Trilha de Instâncias:** **Trilha A — Fronteiras e Watershed** (representação em 3 classes: Fundo, Interior do Núcleo e Fronteira de Contato, decodificada via Watershed com marcadores).
* **Ablações (Parte 3):** 
  * Eixo 1 (Recuperação de Resolução): U-Net clássica com Skips vs. Sem Skips / Unpooling.
  * Eixo 2 (Funções de Perda): Cross-Entropy Ponderada vs. Focal Loss variando $\gamma \in \{0, 1, 2, 5\}$.
* **Teste de Estresse (Parte 6):** Avaliação de robustez sob corrupções sintéticas (Blur Gaussiano, Ruído Gaussiano e Contraste) em 3 intensidades com curvas de degradação.

---

## 🗺️ Mapeamento das Partes do PA1 (Onde está, Como funciona e Como reproduzir)

Abaixo está o guia completo de rastreabilidade entre o enunciado [PA1.pdf](PA1.pdf) e a base de código:

### Parte 0 — Teste Unitário Sintético
* **Objetivo:** Garantir que o pipeline de dados, a U-Net e as métricas de instância funcionam perfeitamente em menos de 5 minutos, antes de usar dados reais.
* **Onde está:**
  * Gerador de Elipses: [`src/pa1/data/synthetic.py`](src/pa1/data/synthetic.py)
  * Arquitetura U-Net: [`src/pa1/models/unet.py`](src/pa1/models/unet.py)
  * Métricas (mAP@[.5:.95], Matching Hungarian): [`src/pa1/metrics/instance.py`](src/pa1/metrics/instance.py)
  * Notebook exploratório: [`pa1.ipynb`](pa1.ipynb)
* **Como foi implementado:** Imagens sintéticas 128×128 com 5 a 20 elipses sobrepostas com contraste e ruído. O modelo Mini U-Net aprende a separar fundo de objeto com perda BCE+Dice.
* **Como reproduzir:**
  ```bash
  uv run pa1 --epochs 10
  ```
  *(Artefatos visuais salvos em `outputs/parte0_resultados.png` e `outputs/parte0_qualitativo.png`)*
* **Status:** Concluído ✅

---

### Parte 1 — Baseline de Segmentação Semântica
* **Objetivo:** Treinar a U-Net para segmentação binária (fundo vs. núcleo) no DSB2018, extrair instâncias pelo método ingênuo (limiar + componentes conexos) e quantificar a falha de contagem e mAP em imagens com alta densidade de núcleos encostados.
* **Onde está:**
  * Dataset e Split DSB2018: [`src/pa1/data/dsb2018.py`](src/pa1/data/dsb2018.py) *(Responsável: Bruno)*
  * Extração Ingênua: [`src/pa1/postprocessing/connected_components.py`](src/pa1/postprocessing/connected_components.py)
  * Avaliação de Instâncias: [`src/pa1/metrics/instance.py`](src/pa1/metrics/instance.py)
* **Como reproduzir:**
  ```bash
  uv run pa1 --synthetic false --data-dir data/stage1_train --epochs 20
  ```
* **Status:** A Fazer (Fase 1 do Plano de Execução)

---

### Parte 2 — Uma Cabeça de Instâncias (Trilha A: Fronteiras e Watershed)
* **Objetivo:** Resolver a fusão indevida de núcleos encostados prevendo 3 classes (`0: fundo`, `1: interior`, `2: fronteira entre núcleos`) e decodificando com o algoritmo de Watershed onde os interiores erodidos servem de sementes (marcadores).
* **Onde está:**
  * Geração de Targets 3-Classes: [`src/pa1/data/targets.py`](src/pa1/data/targets.py) *(Responsável: Elisa)*
  * Pós-Processamento Watershed: [`src/pa1/postprocessing/watershed.py`](src/pa1/postprocessing/watershed.py) *(Responsável: Elisa)*
  * Perda Focal balanceada: [`src/pa1/losses/segmentation.py`](src/pa1/losses/segmentation.py)
* **Como reproduzir:**
  ```bash
  uv run pa1 --config configs/trilha_a_watershed.yaml
  ```
* **Status:** A Fazer (Fase 2 do Plano de Execução)

---

### Parte 3 — Ablações
* **Objetivo:** Isolar o efeito de componentes arquiteturais e perdas, treinando com 2 seeds cada e reportando média $\pm$ desvio padrão ($\mu \pm \sigma$).
  * *Eixo 1 (Resolução):* U-Net com Skip Connections vs. Decodificador sem skips / Unpooling.
  * *Eixo 2 (Perdas):* Cross-Entropy ponderada vs. Focal Loss com $\gamma \in \{0, 1, 2, 5\}$.
* **Onde está:**
  * Modelos de ablação: [`src/pa1/models/unet.py`](src/pa1/models/unet.py)
  * Variações da Focal Loss: [`src/pa1/losses/segmentation.py`](src/pa1/losses/segmentation.py)
* **Status:** A Fazer (Fase 4 do Plano de Execução — Responsável: Elisa)

---

### Parte 4 — Inferência em Mosaico
* **Objetivo:** Processar imagens grandes via janelas deslizantes (patches com overlap de acordo com o slide 83 da aula), evidenciar o artefato de quebra de instâncias na borda dos patches e propor algoritmo de fusão por IoU na sobreposição.
* **Onde está:**
  * Módulo de Mosaico e Costura: [`src/pa1/tiling/mosaic.py`](src/pa1/tiling/mosaic.py) *(Responsável: Bruno)*
* **Status:** A Fazer (Fase 3 do Plano de Execução)

---

### Parte 5 — Galeria de Falhas e Campo Receptivo Teórico
* **Objetivo:** Selecionar 5 casos severos de falha do modelo final com imagem, ground truth, predição e mapa intermediário de fronteiras. Calcular formalmente o Campo Receptivo Teórico do encoder (slides 35–38), compará-lo com o histograma de diâmetros dos núcleos e implementar uma correção testada (antes/depois).
* **Onde está:** Seção no notebook e documentação com deduções matemáticas.
* **Status:** A Fazer (Fase 5 do Plano de Execução — Trabalho Conjunto)

---

### Parte 6 — Teste de Estresse (Corrupções)
* **Objetivo:** Medir a robustez do modelo frente a perturbações visuais (Gaussian Blur, Ruído e Variações de Contraste) em 3 intensidades, traçando a curva de decaimento do mAP.
* **Onde está:**
  * Pipeline de Corrupções: [`src/pa1/stress/corruptions.py`](src/pa1/stress/corruptions.py) *(Responsável: Bruno)*
* **Status:** A Fazer (Fase 4 do Plano de Execução)

---

## 📁 Estrutura do Repositório

```
.
├── config.yaml             ← Central de parâmetros (épocas, lr, batch_size, paths)
├── PLANO_DE_EXECUCAO.md    ← Guia passo a passo de desenvolvimento para Bruno e Elisa
├── AI_LOG.md               ← Registro auditável de utilização de ferramentas de IA
├── diretizes.txt           ← Princípios de simplicidade, didática e clareza do time
├── PA1.pdf                 ← Enunciado oficial
├── pyproject.toml          ← Metadados do pacote e registro do CLI `pa1`
├── outputs/                ← Checkpoints (.pt), curvas de treino e diagnósticos visuais
├── src/
│   └── pa1/                ← Pacote modular reutilizável
│       ├── main.py         ← CLI central
│       ├── config.py       ← Dataclasses tipadas para parsing do config.yaml
│       ├── data/           ← Synthetic ellipses e loader do DSB2018
│       ├── models/         ← U-Net e blocos convolucionais
│       ├── losses/         ← BCEDice, Focal Loss e variantes
│       ├── metrics/        ← IoU, mAP@[.5:.95], Hungarian/Greedy matching
│       ├── postprocessing/ ← Componentes conexos e Watershed
│       └── utils/          ← Seeds, detecção de hardware e plotters
├── pa1.ipynb               ← Notebook de exploração
└── inferencia.ipynb        ← [Em dev] Entregável: avaliação em imagem avulsa
```

---

## ⚙️ 1. Configuração do Ambiente com `uv`

O projeto utiliza [`uv`](https://docs.astral.sh/uv/) para gerenciar dependências de forma rápida e reprodutível:

```bash
# Sincroniza e cria o ambiente virtual isolado (.venv)
uv sync
```

---

## 📥 2. Download do Dataset (DSB2018 `stage1_train`)

Conforme a Opção A, usamos exclusivamente o conjunto `stage1_train`.

### Opção A: Via Kaggle API (Mais Rápido)
```bash
kaggle competitions download -c data-science-bowl-2018 -f stage1_train.zip
mkdir -p data/stage1_train
unzip stage1_train.zip -d data/stage1_train/
rm stage1_train.zip
```

### Opção B: Download Direto (Broad Institute BBBC038)
* Acesse [Broad Institute BBBC038](https://bbbc.broadinstitute.org/BBBC038)
* Baixe o arquivo de treino e descompacte em `data/stage1_train/`.

*(O diretório `data/` está no `.gitignore` para não versionar arquivos pesados).*

---

## 🚀 3. Como Operar a CLI (`pa1`)

A CLI aceita os parâmetros do [config.yaml](config.yaml) ou sobrescritas via terminal:

```bash
# Rodar treinamento padrão:
uv run pa1

# Treinar por 10 épocas com learning rate customizado:
uv run pa1 --epochs 10 --lr 5e-4

# Apenas avaliar um checkpoint salvo anteriormente:
uv run pa1 --eval-only --checkpoint outputs/model.pt
```

---

## 👥 4. Colaboração da Dupla

Para detalhes de interfaces, branches do Git e cronograma de integração entre **Bruno** e **Elisa**, consulte o documento completo:  
👉 **[PLANO_DE_EXECUCAO.md](PLANO_DE_EXECUCAO.md)**
