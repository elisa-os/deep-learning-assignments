# Deep Learning Assignments — FGV CDIA

**Disciplina:** Aprendizado Profundo | FGV CDIA  
**Professor:** Dario Oliveira | **Monitor:** Erick Brito  
**Alunos:** Bruno Ferreira & Elisa Soares  

Este repositório centraliza os trabalhos práticos (Programming Assignments) da disciplina de Aprendizado Profundo. Cada assignment é encapsulado em sua respectiva pasta (`pa1/`, e futuramente `pa2/`, etc.), mantendo código, configurações, dados e artefatos organizados de forma independente.

---

## Entregáveis (o que o enunciado pede)

### Ambiente

O projeto usa [`uv`](https://docs.astral.sh/uv/) para gerenciar dependências e comandos. Sincronize o ambiente com:

```bash
uv sync
```

Isso cria o ambiente virtual isolado (`.venv`) e instala todas as dependências listadas em `pyproject.toml`, incluindo PyTorch, scikit-image, scipy, pandas, matplotlib e albumentations. O pacote `pa1` está configurado via `module-root` no `pyproject.toml`; não é necessário uma pasta `src/`.

Comandos disponíveis no CLI:

```bash
uv run pa1            # pipeline principal (treino/avaliação por parte)
uv run pa1-ablation   # ablações do Eixo 1 e Eixo 2 (Parte 3)
uv run pa1-mosaic     # inferência em mosaico (Parte 4)
uv run pa1-stress     # teste de estresse (Parte 6)
```

### Download dos dados

O dataset usado é a **Opção A — Data Science Bowl 2018 / BBBC038v1** (`stage1_train`), com ~670 imagens de microscopia e máscaras individuais de núcleos. O split treino/validação/teste já está estratificado por modalidade no loader (`pa1/data/dsb2018.py`).

**Como baixar:**

Opção 1 — site oficial do Broad Institute (sem conta):
```
https://bbbc.broadinstitute.org/BBBC038
```

Opção 2 — Kaggle:
```bash
kaggle competitions download -c data-science-bowl-2018
```

Após extrair, coloque a pasta `stage1_train/` em:
```
pa1/data/stage1_train/
```

Ou seja, a estrutura esperada é:
```
pa1/data/stage1_train/
  00071198d0.../
    images/00071198d0...png
    masks/*.png     (uma máscara por núcleo, em preto e branco)
  003cee893.../
    ...
```

O pipeline detecta automaticamente a estrutura acima; não é necessário renomear ou reorganizar.

### Um comando que treina

Treina a solução principal (Parte 2 — Trilha A: 3 classes + Watershed) com a configuração padrão de `pa1/config.yaml`:

```bash
cd deep-learning-assignments   # ou trabalhe a partir da raiz do repositório
uv run pa1 2
```

Isso treina a U-Net com saída de 3 classes (fundo / interior / fronteira), Focal Loss multiclasse ponderada, e decodifica instâncias via Watershed. O treinamento usa os hiperparâmetros definidos em `pa1/config.yaml` → `parte2` (30 épocas, lr=1e-3, batch_size=8).

Sobrescreva epochs se precisar de uma execução mais curta (ex: validação rápida)

```bash
uv run pa1 2 --epochs 5
```

Outras partes também podem ser treinadas com o mesmo CLI:

```bash
uv run pa1 0     # Parte 0 — teste sintético com elipses
uv run pa1 1     # Parte 1 — baseline binária no DSB2018
uv run pa1 5     # Parte 5 — mesma arquitetura com dilate_bottleneck=4
```

### Um comando que avalia

Avalia o modelo já treinado sem retreinar, usando o checkpoint salvo:

```bash
uv run pa1 2 --eval-only --checkpoint pa1/outputs/checkpoints/parte2_baseline_unet.pt
```

Isso carrega os pesos, roda a inferência em validação e teste, e gera:
- `pa1/outputs/metrics/parte2_baseline_results.json` — métricas médias (mAP, IoU, Dice, erro de contagem)
- `pa1/outputs/metrics/parte2_per_image_instance_metrics.csv` — métricas por imagem
- `pa1/outputs/parte2_resultados.png` — curvas de treino + dispersão mAP × densidade
- `pa1/outputs/parte2_qualitativo.png` — grid qualitativo com as 4 piores imagens

Para avaliar apenas uma parte específica com o checkpoint dela:

```bash
uv run pa1 1 --eval-only --checkpoint pa1/outputs/checkpoints/parte1_baseline_unet.pt
uv run pa1 5 --eval-only --checkpoint pa1/outputs/checkpoints/parte5_baseline_unet.pt
uv run pa1 0 --eval-only --checkpoint pa1/outputs/checkpoints/parte0_baseline_unet.pt
```

### Arquivos entregues junto com o repositório

Além do README.md, o repositório entrega:

- **`pa1/AI_LOG.md`** — log de uso de IA neste assignment, conforme exigido pela política de uso de IA do enunciado (seção 5 do PA1.pdf). Descreve episódios em que IA foi usada e como os problemas foram resolvidos.
- **`pa1/inferencia.ipynb`** — notebook de inferência: recebe o caminho de uma imagem qualquer e devolve a máscara de instâncias colorida e a contagem, rodando sem retreinar. Usa o checkpoint da Parte 2 (`parte2_baseline_unet.pt`).
- **`pa1/outputs/checkpoints/parte2_baseline_unet.pt`** — pesos do modelo final treinado (checkpoint). É o artefato que o `inferencia.ipynb` e o comando de avaliação usam.

---

## Detalhamento por Parte

Cada parte abaixo lista onde o código está, como reproduzir, o que gera e o status.

---

## Mapeamento das Partes

Cada parte abaixo lista:
- **Onde** o código está (arquivo(s) no repositório);
- **Como reproduzir** (comando via `uv run`);
- **O que gera** (saídas e documentos associados);
- **Status** (treinado? avaliado? notebook executável?);

### Parte 0 — Teste unitário sintético

Parte 0 valida o pipeline completo antes de tocar em dados reais, usando um dataset sintético de elipses gerado por `pa1/data/synthetic.py`. É um teste de sanidade: treina em minutos e mostra que as métricas (IoU, Dice, mAP, erro de contagem) funcionam.

- **Onde:** `pa1/data/synthetic.py` (dataset), `pa1/main.py` (treino), configuração `parte0` em `pa1/config.yaml`
- **Como reproduzir:** `uv run pa1 0` ou `uv run pa1 0 --epochs 5` (versão rápida)
- **Saídas:**
  - `pa1/outputs/parte0_synthetic_samples.png` — grid 2×4 com imagens sintéticas + máscaras de instâncias
  - `pa1/outputs/parte0_resultados.png` — curvas de loss/IoU/Dice + dispersão mAP × densidade
  - `pa1/outputs/parte0_qualitativo.png` — grid 4×4 comparativo (imagem/GT/pred/binário)
  - `pa1/outputs/checkpoints/parte0_baseline_unet.pt` — checkpoint do modelo treinado
  - `pa1/outputs/metrics/parte0_baseline_results.json` — métricas médias
  - `pa1/outputs/metrics/parte0_per_image_instance_metrics.csv` — métricas por imagem
- **Status:** Treinado e avaliado

### Parte 1 — Baseline de segmentação semântica

Treina U-Net binária sobre o DSB2018 e extrai instâncias pelo método ingênuo: limiar + componentes conexos (`pa1/postprocessing/connected_components.py`). Reporta IoU, Dice e mAP de instâncias, além do gráfico de mAP vs. densidade (a tendência de degradação com mais núcleos por imagem deve ficar visível — é o "quantifiquem o fracasso" do enunciado).

- **Onde:** `pa1/data/dsb2018.py` (dataset + estratificação por modalidade), `pa1/postprocessing/connected_components.py` (extração de instâncias binárias), `pa1/main.py` (treino/avaliação), configuração `parte1` em `pa1/config.yaml`
- **Como reproduzir:** `uv run pa1 1` ou `uv run pa1 1 --epochs 5` (versão rápida)
- **Saídas:**
  - `pa1/outputs/checkpoints/parte1_baseline_unet.pt` — checkpoint do modelo binário treinado
  - `pa1/outputs/metrics/parte1_baseline_results.json` — métricas médias de val e test (IoU, Dice, mAP, count_error)
  - `pa1/outputs/metrics/parte1_per_image_instance_metrics.csv` — métricas completas por imagem (colunas descritas na seção "Métricas por imagem" abaixo)
  - `pa1/outputs/parte1_resultados.png` — curvas de loss/IoU/Dice + dispersão mAP × densidade
  - `pa1/outputs/parte1_qualitativo.png` — grid 4×4 comparativo sobre dados reais (imagem/GT/pred/binário)
- **Status:** Treinado e avaliado

### Parte 2 — Trilha A: Fronteiras + Watershed (3 classes)

A solução principal da dupla. Mantém o encoder-decoder da Parte 1 e muda o que a rede prevê: 3 classes (fundo / interior / fronteira entre instâncias). O pós-processamento decodifica instâncias com Watershed usando os interiores como marcadores.

**Como gerar o rótulo de fronteira a partir das máscaras individuais:**
`pa1/data/targets.py` → função `generate_3class_target(mask_instances)`:
1. Dilata cada máscara individual por 2px (espessura da fronteira);
2. Faz a diferença entre a versão dilatada e a versão original → região de fronteira;
3. Atribui: 0=fundo, 1=interior, 2=fronteira.

**Como pesar a classe fronteira (minoritária):**
`pa1/losses/segmentation.py` → `MulticlassFocalLoss` recebe pesos de classe `[1.0, 1.0, 8.0]` (fundo=1, interior=1, fronteira=8), configurados em `pa1/config.yaml` → `parte2.loss.weights`.

- **Onde:** `pa1/data/targets.py` (target 3 classes), `pa1/losses/segmentation.py` (Focal Loss multiclasse com pesos), `pa1/postprocessing/watershed.py` (decodificação Watershed com marcadores), `pa1/main.py` (treino/avaliação), configuração `parte2` em `pa1/config.yaml`
- **Como reproduzir:** `uv run pa1 2` ou `uv run pa1 2 --epochs 5` (versão rápida)
- **Saídas:**
  - `pa1/outputs/checkpoints/parte2_baseline_unet.pt` — checkpoint do modelo 3 classes treinado
  - `pa1/outputs/metrics/parte2_baseline_results.json` — métricas médias de val e test
  - `pa1/outputs/metrics/parte2_per_image_instance_metrics.csv` — métricas por imagem (Watershed)
  - `pa1/outputs/parte2_resultados.png` — curvas de loss/IoU/Dice + dispersão mAP × densidade
  - `pa1/outputs/parte2_qualitativo.png` — grid 4×4 comparativo com decodificação Watershed
- **Status:** Treinado e avaliado

### Parte 3 — Ablações (Eixo 1 e Eixo 2)

O enunciado pede ablações em 2 eixos com 2 seeds cada, reportando média ± desvio. A dupla implementou:

**Eixo 1 — Como recuperar resolução:** compara U-Net completa (com skip connections) vs. decodificador sem skips (`U-Net decoder-only`). Skip connections são o mecanismo que a dupla escolheu usar na solução principal; o Eixo 1 quantifica o que se perde sem eles.

**Eixo 2 — Função de perda:** varia γ ∈ {0, 1, 2, 5} da Focal Loss, com 2 seeds cada (42 e 123). O γ=0 é equivalente a Cross-Entropy padrão, γ=1 já introduce o efeito focal, γ=2 e γ=5 penalizam mais os exemplos fáceis.

- **Onde:** `pa1/ablation.py` (script automatizado que roda todas as configurações), `pa1/models/unet.py` (decoder sem skips, `use_skips=False`), `pa1/losses/segmentation.py` (Focal Loss com γ variável), configuração em `pa1/config.yaml` (parâmetros de cada run)
- **Como reproduzir:** `uv run pa1-ablation`
- **Saídas (em `pa1/outputs/parte3_ablation/`):**
  - `pa1/outputs/parte3_ablation/checkpoints/` — 12 checkpoints (4 configurações × 2 seeds + 2 seeds do Eixo 1 sem skips):
    - `eixo1_com_skips_seed42.pt`, `eixo1_com_skips_seed123.pt`
    - `eixo1_sem_skips_seed42.pt`, `eixo1_sem_skips_seed123.pt`
    - `eixo2_gamma0_seed42.pt`, `eixo2_gamma0_seed123.pt`
    - `eixo2_gamma1_seed42.pt`, `eixo2_gamma1_seed123.pt`
    - `eixo2_gamma2_seed42.pt`, `eixo2_gamma2_seed123.pt`
    - `eixo2_gamma5_seed42.pt`, `eixo2_gamma5_seed123.pt`
  - `pa1/outputs/parte3_ablation/metrics/ablation_results.json` — resultados brutos de todas as configurações
  - `pa1/outputs/parte3_ablation/metrics/ablation_summary.csv` — tabela agregada (μ ± σ) por configuração
  - `outputs/parte3_ablation_eixo1_map.png` — gráfico de barras: mAP com vs. sem skips
  - `outputs/parte3_ablation_eixo2_map.png` — gráfico de barras: mAP por γ
  - `outputs/parte3_ablation_eixo2_count.png` — gráfico de barras: erro de contagem por γ
- **Status:** Ablações executadas (12 run total, 2 seeds por configuração)

### Parte 4 — Inferência em mosaico

O enunciado descreve o problema: inferência em tiles com sobreposição funciona para segmentação semântica, mas para instâncias objetos cortados na fronteira entre tiles são fragmentados. A dupla monta um mosaico sintético a partir de imagens do DSB2018, roda inferência em tiles sobrepostos (128×128 com stride=64), e implementa fusão de instâncias via grafo de equivalência construído com `scipy.sparse.csgraph`.

A correção proposta: quando duas instâncias preditas em tiles adjacentes têm sobreposição na região de sobreposição dos tiles ( IoU > 0 ) ou centros próximos (|Δx|+|Δy| < threshold), elas são fundidas no grafo como mesmo ID. O mAP é medido antes e depois da correção.

- **Onde:** `pa1/tiling/mosaic.py` (montagem do mosaico, inferência sliding window, construção do grafo de equivalência e fusão), `pa1/tiling/__init__.py`
- **Como reproduzir:** `uv run pa1-mosaic` (requer checkpoint da Parte 2: `pa1/outputs/checkpoints/parte2_baseline_unet.pt`)
- **Saídas (em `pa1/outputs/`):**
  - `outputs/parte4_mosaic_fusion.png` — mosaico 512×512 com 4 quadrantes (imagem/GT/sem fusão/com fusão) + grade amarela mostrando as fronteiras dos tiles
  - `outputs/parte4_map_comparison.png` — gráfico de barras comparando mAP sem fusão vs. com fusão
  - `outputs/parte4_zoom_fusion.png` — zoom 112×112 na interseção central mostrando células cortadas sem fusão e consertadas com fusão
- **Status:** Executado (mosaico criado, inferência rodada, fusão implementada, mAP comparado)

### Parte 5 — Galeria de falhas e Campo Receptivo

O enunciado exige: 5 imagens onde o modelo final erra feio, cada uma com figura (imagem/GT/predição/mapa intermediário) + diagnóstico escrito, **calculando o campo receptivo teórico do encoder** e comparando com a distribuição de tamanhos dos objetos do dataset. Se usaram atrous convolution, mostrar o campo receptivo com e sem ela. E fazer uma correção: implementar a mudança que o diagnóstico sugere, mostrar antes/depois.

A dupla calculou o campo receptivo teórico da U-Net (sem dilatação) = **140px**, e mostra que a maioria dos núcleos do DSB2018 tem diâmetro < 140px, mas há aglomerados gigantes que excedem esse limite (o pixel central de um aglomerado grande nunca enxerga as duas bordas). A intervenção: atrous convolution no gargalo com `dilation=4` expande o RF para **332px** sem adicionar parâmetros.

- **Onde:** `pa1/parte5_falhas_rf.ipynb` (notebook completo com dedução do RF passo a passo, histograma, galeria de 5 piores falhas e comparação lado a lado padrão vs. dilatado), `pa1/models/unet.py` (parâmetro `dilate_bottleneck` que controla atrous convolution no bottleneck)
- **Como reproduzir (treino da Parte 5):** `uv run pa1 5` (treina o modelo com `dilation=4` e gera os métricas e gráficos dela)
- **Como reproduzir (notebook de análise — sem retreinar):** abrir `pa1/parte5_falhas_rf.ipynb` no Jupyter e executar todas as células (as células 1-6 já vêm com outputs gravados; a célula 6 referencia os checkpoints `parte2_baseline_unet.pt` e `parte5_baseline_unet.pt` que já existem em `pa1/outputs/checkpoints/`)
- **Saídas:**
  - `pa1/parte5_falhas_rf.ipynb` — notebook executável com tudo o que o enunciado pede na Parte 5:
    - Célula 2: histograma de diâmetros dos núcleos do DSB2018 vs. linha vertical em RF=140px
    - Célula 3: seleção das 5 piores imagens pelo mAP (idx=82, 84, 75, 90, 88)
    - Célula 4: galeria das 5 falhas, cada uma com: imagem RGB, GT de instâncias, probabilidade de fronteira (heatmap magma), watershed previsto
    - Célula 5: instanciação do modelo dilatado com `dilation=4` e impressão da arquitetura do bottleneck
    - Célula 6: comparação lado a lado do modelo padrão (RF=140px) vs. modelo dilatado (RF=332px) na imagem idx=90, com contagem de instâncias em cada Watershed
  - `pa1/outputs/checkpoints/parte5_baseline_unet.pt` — checkpoint do modelo com dilation=4 (treinado na Parte 5)
  - `pa1/outputs/metrics/parte5_baseline_results.json` — métricas médias do modelo dilatado
  - `pa1/outputs/metrics/parte5_per_image_instance_metrics.csv` — métricas por imagem do modelo dilatado
  - `pa1/outputs/parte5_resultados.png` — curvas de loss/IoU/Dice do treino com dilation=4
  - `pa1/outputs/parte5_qualitativo.png` — grid qualitativo com decodificação Watershed do modelo dilatado
- **Status:** Notebook executável com outputs gravados; checkpoint da Parte 5 treinado e avaliado

### Parte 6 — Teste de estresse (corrupções sintéticas)

O enunciado oferece 3 opções para o teste de estresse. A dupla escolheu **corrupções** (blur, ruído, contraste) em 3 intensidades, gerando curva de degradação do mAP. Avalia o modelo congelado da Parte 2 sem retreinar.

As corrupções implementadas em `pa1/stress/corruptions.py`:
- **Gaussian Blur:** σ ∈ {0, 1, 2, 3.5} (4 níveis)
- **Gaussian Noise:** std ∈ {0, 0.05, 0.15, 0.30} (4 níveis)
- **Contraste:** α ∈ {1.0, 0.7, 0.4, 0.2} (4 níveis)

Cada corrupção é aplicada à imagem de entrada antes da inferência, e o mAP é calculado sob a versão corrompida.

- **Onde:** `pa1/stress/corruptions.py` (funções de corrupção + avaliação + plotagem), `pa1/stress/__init__.py`
- **Como reproduzir:** `uv run pa1-stress` (requer checkpoint da Parte 2: `pa1/outputs/checkpoints/parte2_baseline_unet.pt`)
- **Saídas:**
  - `outputs/parte6_stress_test.png` — 3 subplots com curvas de degradação mAP vs. intensidade para Blur, Noise e Contrast, com linha tracejada cinza mostrando baseline (corrupção σ=0 / std=0 / α=1.0)
- **Status:** Executado (modelo congelado avaliado sob 3 corrupções × 4 intensidades)

---

## Métricas por imagem — `parte*_per_image_instance_metrics.csv`

Cada parte gera um CSV com uma linha por imagem avaliada, salvo em `pa1/outputs/metrics/`. As colunas são:

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `idx` | int | Índice sequencial da imagem na ordem de iteração do loader |
| `n_gt` | int | Número de instâncias GT na imagem |
| `n_pred` | int | Número de instâncias previstas pelo modelo (pós-decodificação) |
| `count_error` | int | `abs(n_pred - n_gt)` |
| `iou_sem` | float | IoU semântico binário para a imagem |
| `dice_sem` | float | Dice semântico binário para a imagem |
| `mAP` | float | mAP@[0.50:0.95] da imagem (média sobre os 10 limiares) |

Além disso, para cada limiar T em {50, 55, 60, 65, 70, 75, 80, 85, 90, 95} (passo 0.05):

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `tp_T` | int | Verdadeiros positivos (instâncias previstas com IoU ≥ T/100) |
| `fp_T` | int | Falsos positivos |
| `fn_T` | int | Falsos negativos |
| `ap_T` | float | AP por limiar (usado no cálculo do mAP) |

Exemplo de linhas:
```
idx,n_gt,n_pred,count_error,iou_sem,dice_sem,mAP,tp_50,fp_50,fn_50,ap_50,tp_55,fp_55,fn_55,ap_55,...
0,5,3,2,0.71,0.83,0.62,3,0,2,0.60,3,0,2,0.60,...
1,2,2,0,0.88,0.93,0.85,2,0,0,0.80,2,0,0,0.80,...
```

**Como consultar rápido:**

```bash
cd pa1

# 5 piores imagens pelo mAP (útil para galeria de falhas da Parte 5):
uv run python -c "import pandas as pd; df=pd.read_csv('outputs/metrics/parte2_per_image_instance_metrics.csv'); print(df.sort_values('mAP').head(5))"

# 5 melhores imagens:
uv run python -c "import pandas as pd; df=pd.read_csv('outputs/metrics/parte2_per_image_instance_metrics.csv'); print(df.sort_values('mAP', ascending=False).head(5))"

# Distribuição de erro de contagem:
uv run python -c "import pandas as pd; df=pd.read_csv('outputs/metrics/parte2_per_image_instance_metrics.csv'); print(df['count_error'].value_counts().sort_index())"
```

O exportador é implementado em `pa1/utils/export.py` (`PerImageMetricsWriter`). Recebe um `filename` no momento da escrita, permitindo manter arquivos separados por parte sem mudar a lógica de avaliação:

```python
from pa1.utils import PerImageMetricsWriter
writer = PerImageMetricsWriter(Path("outputs"))
# ... durante avaliação, writer.add(...) por imagem ...
csv_path = writer.write("parte2_per_image_instance_metrics.csv")
```

---

## Configuração via `pa1/config.yaml`

O comportamento do pipeline é controlado por `pa1/config.yaml`, que define seções separadas por parte (`parte0`, `parte1`, `parte2`, `parte5`). Cada seção especifica: dados sintéticos ou reais, diretório dos dados, canais de entrada/saída, épocas, learning rate, checkpoint e modo de avaliação.

Exemplo da seção `parte2`:
```yaml
parte2:
  synthetic: false
  data_dir: pa1/data/stage1_train
  n_samples: 100
  batch_size: 8
  num_workers: 2
  in_channels: 3         # RGB (DSB2018)
  out_channels: 3       # 3 = Trilha A (fundo/interior/fronteira)
  epochs: 30
  lr: 1.0e-3
  checkpoint: null
  eval_only: false
```

Overrides via linha de comando (ex: `--epochs 5 --lr 1e-3`) sobrescrevem os valores do YAML para aquela execução específica.

---

## Notas

- **Regra de histórico:** Se uma imagem com o mesmo nome já existir em `pa1/outputs/`, ela é substituída pela mais recente. Para arquivar execuções anteriores, crie subpastas dentro de `outputs/` (ex: `pa1/outputs/historico/`); o pipeline não lê nem modifica arquivos dentro de subpastas.
- **Checkpoint de avaliação única:** Use `--eval-only --checkpoint <caminho>` para avaliar sem retreinar. O CSV de métricas por imagem também é gerado nesse modo.
- **API de exportação:** A classe `PerImageMetricsWriter` está em `pa1/utils/export.py` e pode ser reutilizada em partes subsequentes passando um filename diferente a cada parte.
- **Matching de instâncias:** A regra de matching usada para o mAP é **Hungarian (assignment ótimo)** implementada em `pa1/metrics/instance.py`. Diferente do matching guloso por IoU decrescente, o Hungarian garante o emparelhamento global ótimo entre instâncias previstas e GT, o que pode resultar em mAP ligeiramente diferente do que se obtia com matching guloso. A escolha está documentada aqui porque o enunciado pede explicitamente que a regra de matching fique explícita.

---

## Arquivos do repositório

```
deep-learning-assignments/
├── pyproject.toml              # Dependências e comandos uv
├── uv.lock                     # Lockfile determinístico
├── .gitignore
├── README.md                   # Este arquivo
│
└── pa1/
    ├── PA1.pdf                 # Enunciado oficial do PA1
    ├── AI_LOG.md               # Log de uso de IA (obrigatório pela política do enunciado)
    ├── PLANO_DE_EXECUCAO.md    # Plano de execução sequencial detalhado pela dupla
    ├── config.yaml             # Configuração por parte (parte0, parte1, parte2, parte5)
    ├── config.py               # Carregador de configuração (load_config())
    ├── main.py                 # Pipeline principal (treino/avaliação por parte)
    ├── ablation.py             # Script automatizado de ablações (Parte 3)
    │
    ├── data/
    │   ├── __init__.py
    │   ├── synthetic.py        # Dataset sintético de elipses (Parte 0)
    │   ├── dsb2018.py          # Dataset real DSB2018 com estratificação por modalidade
    │   └── targets.py          # Geração do target 3 classes (fundo/interior/fronteira)
    │
    ├── models/
    │   ├── __init__.py
    │   ├── unet.py             # Arquitetura UNet (encoder-decoder + skip connections)
    │   └── heads.py            # SegmentationHead e BoundaryAwareHead
    │
    ├── losses/
    │   ├── __init__.py
    │   └── segmentation.py     # BCEDiceLoss, FocalLoss, MulticlassFocalLoss, MulticlassDiceLoss
    │
    ├── metrics/
    │   ├── __init__.py
    │   └── instance.py         # Implementação do matching Hungarian + mAP@[0.50:0.95] + count_error
    │
    ├── postprocessing/
    │   ├── __init__.py
    │   ├── connected_components.py  # Extração de instâncias binárias (Parte 1)
    │   └── watershed.py             # Decodificação Watershed com marcadores (Parte 2)
    │
    ├── tiling/
    │   ├── __init__.py
    │   └── mosaic.py           # Mosaico, sliding windows, grafo de equivalência e fusão (Parte 4)
    │
    ├── stress/
    │   ├── __init__.py
    │   └── corruptions.py      # Corrupções (blur, noise, contraste) + avaliação (Parte 6)
    │
    ├── utils/
    │   ├── __init__.py
    │   ├── device.py           # get_device()
    │   ├── seed.py             # set_seed()
    │   ├── export.py           # PerImageMetricsWriter (exportação de métricas por imagem)
    │   └── visualize.py        # plot_synthetic_samples, plot_training_results, plot_qualitative_results
    │
    ├── inferencia.ipynb        # Notebook de inferência: aceita imagem qualquer, devolve máscara colorida + contagem
    └── parte5_falhas_rf.ipynb # Notebook completo da Parte 5: RF teórico, histograma, galeria de 5 falhas, comparação RF vs. RF dilatado
```

---

Fim do README.md.
