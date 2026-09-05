# Plano de Execução PA1 — Segmentação de Instâncias

**Disciplina:** Aprendizado Profundo | FGV CDIA  
**Alunos:** Bruno Ferreira & Elisa Soares  
**Decisões de Projeto:**
- **Dataset:** Opção A — DSB2018 (`stage1_train`)
- **Abordagem de Instância:** Trilha A — Fronteiras e Watershed
- **Ablações:** 
  - Eixo 1 (Recuperação de Resolução: U-Net com Skips vs. SegNet / Sem Skips)
  - Eixo 2 (Funções de Perda: Cross-Entropy Ponderada vs. Focal Loss com $\gamma \in \{0, 1, 2, 5\}$)
- **Teste de Estresse:** Corrupções Sintéticas (Gaussian Blur, Gaussian Noise e Contraste) em 3 intensidades
- **Regra de Matching para mAP:** Hungarian Matching (ou Guloso por IoU decrescente) avaliado em $\text{IoU} \in [0.50 : 0.05 : 0.95]$

---

## 1. Visão Geral da Arquitetura do Repositório

O projeto segue uma estrutura modular em que cada componente possui responsabilidade única e interfaces bem definidas. Abaixo, o status atual de cada módulo no repositório:

```
src/pa1/
├── data/
│   ├── stage1_train/           # [✅ Baixado] 670 pastas de imagens e máscaras do DSB2018 prontas para uso
│   ├── synthetic.py            # [✅ Concluído] Geração de dataset analítico de elipses (Parte 0)
│   ├── dsb2018.py              # [⏳ A Fazer] Ingestão, pré-processamento e split estratificado do DSB2018 (Passo 1)
│   └── targets.py              # [⏳ A Fazer] Geração do mapa de 3 classes: Fundo / Interior / Fronteira (Passo 2)
├── models/
│   ├── unet.py                 # [🟡 Parcial] U-Net com skips implementada [✅]; variante sem skips [⏳ A Fazer]
│   └── heads.py                # [⏳ A Fazer] Cabeça de saída multiclasse (3 classes) para Trilha A
├── losses/
│   └── segmentation.py         # [🟡 Parcial] BCEDiceLoss e Focal Loss binária [✅]; Focal Loss multiclasse [⏳ A Fazer]
├── postprocessing/
│   ├── connected_components.py # [✅ Concluído] Pós-processamento ingênuo por componentes conexos (Passo 3)
│   └── watershed.py            # [⏳ A Fazer] Decodificação watershed com marcadores de interior e bacias (Passo 3)
├── metrics/
│   └── instance.py             # [✅ Concluído] Matching Hungarian/Greedy e cálculo de mAP@[0.50:0.95] e erro de contagem
├── tiling/
│   └── mosaic.py               # [⏳ A Fazer] Inferência em janelas deslizantes e fusão de instâncias na sobreposição (Passo 7)
└── stress/
    └── corruptions.py          # [⏳ A Fazer] Aplicação de perturbações sintéticas (blur, ruído, contraste) em 3 níveis (Passo 9)
```

---

## 2. Roteiro de Implementação Sequencial (Status por Tópico: O Que, Onde e Como Fazer)

As etapas a seguir estão dispostas na ordem estrita de execução técnica, indicando detalhadamente o que já está implementado e o que ainda precisa ser feito em cada tópico.

---

### Passo 0: Teste Unitário Sintético (Parte 0 do PA1) — `[✅ Concluído]`
*Objetivo: Validar toda a cadeia de treinamento e avaliação com dados sintéticos antes de rodar nos dados reais.*

- **O que fazer:**
  - `[✅ Concluído]` Gerar um conjunto de dados analítico rápido com máscaras de instância exatas (`SyntheticEllipseDataset` e `make_synthetic_loader`).
  - `[✅ Concluído]` Treinar uma U-Net pequena demonstrando convergência em menos de 5 minutos (loop funcional em `main.py`).
  - `[✅ Concluído]` Validar o pipeline de treino, cálculo de perda e métrica básica de instância (IoU, Dice, mAP e erro de contagem validados no terminal e nas células 0 a 19 de `pa1.ipynb`).
- **Onde fazer:**
  - `src/pa1/data/synthetic.py` `[✅ Implementado]`
  - `src/pa1/models/unet.py` `[✅ Implementado]`
  - `src/pa1/main.py` `[✅ Implementado]`
  - `pa1.ipynb` `[✅ Implementado (células 0-19)]`
- **Como fazer:**
  1. Criar imagens $128 \times 128$ contendo entre 5 e 20 elipses aleatórias com sobreposição proposital, níveis variados de ruído e contraste.
  2. Retornar máscaras inteiras de instância $1 \dots N$ e máscara semântica binária.
  3. Executar treino de 5 a 10 épocas com batch size pequeno (ex: 8 ou 16) usando otimizador Adam.
  4. Registrar tempo de execução (< 5 min) e evolução da perda como teste de sanidade.

---

### Passo 1: Ingestão, Pré-processamento e Estratificação do DSB2018 — `[⏳ A Fazer / Dados Brutos Disponíveis]`
*Objetivo: Estruturar o carregamento do dataset real com particionamento estratificado confiável.*

- **O que fazer:**
  - `[✅ Concluído]` Download e extração das pastas de `stage1_train` (já presentes localmente em `src/pa1/data/stage1_train/` com 670 imagens e suas respectivas máscaras).
  - `[⏳ A Fazer]` Leitura das pastas `<image_id>/images/*.png` e consolidação de todas as anotações individuais `<image_id>/masks/*.png` em máscaras unificadas (semântica e instâncias $1 \dots N$).
  - `[⏳ A Fazer]` Classificação automática das imagens por modalidade biológica (fluorescência fundo escuro, campo claro fundo claro, e histologia H&E colorida) por meio de estatísticas de cor e histograma.
  - `[⏳ A Fazer]` Split estratificado: dividir em **70% Treino**, **15% Validação** e **15% Teste**, preservando a proporção de cada modalidade.
  - `[⏳ A Fazer]` Implementar `Dataset` e `DataLoader` PyTorch com augmentations (`albumentations`: RandomCrop 256x256, flips, rotações) e remover o `NotImplementedError` de `src/pa1/main.py`.
- **Onde fazer:**
  - `src/pa1/data/dsb2018.py` `[⏳ A Criar]`
  - `src/pa1/main.py` `[⏳ Atualizar bloco de carregamento de dados reais]`
  - `config.yaml` `[⏳ Atualizar chave data.data_dir]`
- **Como fazer:**
  1. Ler os diretórios `<image_id>/images/*.png` e consolidar todas as anotações individuais em `<image_id>/masks/*.png` em uma única máscara com IDs $1 \dots N$.
  2. **Estratificação por modalidade:** Extrair média e desvio padrão dos canais de cor e intensidade de fundo para agrupar as amostras em 3 grupos:
     - Fluorescência (fundo escuro, núcleos brilhantes);
     - Campo claro (fundo claro, núcleos escuros);
     - Histologia corada / H&E (coloração púrpura/rosada).
  3. Dividir os dados em **70% Treino**, **15% Validação** e **15% Teste**, preservando a proporção exata de cada modalidade em cada conjunto.
  4. Configurar data augmentation com `albumentations` (RandomCrop 256x256, HorizontalFlip, VerticalFlip, RandomRotate90) apenas para o treino.

---

### Passo 2: Geração de Targets de 3 Classes para Trilha A — `[⏳ A Fazer]`
*Objetivo: Formular a representação multiclasse que permite ao modelo aprender a separação física entre núcleos encostados.*

- **O que fazer:**
  - `[⏳ A Fazer]` Criar função para converter máscaras individuais de instâncias em mapa de 3 classes (0: Fundo, 1: Interior erodido, 2: Fronteira dilatada).
  - `[⏳ A Fazer]` Parametrizar a espessura da fronteira (2 a 3 pixels) assegurando que núcleos vizinhos tenham uma barreira contínua.
  - `[⏳ A Fazer]` Integrar a transformação diretamente no `Dataset` do DSB2018 para retornar o tensor `mask_3class` no batch.
- **Onde fazer:**
  - `src/pa1/data/targets.py` `[⏳ A Criar]`
  - `src/pa1/data/dsb2018.py` `[⏳ Integrar na saída do getitem]`
- **Como fazer:**
  1. Para cada imagem com máscaras de instâncias individuais $\{M_i\}_{i=1}^N$:
     - **Classe 0 (Fundo):** Pixels onde $\bigcup M_i = 0$.
     - **Classe 1 (Interior):** Erosão morfológica de cada máscara $M_i$ por um elemento estruturante (disco com raio $r_e \approx 1\text{--}2\text{ px}$).
     - **Classe 2 (Fronteira):** Zona de separação formada pela casca externa de cada núcleo e pela sobreposição das dilatações das bordas individuais ($r_d \approx 2\text{--}3\text{ px}$) onde instâncias vizinhas se encostam.
  2. Garantir que as fronteiras entre núcleos encostados tenham espessura de ao menos 2 a 3 pixels contínuos, impedindo que a binarização una os interiores.
  3. Integrar essa transformação no `Dataset` do DSB2018.

---

### Passo 3: Decodificadores de Instância e Métrica mAP@[0.5:0.95] — `[🟡 Parcialmente Implementado]`
*Objetivo: Implementar o cálculo formal da métrica de avaliação e as duas estratégias de pós-processamento.*

- **O que fazer:**
  - `[✅ Concluído]` Implementar decodificação ingênua via limiar e componentes conexos (`semantic_to_instances` em `connected_components.py`).
  - `[✅ Concluído]` Implementar cálculo formal de mAP@[0.50:0.05:0.95] com matching Hungarian (`scipy.optimize.linear_sum_assignment`) e busca gulosa, além do erro absoluto de contagem (`compute_map` em `metrics/instance.py`).
  - `[⏳ A Fazer]` Implementar decodificação avançada Watershed com marcadores de interior (`watershed_to_instances` em `watershed.py`).
- **Onde fazer:**
  - `src/pa1/postprocessing/connected_components.py` `[✅ Implementado]`
  - `src/pa1/metrics/instance.py` `[✅ Implementado]`
  - `src/pa1/postprocessing/watershed.py` `[⏳ A Criar]`
- **Como fazer:**
  1. **Decodificação Ingênua:** Aplicar limiar de probabilidade ($p > 0.5$) na predição semântica e rotular componentes conexos com `scipy.ndimage.label`.
  2. **Decodificação Watershed:**
     - Obter mapa de probabilidade da classe `Interior` e aplicar limiar alto ($p > 0.6$) para gerar os marcadores (sementes iniciais).
     - Como alternativa/refinamento, aplicar `scipy.ndimage.distance_transform_edt` na máscara de foreground e extrair máximos locais (`skimage.feature.peak_local_max`).
     - Utilizar o mapa de probabilidade invertido (ou distância negativa) como bacia topográfica e aplicar `skimage.segmentation.watershed(mask=foreground)`.
  3. **Métrica mAP de Instância:**
     - Para cada limiar $t \in [0.50, 0.55, \dots, 0.95]$:
       - Calcular matriz de IoU entre todas as instâncias preditas e os ground truths.
       - Realizar pareamento unívoco (Hungarian algorithm via `scipy.optimize.linear_sum_assignment` ou matching guloso ordenado por IoU decrescente).
       - Pares com $\text{IoU} \ge t$ são computados como True Positive (TP); predições restantes são False Positives (FP); ground truths não pareados são False Negatives (FN).
       - Calcular $\text{AP}(t) = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$.
     - O $\text{mAP}$ final é a média dos $\text{AP}(t)$ sobre todos os limiares.
  4. **Erro de Contagem:** Computar o erro absoluto médio por imagem $|\hat{N}_{\text{pred}} - N_{\text{true}}|$.

---

### Passo 4: Baseline de Segmentação Semântica e Análise de Fracasso (Parte 1 do PA1) — `[🟡 Parcial / Treino Real Pendente]`
*Objetivo: Estabelecer o ponto de partida com segmentação binária e evidenciar a limitação fundamental da abordagem semântica.*

- **O que fazer:**
  - `[✅ Concluído]` U-Net binária e função de perda `BCEDiceLoss` implementadas no código.
  - `[⏳ A Fazer]` Executar o treinamento da U-Net binária sobre o split de treino do DSB2018 até convergência.
  - `[⏳ A Fazer]` Avaliar na validação e teste: reportar IoU e Dice semânticos, além de mAP@[0.5:0.95] e erro de contagem via componentes conexos.
  - `[⏳ A Fazer]` Quantificação do fracasso: gerar o gráfico de dispersão com curva de tendência `mAP vs. Densidade de Objetos (núcleos por imagem)` demonstrando o colapso do método ingênuo quando há aglomeração.
- **Onde fazer:**
  - `src/pa1/models/unet.py` `[✅ Pronto]`
  - `src/pa1/losses/segmentation.py` `[✅ Pronto]`
  - `src/pa1/main.py` `[⏳ Executar treino binário real]`
  - Resultados e gráficos salvos em `outputs/baseline/` `[⏳ A Gerar]`
- **Como fazer:**
  1. Treinar a U-Net binária utilizando Cross-Entropy ou Dice Loss nos dados de treino do DSB2018.
  2. Avaliar no conjunto de validação reportando IoU e Dice semânticos.
  3. Converter a predição binária em instâncias via componentes conexos e avaliar com mAP@[0.5:0.95] e erro absoluto de contagem.
  4. **Quantificação do fracasso:**
     - Calcular para cada imagem de teste o número de núcleos reais (densidade) e o mAP obtido.
     - Plotar gráfico de dispersão com curva de regressão/tendência: `mAP vs. Número de Núcleos por Imagem`.
     - Mostrar que para imagens com alta aglomeração de núcleos o mAP desaba porque núcleos colados são mesclados em um único componente conexo gigante.

---

### Passo 5: Treinamento da Trilha A — Fronteiras e Watershed (Parte 2 do PA1) — `[⏳ A Fazer]`
*Objetivo: Fazer o mesmo encoder-decoder produzir predições instance-aware através da Trilha A.*

- **O que fazer:**
  - `[⏳ A Fazer]` Configurar U-Net para saída de 3 classes (0: fundo, 1: interior, 2: fronteira) em `heads.py` ou via parâmetro `out_channels=3`.
  - `[⏳ A Fazer]` Estender `FocalLoss` em `losses/segmentation.py` para suporte multiclasse ponderado ($\alpha_c$ ajustável para a classe fronteira).
  - `[⏳ A Fazer]` Treinar a rede nos dados do DSB2018 com o target de 3 classes.
  - `[⏳ A Fazer]` Decodificar predições com Watershed e avaliar no conjunto de teste.
  - `[⏳ A Fazer]` Montar tabela e gráficos comparativos lado a lado (Baseline vs. Trilha A) comprovando o ganho de mAP.
- **Onde fazer:**
  - `src/pa1/models/heads.py` `[⏳ A Criar]`
  - `src/pa1/losses/segmentation.py` `[⏳ Atualizar com Focal multiclasse ponderada]`
  - Treinamento e avaliação via `src/pa1/main.py`
  - Saídas e gráficos em `outputs/trilha_a/` `[⏳ A Gerar]`
- **Como fazer:**
  1. Substituir a camada final da U-Net por uma convolução $1 \times 1$ com 3 saídas.
  2. Implementar `FocalLoss` multiclasse:
     $$\mathcal{L}_{focal} = - \alpha_c (1 - p_c)^\gamma \log(p_c)$$
     ponderando a classe fronteira com maior peso relativo para contrabalancear seu desbalanceamento severo (< 5% dos pixels totais).
  3. Treinar no conjunto de treino do DSB2018 com validação periódica.
  4. Decodificar via `watershed_to_instances` e avaliar no conjunto de teste.
  5. Montar tabela comparativa lado a lado:
     - IoU Semântico
     - mAP@[0.5:0.95] de Instância
     - Erro Absoluto Médio de Contagem
     - Gráfico comparativo de separação em imagens densas.

---

### Passo 6: Ablações Sistemáticas (Parte 3 do PA1) — `[⏳ A Fazer]`
*Objetivo: Investigar o impacto isolado de componentes arquiteturais e funções de perda com rigor estatístico.*

- **O que fazer:**
  - `[⏳ A Fazer]` **Eixo 1 (Recuperação de Resolução):** Implementar variante da U-Net sem skip connections (apenas interpolação/transposed conv) e comparar com a U-Net padrão com skips.
  - `[⏳ A Fazer]` **Eixo 2 (Função de Perda):** Treinar comparando Cross-Entropy Ponderada vs. Focal Loss variando $\gamma \in \{0, 1, 2, 5\}$.
  - `[⏳ A Fazer]` Executar cada configuração com **2 seeds** (ex: 42 e 123) e reportar **média $\pm$ desvio padrão** ($\mu \pm \sigma$).
  - `[⏳ A Fazer]` Consolidar tabelas e curvas de ablação para a apresentação.
- **Onde fazer:**
  - `src/pa1/models/unet.py` `[⏳ Adicionar opção sem skips no decoder]`
  - `src/pa1/losses/segmentation.py` `[⏳ Suporte aos diferentes gammas e perdas]`
  - Saídas salvas em `outputs/ablation/` `[⏳ A Gerar]`
- **Como fazer:**
  1. **Eixo 1 — Recuperação de Resolução:**
     - *Configuração A (Padrão):* U-Net com Skip Connections (concatenação de features do encoder no decoder).
     - *Configuração B (Sem Skips):* Decoder simples baseado em interpolação bilinear / transposed conv sem skip connections (estilo SegNet simplificado / decodificador ingênuo com mesmo encoder).
     - Analisar a perda de fidelidade na reconstituição das bordas finas entre núcleos quando não há skips.
  2. **Eixo 2 — Função de Perda:**
     - Comparar Cross-Entropy Ponderada com Focal Loss variando $\gamma \in \{0, 1, 2, 5\}$.
     - Avaliar se valores maiores de $\gamma$ ajudam o modelo a focar nos pixels difíceis de fronteira e qual o impacto nas métricas de mAP e precisão de contorno.
  3. Executar todas as configurações com as seeds 42 e 123.
  4. Consolidar os resultados em tabelas formatadas com média e desvio padrão para apresentação.

---

### Passo 7: Inferência em Mosaico e Fusão de Borda (Parte 4 do PA1) — `[⏳ A Fazer]`
*Objetivo: Demonstrar a inferência em imagens gigantes por janelas deslizantes e resolver a fragmentação de instâncias na borda dos tiles.*

- **O que fazer:**
  - `[⏳ A Fazer]` Montar um mega-mosaico costurando 4 a 9 imagens de teste do DSB2018.
  - `[⏳ A Fazer]` Executar inferência em janelas deslizantes com sobreposição ($256 \times 256$ com overlap de 64 px conforme slide 83).
  - `[⏳ A Fazer]` Evidenciar visualmente o problema de corte de instâncias na linha de fronteira entre tiles.
  - `[⏳ A Fazer]` Implementar algoritmo de fusão/costura de instâncias que se tocam na faixa de sobreposição.
  - `[⏳ A Fazer]` Medir e reportar o mAP@[0.5:0.95] do mosaico antes e depois da fusão.
- **Onde fazer:**
  - `src/pa1/tiling/mosaic.py` `[⏳ A Criar]`
  - Figuras e métricas salvas em `outputs/mosaic/` `[⏳ A Gerar]`
- **Como fazer:**
  1. Costurar de 4 a 9 imagens de teste do DSB2018 para formar uma imagem sintética de grande escala (ex: $1024 \times 1024$).
  2. Implementar janelas deslizantes com patches de $256 \times 256$ e overlap de $64\text{ px}$.
  3. Fazer a média das probabilidades semânticas na região sobreposta e aplicar watershed tile por tile.
  4. Evidenciar o problema: instâncias que caem na linha de divisão são cortadas ao meio e contabilizadas como dois objetos distintos, degradando o erro de contagem e gerando falsos positivos na borda.
  5. **Algoritmo de Fusão de Instâncias:**
     - Analisar as máscaras de instâncias na faixa de sobreposição de 64 px entre tiles adjacentes.
     - Quando duas predições de tiles vizinhos apresentarem sobreposição espacial com $\text{IoU} > \tau_{\text{merge}}$ (ex: $\tau_{\text{merge}} = 0.3$), fundi-las em um único ID de instância.
  6. Calcular e reportar o mAP@[0.5:0.95] e a contagem de núcleos no mosaico completo **antes** e **depois** da fusão de bordas.

---

### Passo 8: Galeria de Falhas e Análise de Campo Receptivo (Parte 5 do PA1) — `[⏳ A Fazer]`
*Objetivo: Confrontar a teoria do campo receptivo com as falhas observadas do modelo e implementar uma intervenção corretiva funcional.*

- **O que fazer:**
  - `[⏳ A Fazer]` Dedução matemática analítica formal do Campo Receptivo Teórico (RF) do encoder da U-Net passo a passo (slides 35–38).
  - `[⏳ A Fazer]` Calcular o histograma de diâmetros dos núcleos no DSB2018 e comparar com o RF teórico calculado.
  - `[⏳ A Fazer]` Selecionar 5 imagens emblemáticas de erro feio e montar figura de 4 painéis com diagnósticos detalhados.
  - `[⏳ A Fazer]` Implementar 1 intervenção/correção motivada pelo diagnóstico e demonstrar a comparação quantitativa/visual antes vs. depois.
- **Onde fazer:**
  - Scripts de análise e figuras salvas em `outputs/failures/` `[⏳ A Gerar]`
- **Como fazer:**
  1. **Cálculo Analítico do Campo Receptivo (RF):**
     - Utilizar as fórmulas dos slides 35–38 da aula:
       $$r_{0} = 1, \quad r_{l} = r_{l-1} + (k_l - 1) \cdot j_{l-1}, \quad j_l = j_{l-1} \cdot s_l$$
       onde $k_l$ é o tamanho do kernel, $s_l$ o stride e $j_l$ o salto cumulativo.
     - Detalhar passo a passo cada camada convolucional e de pooling do encoder da U-Net até o gargalo (bottleneck).
  2. **Histograma de Diâmetros:**
     - Calcular o diâmetro equivalente ($\sqrt{4 \cdot \text{Área} / \pi}$) de todas as instâncias do dataset de teste e plotar o histograma com o limite do RF teórico assinalado.
  3. **Painel de 5 Falhas Críticas:**
     - Para cada uma das 5 imagens selecionadas, montar figura com 4 painéis:
       `[Imagem Original | Ground Truth | Predição de Instâncias | Mapa de Probabilidade de Fronteira]`.
     - Adicionar diagnóstico técnico explícito (ex: núcleo gigante cujo diâmetro excede o RF da rede, conglomeração extrema de núcleos com bordas indistinguíveis, baixa relação sinal-ruído em imagem de campo claro).
  4. **Correção e Avaliação:**
     - Implementar uma intervenção direcionada ao diagnóstico (ex: escalonamento de entrada com multi-escala, ajuste no limiar morfológico do watershed ou adição de convoluções dilatadas/atrous no gargalo).
     - Apresentar a comparação visual e numérica de desempenho antes vs. depois da correção.

---

### Passo 9: Teste de Estresse por Corrupções Sintéticas (Parte 6 do PA1) — `[⏳ A Fazer]`
*Objetivo: Avaliar quantitativamente a sensibilidade e degradação do modelo frente a perturbações nas imagens de entrada sem retreino.*

- **O que fazer:**
  - `[⏳ A Fazer]` Implementar gerador de corrupções com 3 perturbações (Gaussian Blur, Gaussian Noise e Variação de Contraste) em 3 intensidades cada (leve, moderado, severo).
  - `[⏳ A Fazer]` Avaliar o modelo final congelado (sem retreino) em cada cenário perturbado.
  - `[⏳ A Fazer]` Plotar as curvas de degradação de mAP@[0.5:0.95] em função da severidade do ruído.
- **Onde fazer:**
  - `src/pa1/stress/corruptions.py` `[⏳ A Criar]`
  - Figuras e tabelas salvas em `outputs/stress/` `[⏳ A Gerar]`
- **Como fazer:**
  1. Implementar as 3 corrupções:
     - **Gaussian Blur:** $\sigma \in \{1.0, 2.0, 3.5\}$;
     - **Gaussian Noise:** aditivo com desvio padrão $\sigma_{\text{noise}} \in \{0.05, 0.15, 0.30\}$;
     - **Contraste / Brilho:** fator multiplicativo de escala $\alpha \in \{0.7, 0.4, 0.2\}$ ou saturação extrema.
  2. Para cada corrupção e cada nível de severidade, rodar a inferência com o modelo final (sem nenhum fine-tuning).
  3. Gerar gráficos de linha: `mAP@[0.5:0.95] vs. Nível de Severidade` para cada perturbação.
  4. Documentar quais perturbações causam maior perda de separação de instâncias (ex: blur borra as fronteiras estreitas, enquanto ruído afeta os marcadores de semente).

---

### Passo 10: Consolidação dos Entregáveis e Reprodutibilidade — `[🟡 Parcialmente Iniciado]`
*Objetivo: Empacotar todo o código, checkpoints e documentação para entrega e apresentação.*

- **O que fazer:**
  - `[⏳ A Fazer]` Notebook executável `inferencia.ipynb` (recebe caminho de qualquer imagem externa, roda inferência e plota instâncias coloridas com contagem total sem retreinar).
  - `[⏳ A Fazer]` Arquivo `AI_LOG.md` descrevendo reflexivamente o uso de ferramentas de IA ao longo do projeto.
  - `[🟡 Parcial]` Atualizar `README.md` com os comandos finais de treino e avaliação reproduzíveis via `uv`.
  - `[⏳ A Fazer]` Preservar e documentar o checkpoint dos pesos do modelo final (`outputs/checkpoints/best_model.pt`).
- **Onde fazer:**
  - `inferencia.ipynb` `[⏳ A Criar / Atualizar a partir de pa1.ipynb]`
  - `AI_LOG.md` `[⏳ A Criar]`
  - `README.md` `[🟡 Atualizar comandos]`
  - `outputs/checkpoints/best_model.pt` `[⏳ A Salvar após treino]`
- **Como fazer:**
  1. **`inferencia.ipynb`:**
     - Notebook limpo e autocontido que carrega o modelo treinado a partir de `best_model.pt`.
     - Recebe o caminho de uma imagem qualquer (ex: PNG externa ou do teste).
     - Executa o pré-processamento, a inferência da rede e a decodificação Watershed.
     - Plota a imagem original, as instâncias segmentadas coloridas com colormap discreto e exibe a contagem total de núcleos encontrados.
  2. **`AI_LOG.md`:**
     - Registro reflexivo sobre como a IA foi utilizada (auxílio na vetorização do matching Hungarian, otimização da decodificação watershed, dedução analítica do RF).
     - Citação de desafios encontrados e como foram superados.
  3. **`README.md`:**
     - Instruções de setup do ambiente com `uv` (`uv sync`).
     - Instrução de download dos dados.
     - Comando único para treino completo (`uv run python -m pa1.main --train`).
     - Comando único para avaliação e geração dos artefatos (`uv run python -m pa1.main --eval`).

---

## 3. Interfaces e Contratos de Dados entre Módulos

Para assegurar integração direta e contínua entre as etapas, os seguintes contratos de dados devem ser respeitados:

### Contrato do Dicionário de Batch (`DataLoader`):
```python
batch = {
    "image": torch.Tensor,         # Shape: (B, C, H, W), float32 normalizado em [0, 1]
    "mask_semantic": torch.Tensor, # Shape: (B, H, W), int64, {0: fundo, 1: núcleo}
    "mask_3class": torch.Tensor,   # Shape: (B, H, W), int64, {0: fundo, 1: interior, 2: fronteira}
    "mask_instances": torch.Tensor # Shape: (B, H, W), int64, rótulos 0 (fundo) e 1..N (instâncias)
}
```

### Contrato da Decodificação Watershed:
```python
def watershed_to_instances(
    prob_3class: np.ndarray,          # Shape: (3, H, W), probabilidades após softmax
    interior_threshold: float = 0.6,  # Limiar de confiança para sementes
    min_area: int = 15,               # Filtro de ruído para instâncias espúrias
) -> np.ndarray:                      # Retorna: (H, W) int32, 0 para fundo e 1..N para instâncias
    ...
```

### Contrato da Métrica de Instância:
```python
def evaluate_instances(
    pred_instances: np.ndarray,       # Shape: (H, W), rótulos de instâncias preditas
    gt_instances: np.ndarray,         # Shape: (H, W), rótulos de instâncias ground truth
    iou_thresholds: list[float] = [0.5 + 0.05 * i for i in range(10)], # [0.50:0.05:0.95]
    matcher: str = "hungarian",       # "hungarian" ou "greedy"
) -> dict[str, float]:                # Retorna: {"mAP": float, "ap_50": float, "count_error": int, ...}
    ...
```

---

## 4. Matriz Sequencial de Execução com Status Atual

| Ordem | Etapa | Status | Entregável / Saída | Critério de Verificação |
| :---: | :--- | :---: | :--- | :--- |
| **1** | **Passo 0: Unit Test Sintético** | `[✅ Concluído]` | Pipeline executável em `synthetic.py` e `pa1.ipynb` | Treino converge em menos de 5 min com métricas calculadas |
| **2** | **Passo 1: Dataset DSB2018** | `[⏳ A Fazer]` | `dsb2018.py` com split estratificado | Divisão 70/15/15 mantendo proporção exata das 3 modalidades |
| **3** | **Passo 2: Target 3 Classes** | `[⏳ A Fazer]` | `targets.py` com interior e fronteira | Separação contínua (2-3px) de núcleos colados visível no batch |
| **4** | **Passo 3: Watershed & Métricas** | `[🟡 Parcial]` | `watershed.py` e `instance.py` | mAP@[0.5:0.95] computado via matching formal Hungarian/Greedy |
| **5** | **Passo 4: Baseline Semântica** | `[🟡 Parcial]` | Pesos da baseline e gráfico de fracasso | Gráfico `mAP vs. Densidade de Objetos` evidenciando queda de performance |
| **6** | **Passo 5: Trilha A (Fronteiras)** | `[⏳ A Fazer]` | Modelo treinado com Focal Loss | Tabela lado a lado mostrando ganho expressivo de mAP sobre a baseline |
| **7** | **Passo 6: Ablações** | `[⏳ A Fazer]` | Tabelas de ablação (Eixo 1 e Eixo 2) | Médias e desvios reportados ($\mu \pm \sigma$) em 2 seeds para cada caso |
| **8** | **Passo 7: Mosaico e Fusão** | `[⏳ A Fazer]` | `mosaic.py` e figuras de costura | Comparação quantitativa do mAP antes e depois da fusão de bordas |
| **9** | **Passo 8: Galeria de Falhas** | `[⏳ A Fazer]` | Dedução do RF e painel com 5 falhas | RF deduzido formalmente, comparado ao histograma e 1 correção validada |
| **10** | **Passo 9: Teste de Estresse** | `[⏳ A Fazer]` | `corruptions.py` e curvas de degradação | Curvas de degradação de mAP sob blur, ruído e contraste em 3 intensidades |
| **11** | **Passo 10: Entregáveis Finais** | `[🟡 Parcial]` | `inferencia.ipynb`, `AI_LOG.md`, `README.md` | Notebook roda sem retreino; ambiente e treino reproduzíveis via linha de comando |

---

## 5. Boas Práticas de Engenharia e Controle de Versão

- **Feature Branches por Funcionalidade:**
  - Trabalhar em branches temáticas baseadas na etapa em desenvolvimento (ex: `feature/dataset-stratification`, `feature/watershed-pipeline`, `feature/mosaic-tiling`, `feature/ablation-study`).
- **Commits Atômicos e Frequentes:**
  - Manter histórico contínuo e bem distribuído ao longo das duas semanas (`feat: implement hungarian instance matcher`, `fix: adjust border erosion radius for small nuclei`).
- **Prevenção de Arquivos Pesados no Git:**
  - Manter pastas de dados brutos e imagens no `.gitignore`.
  - Checkpoints pesados devem ser salvos em `outputs/checkpoints/` e versionados externamente ou ignorados, mantendo apenas o link ou o peso final estritamente necessário.
- **Reprodutibilidade:**
  - Fixação determinística de seeds para PyTorch, NumPy e Python random, viabilizando a conferência exata dos resultados.
