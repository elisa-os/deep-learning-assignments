# Plano de Execução PA1 — Segmentação de Instâncias
**Disciplina:** Aprendizado Profundo | FGV CDIA  
**Alunos:** Bruno Ferreira & Elisa Soares  
**Decisões de Projeto:**
- **Dataset:** Opção A — DSB2018 (`stage1_train`)
- **Abordagem de Instância:** Trilha A — Fronteiras e Watershed
- **Ablações:** Eixo 1 (Recuperação de Resolução: U-Net com Skips vs. SegNet / Sem Skips) e Eixo 2 (Perdas: Focal Loss $\gamma \in \{0, 1, 2, 5\}$)
- **Teste de Estresse:** Corrupções sintéticas (Blur, Ruído Gaussiano e Contraste) em 3 intensidades

---

## 1. Visão Geral da Arquitetura e Contrato de Código

Para garantir que Bruno e Elisa trabalhem simultaneamente sem conflitos de merge no Git e sem bloqueios de dependência, a divisão é feita por módulos com **contratos de interface definidos**:

```
src/pa1/
├── data/
│   ├── synthetic.py       ← [Pronto] Teste unitário (Parte 0)
│   ├── dsb2018.py         ← [Bruno] Loader e split estratificado do DSB2018
│   └── targets.py         ← [Elisa] Geração de máscaras 3 classes (Fundo/Interior/Fronteira)
├── models/
│   ├── unet.py            ← [Pronto/Elisa] U-Net e variantes para ablação (Eixo 1)
│   └── heads.py           ← [Elisa] Cabeça 3-classes e mapas auxiliares
├── losses/
│   └── segmentation.py    ← [Elisa/Pronto] Focal Loss parametrizável por gamma e pesos
├── postprocessing/
│   ├── connected_components.py ← [Pronto/Bruno] Baseline ingênuo (Parte 1)
│   └── watershed.py       ← [Elisa] Decodificação watershed com marcadores
├── metrics/
│   └── instance.py        ← [Pronto] Matching Hungarian/Greedy e mAP@[.5:.95]
├── tiling/
│   └── mosaic.py          ← [Bruno] Inferência em mosaico e fusão de bordas (Parte 4)
└── stress/
    └── corruptions.py     ← [Bruno] Gerador de corrupções (Parte 6)
```

---

## 2. Divisão de Responsabilidades

### 👤 Trilha 1: Bruno (Infraestrutura de Dados, Baseline, Mosaico e Estresse)

Bruno foca em estruturar os dados reais, consolidar a baseline de segmentação semântica, implementar a inferência em imagens gigantes (mosaico) e preparar os entregáveis de inferência.

#### Tarefas de Bruno:
1. **Pipeline do Dataset DSB2018 (`src/pa1/data/dsb2018.py`):**
   - Script de download/extração do `stage1_train` (via Kaggle API ou link direto BBBC038).
   - Leitura de `<image_id>/images/*.png` e consolidação das máscaras individuais em `<image_id>/masks/*.png`.
   - **Estratificação:** Classificar as imagens por modalidade (fluorescência fundo escuro, campo claro fundo claro, e histologia colorida/H&E) usando estatísticas de cor/intensidade dos canais RGB. Dividir em `train (70%)`, `val (15%)` e `test (15%)` estratificado.
2. **Parte 1 — Baseline Semântica:**
   - Treinar a `UNet` binária existente nos dados do DSB2018.
   - Reportar IoU e Dice semânticos na validação.
   - Avaliar com extração ingênua (`connected_components.py`): mAP@[0.5:0.95] e erro médio de contagem.
   - **Quantificação do fracasso:** Gerar o gráfico de dispersão/tendência: `mAP vs. Densidade de Objetos (número de núcleos por imagem)` demonstrando que a baseline falha quando há aglomeração.
3. **Parte 4 — Inferência em Mosaico (`src/pa1/tiling/mosaic.py`):**
   - Costurar 4 a 9 imagens de teste gerando um "mega-mosaico".
   - Executar inferência em janelas deslizantes (patches de 256x256 com overlap de 64px) conforme slide 83.
   - Evidenciar visualmente o problema de corte de objetos na fronteira dos tiles.
   - Implementar algoritmo de costura/fusão de instâncias (unir componentes que se tocam na faixa de sobreposição se IoU > limiar). Reportar mAP antes e depois da fusão.
4. **Parte 6 — Teste de Estresse (`src/pa1/stress/corruptions.py`):**
   - Aplicar 3 corrupções (Gaussian Blur, Gaussian Noise, Contraste) em 3 níveis de severidade (baixo, médio, alto).
   - Gerar curvas de degradação de mAP@[.5:.95].
5. **Entregável `inferencia.ipynb`:**
   - Notebook limpo que carrega o modelo final e uma imagem arbitrária, retornando a imagem com as instâncias coloridas e a contagem de núcleos.

---

### 👤 Trilha 2: Elisa (Engenharia de Instâncias, Watershed, Perdas e Teoria)

Elisa foca no núcleo da inovação de instâncias (Trilha A): formulação de targets de fronteira, decodificação via Watershed, ablações de perda e arquitetura, e cálculo teórico do campo receptivo.

> **Zero Bloqueio:** Elisa pode desenvolver e testar toda a lógica de fronteiras e watershed imediatamente usando o gerador sintético de elipses existente da Parte 0, sem precisar esperar o download dos dados reais!

#### Tarefas de Elisa:
1. **Target de 3 Classes (`src/pa1/data/targets.py`):**
   - Transformar as máscaras de instâncias individuais em um mapa com 3 classes:
     - `0 = Fundo`
     - `1 = Interior do Núcleo` (núcleo erodido morfologicamente)
     - `2 = Fronteira entre Instâncias` (dilatação das bordas individuais onde duas instâncias se tocam ou a casca externa do núcleo).
   - Parametrizar a espessura da fronteira (ex: 2 a 3 pixels).
2. **Pós-Processamento Watershed (`src/pa1/postprocessing/watershed.py`):**
   - Decodificar as predições da rede:
     - Sementes / Marcadores: áreas de alta confiança da classe `Interior` (ou picos de distância via `scipy.ndimage.distance_transform_edt`).
     - Bacia de atração: mapa de probabilidade ou distância invertida.
     - Executar `skimage.segmentation.watershed` impedindo que núcleos encostados se fundam.
3. **Parte 2 — Treinamento da Trilha A:**
   - Adaptar a cabeça final da UNet para 3 classes (`out_channels = 3`).
   - Aplicar `FocalLoss` balanceada para lidar com o fato de a classe `Fronteira` ser fortemente minoritária (<5% dos pixels).
   - Comparar métricas lado a lado com a Baseline do Bruno (Parte 1).
4. **Parte 3 — Ablações (2 Seeds cada, reportar $\mu \pm \sigma$):**
   - **Eixo 1 (Recuperação de Resolução):** U-Net padrão (Skip Connections) vs. SegNet / Decodificador sem skips (UpSampling simples).
   - **Eixo 2 (Função de Perda):** Cross-Entropy ponderada vs. Focal Loss com $\gamma \in \{0, 1, 2, 5\}$.
5. **Parte 5 — Teoria do Campo Receptivo & Galeria de Falhas:**
   - Dedução analítica formal do Campo Receptivo Teórico (RF) do encoder da U-Net (fórmulas passo a passo dos slides 35–38 da aula).
   - Comparação do RF obtido com o histograma de diâmetros dos núcleos no DSB2018.
   - Proposta e teste de 1 correção baseada no diagnóstico de falha.

---

## 3. Interfaces e Contratos de Código

Para garantir compatibilidade imediata quando os códigos de Bruno e Elisa se conectarem:

### Contrato do Dataset (`batch` retornado pelo DataLoader):
```python
batch = {
    "image": torch.Tensor,         # Shape: (B, C, H, W), float32 normalizado em [0, 1]
    "mask_semantic": torch.Tensor, # Shape: (B, H, W), int64, {0: fundo, 1: núcleo}
    "mask_3class": torch.Tensor,   # Shape: (B, H, W), int64, {0: fundo, 1: interior, 2: fronteira}
    "mask_instances": torch.Tensor # Shape: (B, H, W), int64, IDs de 1..N para cada objeto
}
```

### Contrato da Decodificação Watershed:
```python
def watershed_to_instances(
    prob_3class: np.ndarray,      # Shape: (3, H, W), probabilidades softmax
    interior_threshold: float = 0.6,
    min_area: int = 15,
) -> np.ndarray:                  # Retorna: (H, W) com IDs inteiros das instâncias
    ...
```

---

## 4. Cronograma de Sincronização e Entregáveis

| Etapa | Bruno | Elisa | Ponto de Sincronização |
| :--- | :--- | :--- | :--- |
| **Fase 1** | Implementar `dsb2018.py` e split estratificado | Implementar target 3 classes e testar no sintético | Ambos testam seus módulos isoladamente |
| **Fase 2** | Treinar Baseline Parte 1 e plotar fracasso vs densidade | Implementar `watershed.py` e perda Focal balanceada | Merge de dados e heads na `main` |
| **Fase 3** | Implementar Inferência em Mosaico (Parte 4) | Treinar Trilha A no DSB2018 e comparar com Baseline | Comparação de mAP Parte 1 vs Parte 2 |
| **Fase 4** | Implementar Teste de Estresse por Corrupções (Parte 6) | Rodar Ablações (Eixos 1 e 2 com 2 seeds) | Coleta de tabelas de ablação |
| **Fase 5** | Montar `inferencia.ipynb` e estruturar `AI_LOG.md` | Montar diagnóstico de Campo Receptivo e Galeria (Parte 5) | Selecionar 5 imagens de erro |
| **Fase 6** | Validação final de reprodução (`uv run pa1`) | Preparação visual das figuras para apresentação | Fechamento do repositório |

---

## 5. Boas Práticas de Git para Atender aos Critérios do PA1

O enunciado do [PA1.pdf](PA1.pdf) exige histórico distribuído com acesso para a monitoria.

1. **Branches Nominais:**
   - Bruno: `git checkout -b feature/data-pipeline-baseline`
   - Elisa: `git checkout -b feature/watershed-instance-head`
2. **Commits Pequenos e Frequentes:**
   - Faça commits focados (`feat: add stratified split for DSB2018`, `fix: adjust border dilation radius`).
3. **Arquivos Binários e Pesados:**
   - Nunca comitar arquivos brutos de imagem do DSB2018 no Git. O script deve baixá-los para uma pasta ignorada (`data/` no `.gitignore`).
   - Checkpoints `.pt` finais devem ser salvos em `outputs/checkpoints/` e compartilhados via link se excederem o limite.
