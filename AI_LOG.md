# AI_LOG — Registro de Uso de Inteligência Artificial

**Disciplina:** Aprendizado Profundo | FGV CDIA  
**Alunos:** Bruno Ferreira & Elisa Soares  
**Assignment:** Programming Assignment 1 (PA1) — Segmentação de Instâncias  

Este documento registra os episódios de utilização de ferramentas de Inteligência Artificial durante o desenvolvimento do PA1, conforme orientado na seção 5 do enunciado.

---

## Episódio 1: Modularização da Arquitetura do Repositório
* **Data:** 05/09/2026
* **Ferramenta:** Antigravity / LLM
* **Contexto e Motivação:** O projeto iniciou com código exploratório concentrado em um único notebook (`pa1.ipynb`). Para permitir colaboração simultânea e reprodução limpa via terminal com `uv`, era necessário desacoplar responsabilidades em módulos Python (`src/pa1/`).
* **Como a IA auxiliou:**
  - Criação da estrutura de pacotes: `data`, `models`, `losses`, `metrics`, `postprocessing`, `utils`.
  - Configuração da CLI via `pyproject.toml` e gerenciamento de parâmetros via `config.yaml` com dataclasses tipadas (`config.py`).
  - Adaptação das funções do notebook para módulos reutilizáveis.
* **Validação / Decisões Humanas:**
  - Verificação de paridade de métricas da Parte 0 entre o notebook legado e a CLI refatorada.
  - Revisão e manutenção do padrão semântico exigido pelo enunciado.

---

## Episódio 2: Planejamento Estratégico e Paralelização (Opção A + Trilha A)
* **Data:** 05/09/2026
* **Ferramenta:** Antigravity / LLM
* **Contexto e Motivação:** Definição das escolhas do enunciado (Dataset DSB2018 `stage1_train` e Trilha A - Fronteiras e Watershed) e estruturação do plano de trabalho para duas pessoas (Bruno e Elisa) sem bloqueios ou dependências mútuas.
* **Como a IA auxiliou:**
  - Mapeamento de contratos de interface (estrutura de batches e assinaturas das funções de pós-processamento).
  - Divisão de responsabilidades detalhada em `PLANO_DE_EXECUCAO.md` (Bruno focado em dados, baseline semântica e mosaico; Elisa focada na geração do target de 3 classes, perdas focal e decodificação watershed).
* **Validação / Decisões Humanas:**
  - Alinhamento da escolha do dataset DSB2018 pela facilidade de obtenção e relevância biológica (células encostadas).
  - Escolha da Trilha A pela forte conexão com os conceitos vistos nas aulas de U-Net.

## Episódio 3: Implementação e Otimização das Ablações (Parte 3)
* **Data:** 08/09/2026
* **Ferramenta:** Antigravity / LLM
* **Contexto e Motivação:** Para a Parte 3, era necessário executar uma série de treinamentos variando hiperparâmetros e arquitetura (eixo 1: sem/com skips; eixo 2: gammas da focal loss). Além disso, a etapa de avaliação estava sendo um gargalo massivo de tempo.
* **Como a IA auxiliou:**
  - Criação do script automatizado `pa1/ablation.py` que itera sobre a matriz de configurações dinamicamente, sem sujar o `config.yaml` principal.
  - Implementação das flags de ablação (`use_skips`, `loss_gamma`) nos módulos base (`unet.py`, `config.py`).
  - **Otimização Crítica:** A IA identificou que a lentidão era causada por um loop `for` O(N*P) em Python puro e sobrecarga na priority queue do Watershed por conta de ruídos. A IA refatorou `watershed.py` introduzindo vetorização com `np.bincount` e filtro morfológico prévio (eliminando marcadores espúrios). O tempo de avaliação despencou de ~5 minutos para milissegundos.
* **Validação / Decisões Humanas:**
  - Análise humana atestou que a correção lógica no loop do watershed também melhorou a estabilidade e a acurácia do Erro de Contagem.
  - Aprovação do commmit separado para a otimização de performance, mantendo as boas práticas de git.
