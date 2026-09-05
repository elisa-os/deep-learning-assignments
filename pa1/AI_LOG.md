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
