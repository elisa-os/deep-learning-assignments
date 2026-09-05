# PA1 — Segmentação de Instâncias

**Disciplina:** Aprendizado Profundo | FGV CDIA  
**Alunos:** Bruno Ferreira e Elisa Soares  

Este repositório contém a solução do **Programming Assignment 1 (PA1)** focado em segmentação de instâncias (desde o baseline semântico até representações avançadas de instância, ablações e testes de estresse).

---

## 1. Configuração do Ambiente com `uv`

O projeto utiliza [`uv`](https://docs.astral.sh/uv/) para gerenciar dependências e o ambiente virtual de forma rápida e determinística:

```bash
# Sincroniza o ambiente e instala as dependências (cria .venv)
uv sync
```

---

## 2. Arquitetura do Repositório

Optamos por uma estrutura modular, limpa e funcional:

```
.
├── config.yaml          ← Central de parâmetros do projeto (edite aqui)
├── outputs/             ← Artefatos gerados: checkpoints (.pt), figuras e logs
├── src/
│   └── pa1/             ← Pacote Python do projeto
│       ├── main.py      ← Ponto de entrada do pipeline (CLI)
│       ├── config.py    ← Leitura e tipagem do config.yaml via dataclasses
│       ├── data/        ← Geração sintética (elipses) e loaders de dados reais
│       ├── models/      ← Arquiteturas de redes (UNet com skips)
│       ├── losses/      ← Funções de perda (Dice Loss, Focal Loss, BCE+Dice)
│       ├── metrics/     ← Métricas de instância: mAP@[.5:.95], Hungarian/Greedy matching
│       ├── postprocessing/ ← Métodos de decodificação (componentes conexos, etc.)
│       └── utils/       ← Fixação de seeds, detecção de device e visualização
├── pa1.ipynb            ← Notebook de laboratório e exploração
├── inferencia.ipynb     ← Entregável: roda inferência em imagem avulsa sem retreinar
├── PA1.pdf              ← Enunciado oficial
├── AI_LOG.md            ← Registro de uso e prompts com IA
├── pyproject.toml       ← Configuração do pacote e CLI pa1
└── .gitignore           ← Filtra outputs, checkpoints e caches
```

### Principais Decisões de Design:
- **`config.yaml` na raiz**: Centraliza todos os hiperparâmetros (épocas, lr, batch size, paths). Não é necessário digitar comandos com dezenas de flags no terminal.
- **`outputs/` na raiz**: Mantém artefatos temporários e pesados gerados em execução (imagens de diagnóstico, checkpoints treinados) fora do diretório de código `src/`.
- **`src/pa1/`**: Código modularizado por responsabilidade única (`data`, `models`, `losses`, `metrics`, `postprocessing`). Isso permite reaproveitar o mesmo pipeline na Parte 0 (sintética), Parte 1 (baseline) e Partes 2 a 6.

---

## 3. Como Treinar e Avaliar

### Execução Padrão
Basta editar o arquivo [`config.yaml`](config.yaml) conforme o experimento desejado e rodar:

```bash
uv run pa1
```

O script carregará o `config.yaml`, inicializará os dados/modelos, treinará e exibirá a evolução das métricas (Loss, mAP, erro de contagem de objetos).

### Sobrescrita Opcional por Linha de Comando
Caso queira testar rapidamente algo pontual sem alterar o YAML:

```bash
# Rodar menos épocas para teste rápido
uv run pa1 --epochs 5

# Avaliar um modelo treinado existente
uv run pa1 --eval-only --checkpoint outputs/model.pt

# Usar outro arquivo de configuração
uv run pa1 --config outro_experimento.yaml
```

---

## 4. Entregável de Inferência (`inferencia.ipynb`)

Conforme especificado nos entregáveis do PA1:
- O arquivo [`inferencia.ipynb`](inferencia.ipynb) localiza-se na raiz do repositório.
- Ele **não retreina o modelo**: recebe o caminho de uma imagem e um checkpoint, aplica a inferência e retorna a máscara de instâncias colorida junto com a contagem predita de objetos.

---

## 5. Download dos Dados Reais

Para as partes posteriores à Parte 0:
- **DSB2018** (Opção A): [Kaggle 2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018)
- **SpaceNet** (Opção B): [SpaceNet Challenge](https://spacenet.ai/datasets/)

Ao fazer o download, aponte a chave `data_dir:` no [`config.yaml`](config.yaml) para a pasta dos dados baixados e alterne `synthetic: false`.
