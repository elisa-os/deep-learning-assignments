#!/usr/bin/env bash
# run.sh — wrapper para executar o PA1 com suporte à GTX 1080 Ti (sm_61)
#
# Por que este wrapper existe:
#   O torch 2.7.1+cu118 foi compilado contra NCCL 2.22+, mas o PyPI cu11 só
#   distribui NCCL até 2.21.5. Em treinamento single-GPU o NCCL nunca é
#   chamado, mas o dynamic linker exige os símbolos ao carregar libtorch_cuda.so.
#   O nccl_stub.so fornece esses símbolos como noops seguros.
#
# Uso:
#   bash run.sh                   # equivalente a: uv run pa1
#   bash run.sh --epochs 30       # passa flags normalmente
#   bash run.sh jupyter notebook  # qualquer subcomando uv run
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUB="$SCRIPT_DIR/.venv/nccl_stub.so"

# Compilar o stub se não existir
if [[ ! -f "$STUB" ]]; then
    echo "[run.sh] Compilando nccl_stub.so..."
    bash "$SCRIPT_DIR/scripts/build_nccl_stub.sh"
fi

# Executar com o stub pré-carregado
exec env LD_PRELOAD="$STUB" uv run "$@"
