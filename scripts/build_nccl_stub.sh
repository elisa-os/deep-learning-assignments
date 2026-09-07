#!/usr/bin/env bash
# build_nccl_stub.sh
# Compila e instala um stub compartilhado com símbolos NCCL 2.22+ ausentes
# no wheel pytorch/cu118 para Python 3.11.
#
# Necessário quando usando torch 2.7.1+cu118 em ambiente single-GPU com
# nvidia-nccl-cu11==2.21.5.  O stub implementa noops seguros para todos
# os símbolos novos — eles nunca são chamados em treinamento single-GPU.
#
# Uso:
#   bash scripts/build_nccl_stub.sh           # detecta .venv automaticamente
#   LD_PRELOAD=.venv/nccl_stub.so uv run pa1  # usa o stub ao executar
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
STUB_C="$ROOT_DIR/.venv/nccl_stub_src.c"
STUB_SO="$ROOT_DIR/.venv/nccl_stub.so"

cat > "$STUB_C" << 'EOF'
/* nccl_stub.c — stub para símbolos NCCL 2.22+ ausentes em nvidia-nccl-cu11 2.21.5
 * Seguro para uso single-GPU: o NCCL nunca é chamado nesses cenários.        */
typedef void*          ncclComm_t;
typedef int            ncclResult_t;
typedef unsigned long  ncclMemHandle_t;
typedef void*          ncclUniqueId;
typedef int            ncclRedOp_t;
typedef int            ncclDataType_t;
typedef void*          cudaStream_t;
#define NCCL_SUCCESS 0
ncclResult_t ncclAlltoAll(const void*s,void*r,unsigned long c,ncclDataType_t t,ncclComm_t cm,cudaStream_t st){return NCCL_SUCCESS;}
ncclResult_t ncclCommGetUniqueId(ncclUniqueId*id,ncclComm_t cm){return NCCL_SUCCESS;}
ncclResult_t ncclCommGrow(ncclComm_t*nc,ncclComm_t cm,int*rk,int n){return NCCL_SUCCESS;}
ncclResult_t ncclCommInitRankScalable(ncclComm_t*cm,int n,int r,int nr,void**cb,void*ctx){return NCCL_SUCCESS;}
ncclResult_t ncclCommMemStats(ncclComm_t cm,void*s){return NCCL_SUCCESS;}
ncclResult_t ncclCommResume(ncclComm_t cm){return NCCL_SUCCESS;}
ncclResult_t ncclCommRevoke(ncclComm_t cm){return NCCL_SUCCESS;}
ncclResult_t ncclCommShrink(ncclComm_t*nc,ncclComm_t cm,int*rk,int n){return NCCL_SUCCESS;}
ncclResult_t ncclCommSuspend(ncclComm_t cm){return NCCL_SUCCESS;}
ncclResult_t ncclCommWindowDeregister(ncclComm_t cm,ncclMemHandle_t h){return NCCL_SUCCESS;}
ncclResult_t ncclCommWindowRegister(ncclComm_t cm,void*b,unsigned long sz,ncclMemHandle_t*h){return NCCL_SUCCESS;}
ncclResult_t ncclDevCommCreate(void**dc,ncclComm_t cm){return NCCL_SUCCESS;}
ncclResult_t ncclDevCommDestroy(void*dc){return NCCL_SUCCESS;}
ncclResult_t ncclGather(const void*s,void*r,unsigned long c,ncclDataType_t t,int root,ncclComm_t cm,cudaStream_t st){return NCCL_SUCCESS;}
ncclResult_t ncclGetLsaMultimemDevicePointer(void**p,void*b,ncclComm_t cm){return NCCL_SUCCESS;}
ncclResult_t ncclGetPeerDevicePointer(void**p,void*b,int peer,ncclComm_t cm){return NCCL_SUCCESS;}
ncclResult_t ncclGroupSimulateEnd(unsigned long*bytes){return NCCL_SUCCESS;}
ncclResult_t ncclPutSignal(ncclMemHandle_t dst,const ncclMemHandle_t src,unsigned long off,unsigned long sz,ncclComm_t cm,cudaStream_t st){return NCCL_SUCCESS;}
ncclResult_t ncclSignal(ncclMemHandle_t h,unsigned long off,ncclComm_t cm,cudaStream_t st){return NCCL_SUCCESS;}
ncclResult_t ncclWaitSignal(ncclMemHandle_t h,unsigned long off,ncclComm_t cm,cudaStream_t st){return NCCL_SUCCESS;}
ncclResult_t ncclWinGetUserPtr(void**p,ncclMemHandle_t h,ncclComm_t cm){return NCCL_SUCCESS;}
EOF

gcc -shared -fPIC -o "$STUB_SO" "$STUB_C"
echo "✓ nccl_stub.so compilado em: $STUB_SO"
echo ""
echo "Para usar:"
echo "  LD_PRELOAD=$STUB_SO uv run pa1"
echo "  LD_PRELOAD=$STUB_SO uv run jupyter notebook"
