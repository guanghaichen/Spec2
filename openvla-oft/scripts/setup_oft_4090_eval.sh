#!/usr/bin/env bash

# Prepare an isolated OFT inference environment on the 4090 evaluation host.
# It intentionally does not install flash-attn: OFT L1 inference and the
# lightweight early-exit adapter work with the custom Transformers fork alone.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh"
CONDA_SH="/home/asus/miniconda3/etc/profile.d/conda.sh"
ENV_NAME="${OFT_ENV_NAME:-oft}"
THIRD_PARTY_ROOT="${WORK_ROOT}/openvla-oft-third_party"
TRANSFORMERS_DIR="${THIRD_PARTY_ROOT}/transformers-openvla-oft"
MODEL_DIR="${WORK_ROOT}/hf_files/openvla-7b-oft-finetuned-libero-goal"

source "${CONDA_SH}"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" --clone specvla -y
fi
conda activate "${ENV_NAME}"

export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${WORK_ROOT}/hf_files}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${WORK_ROOT}/hf_files}"

mkdir -p "${THIRD_PARTY_ROOT}" "${WORK_ROOT}/hf_files"
if [[ ! -d "${TRANSFORMERS_DIR}/.git" ]]; then
  git clone --depth 1 https://github.com/moojink/transformers-openvla-oft.git "${TRANSFORMERS_DIR}"
fi

pip install --no-deps -e "${TRANSFORMERS_DIR}"
pip install --no-deps -e "${OFT_ROOT}"
pip install "diffusers==0.30.3" "draccus==0.8.0" "peft==0.11.1" "json-numpy" "h5py"

huggingface-cli download moojink/openvla-7b-oft-finetuned-libero-goal \
  --local-dir "${MODEL_DIR}" \
  --local-dir-use-symlinks False

python - <<'PY'
import torch, transformers
print("OFT environment ready")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__, transformers.__file__)
PY
