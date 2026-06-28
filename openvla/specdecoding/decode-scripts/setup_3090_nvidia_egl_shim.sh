#!/usr/bin/env bash
set -euo pipefail

DRIVER_VERSION="${NVIDIA_DRIVER_VERSION:-}"
if [[ -z "${DRIVER_VERSION}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d '[:space:]')"
  fi
fi

if [[ -z "${DRIVER_VERSION}" ]]; then
  echo "Unable to detect NVIDIA driver version. Set NVIDIA_DRIVER_VERSION explicitly." >&2
  exit 1
fi

INSTALL_ROOT="${NVIDIA_EGL_SHIM_ROOT:-/data/wulin/c/nvidia-egl-${DRIVER_VERSION}}"
RUNFILE="NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run"
RUNFILE_URL="${NVIDIA_RUNFILE_URL:-https://download.nvidia.com/XFree86/Linux-x86_64/${DRIVER_VERSION}/${RUNFILE}}"
EXTRACT_DIR="${INSTALL_ROOT}/NVIDIA-Linux-x86_64-${DRIVER_VERSION}"
SLIM_DIR="${INSTALL_ROOT}/slim-lib"
VENDOR_DIR="${INSTALL_ROOT}/egl_vendor.d"
VENDOR_JSON="${VENDOR_DIR}/10_nvidia_${DRIVER_VERSION//./_}.json"

mkdir -p "${INSTALL_ROOT}"
cd "${INSTALL_ROOT}"

if [[ ! -f "${RUNFILE}" ]]; then
  echo "Downloading ${RUNFILE_URL}"
  wget -O "${RUNFILE}" "${RUNFILE_URL}"
fi

if [[ ! -d "${EXTRACT_DIR}" ]]; then
  echo "Extracting ${RUNFILE}"
  sh "${RUNFILE}" --extract-only
fi

required_files=(
  "libEGL_nvidia.so.${DRIVER_VERSION}"
  "libGLESv2_nvidia.so.${DRIVER_VERSION}"
  "libGLX_nvidia.so.${DRIVER_VERSION}"
  "libnvidia-eglcore.so.${DRIVER_VERSION}"
  "libnvidia-glcore.so.${DRIVER_VERSION}"
  "libnvidia-glsi.so.${DRIVER_VERSION}"
)

for file in "${required_files[@]}"; do
  if [[ ! -f "${EXTRACT_DIR}/${file}" ]]; then
    echo "Missing ${EXTRACT_DIR}/${file}; the downloaded runfile may not match this driver version." >&2
    exit 1
  fi
done

mkdir -p "${SLIM_DIR}" "${VENDOR_DIR}"
for file in "${required_files[@]}"; do
  ln -sf "../NVIDIA-Linux-x86_64-${DRIVER_VERSION}/${file}" "${SLIM_DIR}/${file}"
done
ln -sf "libEGL_nvidia.so.${DRIVER_VERSION}" "${SLIM_DIR}/libEGL_nvidia.so.0"

cat > "${VENDOR_JSON}" <<JSON
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "${SLIM_DIR}/libEGL_nvidia.so.0"
  }
}
JSON

echo "NVIDIA EGL shim is ready."
echo "NVIDIA_EGL_SHIM_DIR=${SLIM_DIR}"
echo "NVIDIA_EGL_VENDOR_JSON=${VENDOR_JSON}"
echo
echo "Example:"
echo "  NVIDIA_EGL_SHIM_DIR=${SLIM_DIR} \\"
echo "  NVIDIA_EGL_VENDOR_JSON=${VENDOR_JSON} \\"
echo "    bash openvla/specdecoding/decode-scripts/run_dflash_libero_goal_eval.sh"
