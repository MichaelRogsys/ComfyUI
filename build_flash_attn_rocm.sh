#!/usr/bin/env bash
set -Eeuo pipefail

app_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source_dir="$app_dir/.flash-attention-src"
venv_dir="$app_dir/.venv"
python="$venv_dir/bin/python"
uv="$venv_dir/bin/uv"

for required in "$python" "$uv" "$venv_dir/bin/rocm-sdk" "$source_dir/setup.py"; do
  if [[ ! -e "$required" ]]; then
    printf 'Missing: %s\n' "$required" >&2
    exit 1
  fi
done

sdk_version=$("$python" -c 'from importlib.metadata import version; sdk = version("rocm-sdk-core"); torch = version("torch"); assert "+rocm" + sdk in torch, f"Torch {torch} does not match ROCm SDK {sdk}"; print(sdk)')
"$python" -c 'import torch; assert torch.cuda.is_available(), "ROCm GPU not visible"; assert torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1100"), "Expected gfx1100 GPU"'
site_packages=$("$python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
export UV_CACHE_DIR="${UV_CACHE_DIR:-$source_dir/agent_space/uv-cache}"
mkdir -p "$UV_CACHE_DIR"
log="$app_dir/flash-attn-build-$(date +%Y%m%d-%H%M%S).log"
: >"$log"
trap 'status=$?; printf "Build failed (exit %s). See %s\n" "$status" "$log" >&2; tail -n 40 "$log" >&2; exit "$status"' ERR

printf 'Building FlashAttention for ROCm %s / gfx1100. Log: %s\n' "$sdk_version" "$log"
"$uv" pip install --python "$python" wheel ninja >>"$log" 2>&1
"$uv" pip install --python "$python" "rocm-sdk-devel==$sdk_version" \
  --index https://nightly.repo.amd.com/rocm/whl-next/ \
  --default-index https://pypi.org/simple >>"$log" 2>&1
"$venv_dir/bin/rocm-sdk" init >>"$log" 2>&1

rocm_root="$site_packages/_rocm_sdk_core"
devel_root="$site_packages/_rocm_sdk_devel"
[[ -d "$rocm_root" && -d "$devel_root/include" && -d "$devel_root/lib" ]]
export ROCM_HOME="$rocm_root" ROCM_PATH="$rocm_root" HIP_PATH="$rocm_root"
export HIP_CLANG_PATH="$rocm_root/lib/llvm/bin" HIP_INCLUDE_PATH="$rocm_root/include"
export HIP_LIB_PATH="$rocm_root/lib" HIP_DEVICE_LIB_PATH="$rocm_root/lib/llvm/amdgcn/bitcode"
export CPATH="$devel_root/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$devel_root/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export PATH="$devel_root/bin:$rocm_root/bin:$rocm_root/lib/llvm/bin:$venv_dir/bin:$PATH"
export GPU_ARCHS=gfx1100 BUILD_TARGET=rocm FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS="${MAX_JOBS:-16}"

if [[ -d "$source_dir/build" ]]; then
  backup="$source_dir/agent_space/build-before-$(date +%Y%m%d-%H%M%S)"
  mv -- "$source_dir/build" "$backup"
  printf 'Preserved previous build artifacts: %s\n' "$backup"
fi

cd "$source_dir"
"$uv" pip install --python "$python" --no-build-isolation \
  --reinstall-package flash-attn . >>"$log" 2>&1
cd "$app_dir"
"$python" - <<'PY' >>"$log" 2>&1
import torch
import flash_attn
import flash_attn_2_cuda
from flash_attn import flash_attn_func

q, k, v = (torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.float16, requires_grad=True) for _ in range(3))
out = flash_attn_func(q, k, v)
assert torch.isfinite(out).all()
out.float().sum().backward()
assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
print("FlashAttention GPU forward/backward OK:", flash_attn.__version__, torch.cuda.get_device_name(0))
PY
printf 'Build and GPU test passed. Log: %s\n' "$log"
