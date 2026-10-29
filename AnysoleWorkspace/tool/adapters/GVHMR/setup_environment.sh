#!/usr/bin/env bash
# Recreate the isolated GVHMR SIGGRAPH Asia 2024 runtime. This script never
# modifies the existing Tactile prefix and refuses to overwrite any prefix.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd -- "$script_dir/../.." && pwd)"
upstream_root="$workspace_root/third_party/GVHMR"
target_prefix="${1:-/opt/conda/envs/GVHMR}"
conda_bin=/opt/conda/bin/conda

if [[ -e "$target_prefix" ]]; then
  echo "Refusing to replace an existing Conda prefix: $target_prefix" >&2
  exit 2
fi
if [[ ! -f "$upstream_root/requirements.txt" || ! -f "$upstream_root/third-party/DPVO/setup.py" ]]; then
  echo "Pinned 2024 GVHMR/DPVO source is missing under $upstream_root" >&2
  exit 2
fi
expected_revision=44fbc1999dd57d2180e882c62c78d356d4c840d0
observed_revision="$(git -c "safe.directory=$upstream_root" -C "$upstream_root" rev-parse HEAD)"
if [[ "$observed_revision" != "$expected_revision" ]]; then
  echo "GVHMR source revision $observed_revision is not pinned 2024 revision $expected_revision" >&2
  exit 2
fi
if [[ ! -x /usr/local/cuda-12.4/bin/nvcc ]]; then
  echo "DPVO build needs the tested /usr/local/cuda-12.4/bin/nvcc toolchain" >&2
  exit 2
fi

"$conda_bin" create --yes --prefix "$target_prefix" python=3.10 pip
"$conda_bin" run --prefix "$target_prefix" python -m pip install setuptools==80.9.0

# chumpy 0.70 has a legacy setup hook that imports pip and therefore must be
# installed separately without PEP-517 build isolation.
requirements_without_chumpy="$(mktemp)"
trap 'rm -f -- "$requirements_without_chumpy"' EXIT
sed '/^[[:space:]]*chumpy[[:space:]]*$/d' "$upstream_root/requirements.txt" > "$requirements_without_chumpy"
"$conda_bin" run --prefix "$target_prefix" python -m pip install -r "$requirements_without_chumpy"
"$conda_bin" run --prefix "$target_prefix" python -m pip install --no-build-isolation chumpy==0.70
# The 2024-pinned ImageIO 2.34.1 calls VideoStream.close(); PyAV 17 removed
# that interface. PyAV 13.1.0 passed the real eight-frame decoder smoke.
"$conda_bin" run --prefix "$target_prefix" python -m pip install av==13.1.0
"$conda_bin" install --yes --prefix "$target_prefix" --channel conda-forge libgl
"$conda_bin" run --prefix "$target_prefix" python -m pip install \
  setuptools==80.9.0 pyarrow==21.0.0 pytest==9.1.1

# The pinned DPVO source is copied into the prefix so editable compilation
# cannot change the Git-managed GVHMR checkout.
"$conda_bin" run --prefix "$target_prefix" python -m pip install \
  'https://data.pyg.org/whl/torch-2.3.0+cu121/torch_scatter-2.1.2%2Bpt23cu121-cp310-cp310-linux_x86_64.whl' \
  numba==0.67.0 pypose==0.9.5
mkdir -p -- "$target_prefix/src"
cp -a -- "$upstream_root/third-party/DPVO" "$target_prefix/src/DPVO"
env CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 \
  "$conda_bin" run --prefix "$target_prefix" python -m pip install \
  --no-deps --no-build-isolation --editable "$target_prefix/src/DPVO"

"$conda_bin" run --prefix "$target_prefix" python -m pip check
echo "GVHMR 2024 runtime installed at $target_prefix"
echo "Licensed SMPLX_NEUTRAL.npz is still required for model construction and inference."
