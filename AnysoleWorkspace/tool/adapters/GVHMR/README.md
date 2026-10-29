# GVHMR SIGGRAPH Asia 2024 baseline

This adapter targets the original [GVHMR SIGGRAPH Asia 2024 project](https://zju3dv.github.io/gvhmr/) and the official `gvhmr_siga24_release.ckpt`. Its upstream checkout is pinned to `zju3dv/GVHMR` commit `44fbc1999dd57d2180e882c62c78d356d4c840d0` (2024-09-19). It excludes later SimpleVO/focal-length additions and the TPAMI 2026 extension.

## Input and outputs

The dataset input is read-only `1_Dataset/<date>/RGB/<method>/<session>/camera_<N>/frame*.jpg`; target person boxes come from the matching date's `annotations/records.parquet`. The adapter also accepts an explicit video with a per-frame target-box sidecar. A dataset unit is one date/method/session/camera sequence. Source frame IDs and physical timestamps are retained for evaluation against MoCap GT′.

Each completed unit publishes validated `result.npz`, `result.pkl`, `overlay.mp4`, and `state.json`. Both result files include native GVHMR 66-value SMPL-X body poses and converted 72-value neutral-SMPL poses, separate camera/global translations, betas, intrinsics, source-frame IDs, and SMPL fit residuals. The pickle is a Python-readable dictionary with top-level `pose`, `betas`, and `trans` arrays; the NPZ remains the canonical validated numeric artifact. Complete dated runs use the precedent short directory names such as `20260419_m1_S1011_c1` and `20260419_m2_S1011_c1`. The overlay draws the annotated target box and fitted SMPL skeleton. The world-frame GVHMR trajectory has its own sequence gauge; world metrics require an explicitly validated coordinate transform. The camera-frame evaluator uses the existing GT′ timing and geometry conventions.

## Assets and runtime

- Inference source: `third_party/GVHMR/`, including its pinned DPVO submodule.
- Public 2024 checkpoints: `models/gvhmr/checkpoints/`, with sizes and SHA-256 in `asset_manifest.json` there. The upstream `inputs/checkpoints` path is an ignored local symlink to this directory.
- Existing licensed neutral SMPL: `models/smpl/SMPL_NEUTRAL.pkl` (a symlink to the 86,955,547-byte chumpy-free model).
- Required licensed SMPL-X: user-provided `models/smplx/SMPLX_NEUTRAL.npz` (the adapter default). The official installation instructions require obtaining it through the SMPL-X registration site. Do not commit or redistribute this asset. The adapter makes a temporary compatible alias for the pinned 2024 model loader.
- Runtime: isolated `/opt/conda/envs/GVHMR` using official 2024 Python 3.10 / Torch 2.3 / CUDA 12.1 versions. The existing Tactile prefix has incompatible Python/Torch/PyTorch3D versions and stays dedicated to the other baselines.

The tested prefix additionally has a prefix-local DPVO native build for moving-camera mode. [setup_environment.sh](setup_environment.sh) records the installation recipe (including the legacy `chumpy` workaround, `libGL.so.1`, PyAV 13.1.0 compatibility pin, and DPVO build toolchain); it refuses to overwrite an existing prefix. Its syntax has been checked, while a clean second installation has not been performed. The live prefix passed dependency consistency, key imports, CUDA tensor operations, and DPVO extension/GPU-operation smoke checks. A real eight-frame end-to-end RGB→SMPL/overlay run and a separate matched-GT camera evaluation also passed.

To re-verify or resume the five public checkpoint downloads, run `/opt/conda/bin/conda run --prefix /opt/conda/envs/Tactile python /private/Tactile/models/gvhmr/fetch_drive.py`. The fetcher checks the pinned manifest before skipping files; it does not supply the licensed body model.

## Execution boundary

Bounded tests must use a temporary root outside `3_Result/`. The separately user-authorized all-method run uses only `3_Result/GVHMR_baseline/`. The seven-GPU pipeline refuses complete manifests without `--full --authorization user-approved`, and refuses `3_Result/` for bounded tests.

## Bounded operator workflow

Run from `2_Code/` and keep every output under a task-owned temporary root, not `3_Result/`. `prepare` reads source RGB and target-box annotations without modifying the dataset; `run` launches the isolated GVHMR environment for a single named unit. The example intentionally stops at eight frames:

```bash
cd /private/Tactile/2_Code
/opt/conda/bin/conda run --prefix /opt/conda/envs/Tactile python -m gvhmr_baseline.cli prepare \
  --date 20260808 --method method_1_offset --session S11073 --camera camera_4 \
  --max-units 1 --max-frames 8 --output-root /private/Tactile/.ai/tmp/gvhmr_operator_smoke
/opt/conda/bin/conda run --prefix /opt/conda/envs/Tactile python -m gvhmr_baseline.cli run \
  --unit-id 20260808/method_1_offset/S11073/camera_4 --max-frames 8 \
  --output-root /private/Tactile/.ai/tmp/gvhmr_operator_smoke
/opt/conda/bin/conda run --prefix /opt/conda/envs/Tactile python -m gvhmr_baseline.cli audit \
  --output-root /private/Tactile/.ai/tmp/gvhmr_operator_smoke
```

The default `run` mode is fixed-camera (`static_cam=True`), matching these stationary camera units. `--moving-camera` selects official DPVO and requires its compiled extension plus the pinned `dpvo.pth`. Run refuses a missing or truncated asset; provide another licensed neutral SMPL-X archive with `--smplx-model /path/to/SMPLX_NEUTRAL.npz` if needed. The source image/video timestamps are retained even though the model's staged input video is encoded at its expected 30 FPS. To retry a recorded failure after fixing an asset, pass `--retry`; do not use `--force` unless replacing an already successful bounded output is intentional.

## Seven-GPU automatic pipeline

`pipeline-run` launches `7 × n` fixed-affinity slots over physical GPUs 0–6. Each slot runs its assigned units sequentially; each unit performs official 2024 inference, SMPL conversion, overlay rendering, available GT′ camera evaluation, checksum publication, and an integrity audit. It records per-unit terminal states and per-GPU sampled utilization/peak memory under `.run/`. Re-running the same manifest/model/config skips audited successful units and retries incomplete ones. A different manifest or model requires a new output root. `pipeline-plan`, `pipeline-status`, and `pipeline-audit` are read-only inspection commands.

For a bounded test only, first use `prepare` with `--max-units` and `--max-frames`, then:

```bash
cd /private/Tactile/2_Code
/opt/conda/bin/conda run --prefix /opt/conda/envs/Tactile python -m gvhmr_baseline pipeline-plan \
  --output-root /private/Tactile/.ai/tmp/gvhmr_operator_smoke
/opt/conda/bin/conda run --prefix /opt/conda/envs/Tactile python -m gvhmr_baseline pipeline-run \
  --output-root /private/Tactile/.ai/tmp/gvhmr_operator_smoke
/opt/conda/bin/conda run --prefix /opt/conda/envs/Tactile python -m gvhmr_baseline pipeline-audit \
  --output-root /private/Tactile/.ai/tmp/gvhmr_operator_smoke
```

`pipeline-run` defaults to the measured `n=4` (28 slots) in `pipeline.py`. On seven L20 GPUs, 16-frame bounded trials rose from 2.51 frames/s (`n=1`) to 7.72 frames/s (`n=4`). On the same 28 × 64-frame manifest, `n=4` completed in 94 s versus 145 s for `n=3`; all 28 units passed GT′ evaluation and artifact audit. A further 28 × 128-frame `n=4` run completed 3,584 frames in 151 s with zero failed units, maximum sampled GPU memory 24,313 MiB and peak container memory 237.5 GB under a 256-GiB limit. `n=5` is disallowed because the measured per-worker host-RAM footprint projects beyond that limit; runtime preflight and a live 8-GiB reserve guard remain in place. These measurements are bounded, not a promise for every full-length sequence. Benchmark outputs are under `.ai/tmp/gvhmr_benchmark_n*/`.

For the authorized complete dataset, `prepare --indexed-inventory ... --annotation-index ...` verifies the prior complete WHAM JSONL inventory, current RGB directory and filename coverage, annotation source hashes, target boxes, consecutive frame IDs, and camera dimensions. This avoids decoding all 1.4 million JPEGs before launch; every JPEG is decoded later by the inference-video writer. Then run `bash 2_Code/gvhmr_baseline/launch_full.sh user-approved` from the workspace root. The launcher creates a detached supervisor under `3_Result/GVHMR_baseline/.run/`, starts at 7×4, reduces `n` if the live cgroup-memory guard triggers, and resumes incomplete units; one ordinary failure retry is bounded. Inspect `.run/supervisor.json`, `.run/supervisor.log`, `.run/pipeline_run.json`, and `pipeline-status` while it runs. Do not launch a second supervisor concurrently.

An explicit video is supported with `prepare-video --video ... --boxes-npy ... --fps ...`, where boxes are a per-frame `[F,4]` raw-pixel `xyxy` NumPy array. A real eight-frame temporary MP4 and box sidecar completed GVHMR→SMPL, overlay and audit; its terminal records `skipped_no_gt_prime`. This mode has no GT′ metadata and cannot use the date-specific evaluator. `evaluate --unit-id ...` applies only to a dated RGB sequence with GT′; it writes camera-frame metrics beside the result. It does not report unvalidated checkerboard/world-frame scores.

Environment build/test evidence and the exact follow-up handoff are recorded in the task record under `.ai/`, which stays outside Git.
