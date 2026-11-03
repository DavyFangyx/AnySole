"""Evaluate AnySole with the canonical session-level metric protocol."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from anysole.data.dataset import AnySoleDataset, collate_windows, load_split_ids
from anysole.data.smpl_io import pelvis_to_smpl_trans, smpl24_pose6d_to_poses, smpl_archive_metadata
from anysole.utils.diffusion import GaussianDiffusion
from anysole.utils.geometry import f2_to_world, fk_pose6d, positions_to_6d_np, rot6d_to_rotmat, rot6d_to_rotmat_np, rotmat_to_6d, rotmat_to_6d_np
from anysole.utils.geometry import compose_rot6d, rot_root6d
from anysole.utils.constraints import apply_v3e_constraints
from anysole.utils.constraints_g import apply_v3g_constraints
from anysole.utils.constraints_h import apply_v3h
from anysole.models.common.model_v3h import apply_foot_residual
from anysole.models.common.model_v3f import sample_v3f
from anysole.models.common.model_v5_1 import v5_1_z_src
from anysole.models import AnySoleModel, MODEL_ANYSOLEV1, MODEL_ANYSOLEV1_INSOLE_DRIFT, MODEL_ANYSOLEV1_POS, MODEL_ANYSOLEV2, MODEL_NAMES
from anysole.train import condition_inputs, load_config, move_batch, resolve_device
from anysole.ablations.insole_drift.templates import load_template_bank
from anysole.registry import get_builder, infer_anysole_paths
from anysole.types import (
    ANYSOLE_ROOT,
    CONFIG_MODE_NAMES,
    CONFIG_T,
    CONFIG_V,
    CONFIG_VT,
    DEFAULT_CONFIG_PATH,
    FPS,
    GAIT_ROOT,
    JOINT_PROTOCOL_CHECKSUM,
    JOINT_NAMES,
    MOTION_PROTOCOL,
    N_JOINTS,
    POSE_DIM,
    assert_batch_shapes,
)

# Quasi-strict load allowance (fix_plan_v3 §3.1): state-dict keys added as
# code hygiene AFTER the baseline checkpoints were trained.  Only these may
# be missing when loading F4a/F2p4-era checkpoints.
_MISSING_OK = frozenset({
    "encoders.t_enc.stream_norm.weight",
    "encoders.t_enc.stream_norm.bias",
})


def _smpl_yup_to_display(points: np.ndarray) -> np.ndarray:
    """Convert native SMPL (+Y up) joints to the display/world (+Z up) frame."""
    points = np.asarray(points, dtype=np.float32)
    return np.stack((points[..., 0], -points[..., 2], points[..., 1]), axis=-1)


def _pack_motion_export(session: dict, windows: list[tuple], pos_mode: bool,
                        overlap_blend: bool = False) -> dict:
    """Place window predictions back on the original session frame grid.

    Dataset windows are intentionally skipped when they contain fake frames.
    Concatenating those windows would silently shift every later prediction in
    time.  The unified archive therefore always has the original ``n_frames``
    length and carries ``valid_mask``/``frame_indices`` explicitly.

    ``overlap_blend``（修改 02，2026-11-03）：重叠窗先按三角权重混合
    （协议同口径 blend_overlap_windows_np），再散回帧网格——npz 与指标
    同源；单覆盖帧直接拷贝，stride=tw 时与现导出逐位一致。
    """
    n_frames = int(np.asarray(session["V_feat"]).shape[0])
    pose_dim = int(windows[0][1].shape[-1]) if windows else POSE_DIM
    identity = np.tile(
        np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        N_JOINTS,
    )
    pose = np.repeat(identity[None], n_frames, axis=0)
    if pose_dim != pose.shape[1]:
        pose = np.zeros((n_frames, pose_dim), dtype=np.float32)
    trans = np.zeros((n_frames, 3), dtype=np.float32)
    seen = np.zeros((n_frames,), dtype=bool)
    ordered = sorted(windows, key=lambda item: int(item[0]))

    if overlap_blend and ordered:
        from anysole.utils.eval_protocol import blend_overlap_windows_np
        pose_b, trans_b, frames_b = blend_overlap_windows_np(
            [(int(s), np.asarray(p), np.asarray(t), np.asarray(g))
             for s, p, t, g in ordered], tw=ordered[0][1].shape[0])
        for t, f in enumerate(frames_b):
            if 0 <= f < n_frames:
                pose[f] = pose_b[t]
                trans[f] = trans_b[t]
                seen[f] = True
    else:
        # Crossfade only across genuinely overlapping time ranges (continuation
        # evals), blending same-time frames.  Back-to-back or gapped windows are
        # stitched directly: the historical 4-frame crossfade blended predictions
        # of different frames, which amplified per-window root-orientation
        # differences (e.g. T2M heading flips) after 6D re-orthonormalization.
        fade = 0 if pos_mode else 4
        if fade > 0:
            for previous, current in zip(ordered, ordered[1:]):
                previous_start = int(previous[0])
                current_start = int(current[0])
                previous_len = int(np.asarray(previous[1]).shape[0])
                current_len = int(np.asarray(current[1]).shape[0])
                if current_start >= previous_start + previous_len:
                    continue  # back-to-back or gapped: nothing to blend
                overlap = previous_start + previous_len - current_start
                fade_n = min(fade, overlap)
                if fade_n < 1:
                    continue
                previous_pose = np.asarray(previous[1])
                current_pose = np.asarray(current[1])
                previous_trans = np.asarray(previous[2])
                current_trans = np.asarray(current[2])
                for j in range(fade_n):
                    alpha = (j + 1) / (fade_n + 1)
                    a = previous_len - fade_n + j
                    b = j + overlap - fade_n
                    if b < 0 or b >= current_len:
                        continue
                    old_a, old_b = previous_pose[a].copy(), current_pose[b].copy()
                    previous_pose[a] = (1 - alpha) * old_a + alpha * old_b
                    current_pose[b] = (1 - alpha) * old_b + alpha * old_a
                    old_a, old_b = previous_trans[a].copy(), current_trans[b].copy()
                    previous_trans[a] = (1 - alpha) * old_a + alpha * old_b
                    current_trans[b] = (1 - alpha) * old_b + alpha * old_a

        for start, window_pose, window_trans, _window_gt_trans in ordered:
            start = int(start)
            window_pose = np.asarray(window_pose, dtype=np.float32)
            window_trans = np.asarray(window_trans, dtype=np.float32)
            if start >= n_frames:
                continue
            length = min(len(window_pose), len(window_trans), n_frames - start)
            if length <= 0:
                continue
            pose[start:start + length] = window_pose[:length]
            trans[start:start + length] = window_trans[:length]
            seen[start:start + length] = True

    fake_mask = np.asarray(
        session.get("fake_mask", np.zeros(n_frames, dtype=np.uint8))
    ).reshape(-1)
    valid = seen.copy()
    if len(fake_mask) == n_frames:
        valid &= fake_mask == 0

    pose_tensor = torch.from_numpy(pose).float().unsqueeze(0)
    trans_tensor = torch.from_numpy(trans).float().unsqueeze(0)
    offsets = torch.from_numpy(np.asarray(session["offsets"])).float()
    parents = torch.from_numpy(np.asarray(session["parents"])).long()
    joints_native = fk_pose6d(pose_tensor, trans_tensor, offsets, parents)[0].numpy()
    return {
        "pose": pose,
        "trans": trans,
        "gt_trans": np.asarray(session["trans_global"], dtype=np.float32),
        "valid": valid,
        "joint_xyz_world": _smpl_yup_to_display(joints_native),
    }


def _tactile_corr(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-frame Pearson correlation over the 96 pressure cells.

    Returns (correlation, valid): frames where either side is constant
    (std ~ 0, e.g. a fully off insole) have no defined correlation and are
    masked out instead of contributing garbage values.
    """
    pred_c = pred - pred.mean(dim=-1, keepdim=True)
    target_c = target - target.mean(dim=-1, keepdim=True)
    denom = (pred_c.square().sum(dim=-1) * target_c.square().sum(dim=-1)).sqrt()
    corr = (pred_c * target_c).sum(dim=-1) / denom.clamp_min(1.0e-8)
    return corr, denom > 1.0e-8


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AnySole V1 with DDIM sampling.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Checkpoint .pt. Omitted when --model-name and --contact-method name a model dir "
        "under results/AnySole (checkpoints/ckpt_last.pt is used).",
    )
    parser.add_argument("--modal", choices=MODEL_NAMES, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--model-name", default=None,
        help="Registered model identifier (e.g. V4A / V3_4a). The checkpoint "
        "architecture is read from the checkpoint config.",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Stacked hyperparameter fields under the model dir "
        "(the full field stack, e.g. tw40_st40_lr0.0001_lp3_lt1_lk1_wu0.05_gc5_"
        "ep740_bs256_sd1); model-name+contact+variant "
        "resolves to <model dir>/<variant>/checkpoints/ckpt_last.pt — the "
        "same address --ckpt would name (2026-09-22 stacked naming).",
    )
    parser.add_argument(
        "--which",
        choices=("last", "best"),
        default="last",
        help="Which checkpoint to address when inferred from --model-name. "
        "last = ckpt_last (end of every run); best = ckpt_best (val-metric-best "
        "intermediate, may be several epochs older — use for final reports).",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="With --model-name: evaluate every variant subdir under "
        "results/AnySole/<model>_*/ (any contact method); --contact-method and "
        "--split are ignored (split defaults to test).",
    )
    parser.add_argument(
        "--contact-method",
        default=None,
        help="AnySole-private contact-label scheme for contact_gt, and the model-dir "
        "suffix when --ckpt is omitted. Defaults to the checkpoint's explicit method.",
    )
    parser.add_argument("--smpl-support-policy", choices=("exclude", "legacy_clip"), default=None,
                        help="Explicit evaluation support policy; default follows checkpoint, legacy for old checkpoints")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit-sessions", type=int, default=None)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--config-id",
        default=None,
        metavar="MODE[,MODE...]",
        help="Evaluation mode(s): VT2M, V2M, T2M; omitted means all three.",
    )
    parser.add_argument(
        "--z-src",
        default=None,
        choices=("V", "VT", "T"),
        metavar="Z",
        help="V5_Av1 判据 ③ T 参与干预：VT 行的 z 来源覆写（V = 禁 T 专家，"
             "z 固定为 g_V 输出；T = 反向对照）。常规跑不加此旗标；两次跑"
             "（常规 vs --z-src V）的 VT2M 行逐会话配对比较 = 干预实验。",
    )
    parser.add_argument(
        "--null-t",
        action="store_true",
        help="V4_Av1 判据 ③ T 参与干预（null-T 注入）：三条件前向的 T 输入"
             "整窗置零；训练正确的修正分支应输出 ≈0 → VT2M 行读数退回"
             "V2M（两次跑——常规 vs --null-t——的 VT2M 行逐会话配对比较"
             "= 干预实验）。修改 03（2026-11-03）起对 V7 家族同样生效"
             "（T2M + null-T = 无条件先验读数）；其余模型忽略（打印提示）。",
    )
    parser.add_argument(
        "--write-motion",
        type=Path,
        default=None,
        metavar="DIR",
        help="Standard SMPL motion NPZ export dir (default: <model>/predictions/eval_motion).",
    )
    parser.add_argument("--no-write-motion", action="store_true", help="Disable SMPL motion export.")
    parser.add_argument("--metrics-out", type=Path, default=None, metavar="FILE",
                        help="Write the whole-body brief metrics (T2M/V2M/VT2M/V2T) to this JSON "
                        "path (default: <model>/metrics/<split>_brief.json).")
    parser.add_argument(
        "--protocol-out",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write the full protocol detail metrics (per-part + robustness) to this JSON "
        "path. Default: <model>/metrics/<split>.json.",
    )
    parser.add_argument(
        "--protocol-seed",
        type=int,
        default=0,
        help="F0a: RNG seed for the protocol pass (matters for diffusion sampling). "
        "Sigma-seed needs three runs with seeds 0/1/2.",
    )
    parser.add_argument(
        "--no-robustness",
        action="store_true",
        help="F0a: skip the protocol robustness rows (contiguous 20-40%% V / T frame "
        "dropout under VT2M).",
    )
    parser.add_argument(
        "--overlap-stride",
        type=int,
        default=None,
        metavar="N",
        help="修改 02（2026-11-03，接缝归因与重叠拼接推理）：声明口径的重叠窗评估。"
        "N ∈ [1, tw]（默认 None = 现协议，行为逐位不变）；N < tw 时协议与导出改走"
        "三角权重混合组装（锚点网格保持原非重叠网格，n_gt_anchors_used 与现协议"
        "逐位相同）；N = tw 为等价性对照（新代码路径须与现协议逐位一致）。",
    )
    parser.add_argument(
        "--dump-detail",
        action="store_true",
        help="Also write per-window and per-session metric detail JSONs next to the "
        "protocol file (window_detail_<split>.json / session_detail_<split>.json).",
    )
    return parser.parse_args(argv)


def _parse_config_values(value: Optional[str]) -> List[int]:
    mode_to_config = dict(zip(CONFIG_MODE_NAMES, (CONFIG_VT, CONFIG_V, CONFIG_T)))
    if value is None:
        return [CONFIG_VT, CONFIG_V, CONFIG_T]
    modes = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not modes:
        raise ValueError("--config-id must contain at least one mode")
    unknown = [item for item in modes if item not in mode_to_config]
    if unknown:
        raise ValueError("Unknown mode(s) %s; choose from VT2M,V2M,T2M" % ",".join(unknown))
    return list(dict.fromkeys(mode_to_config[item] for item in modes))


def _sweep_targets(model_name_csv: str, which: str = "last") -> List[tuple[Optional[str], Optional[str], Path]]:
    """--sweep: every variant ckpt under results/AnySole/<model>_*/.

    Contact method is read from the dir name (<model>_<contact>) and
    --contact-method is bypassed; variant subdirs without the requested
    checkpoint are skipped with a warning.  Returns (modal, contact_method,
    ckpt) triples sorted by path.
    """
    anysole_root = Path(os.environ.get("ANYSOLE_RESULTS", str(GAIT_ROOT / "results"))) / "AnySole"
    ckpt_name = "ckpt_last.pt" if which == "last" else "ckpt_best.pt"
    targets: List[tuple[Optional[str], Optional[str], Path]] = []
    for model_name in [item.strip() for item in model_name_csv.split(",") if item.strip()]:
        for contact_dir in sorted(
            p for p in anysole_root.glob("%s_*" % model_name) if p.is_dir()
        ):
            contact = contact_dir.name[len(model_name) + 1:]
            for variant_dir in sorted(p for p in contact_dir.iterdir() if p.is_dir()):
                ckpt = variant_dir / "checkpoints" / ckpt_name
                if ckpt.is_file():
                    targets.append((None, contact, ckpt))
                else:
                    print("sweep %s/%s: no %s, skipped"
                          % (contact_dir.name, variant_dir.name, ckpt_name))
    if not targets:
        raise ValueError(
            "--sweep found no %s checkpoints under results/AnySole/%s_*"
            % (ckpt_name, model_name_csv)
        )
    return targets


def _discover_model_targets(
    modal: Optional[str], contact_method: Optional[str]
) -> List[tuple[str, str, Path]]:
    """Scan results/AnySole for model dirs named <modal>_<contact_method>.

    Returns (modal, contact_method, ckpt) triples sorted by dir name, so a
    bare ``python -m anysole.eval`` re-evaluates every trained model.
    Longest modal names match first (``anysolev1_insole_drift_tactile_abs``
    is the drift modal plus ``tactile_abs``, not ``anysolev1`` plus a long
    contact method), and dirs without ``checkpoints/ckpt_last.pt`` are
    skipped with a warning.
    """
    anysole_root = Path(os.environ.get("ANYSOLE_RESULTS", str(GAIT_ROOT / "results"))) / "AnySole"
    targets: List[tuple[str, str, Path]] = []
    if not anysole_root.is_dir():
        return targets
    for child in sorted(anysole_root.iterdir()):
        if not child.is_dir():
            continue
        for name in sorted(MODEL_NAMES, key=len, reverse=True):
            prefix = name + "_"
            if not (child.name.startswith(prefix) and len(child.name) > len(prefix)):
                continue
            contact = child.name[len(prefix):]
            if (modal is not None and name != modal) or (
                contact_method is not None and contact != contact_method
            ):
                break
            ckpt = child / "checkpoints" / "ckpt_last.pt"
            if ckpt.is_file():                targets.append((name, contact, ckpt))
            else:
                print("skip %s: missing %s" % (child.name, ckpt))
            break
    return targets


def _load_model(checkpoint: dict, config: dict, device: torch.device) -> AnySoleModel:
    saved_config = checkpoint.get("config", {})
    if not isinstance(saved_config, dict):
        raise RuntimeError("checkpoint config is missing or is not a mapping")
    state_pose_dim = getattr(checkpoint.get("model", {}).get("pose_head.pose_mean"), "shape", (POSE_DIM,))[0]
    saved_joints = int(saved_config.get("motion_n_joints", state_pose_dim // 6 if state_pose_dim else N_JOINTS))
    saved_protocol = str(saved_config.get("motion_protocol", ""))
    saved_checksum = str(saved_config.get("joint_protocol_checksum", ""))
    if (saved_joints != N_JOINTS or state_pose_dim != POSE_DIM
            or saved_protocol != MOTION_PROTOCOL
            or saved_checksum != JOINT_PROTOCOL_CHECKSUM):
        raise RuntimeError(
            "checkpoint motion protocol is incompatible with the current native SMPL-24 "
            "joint tree (joints=%s, pose_dim=%s, protocol=%r, joint checksum=%s). "
            "This includes BVH-23 checkpoints and early 24/144 SMPL checkpoints trained "
            "with the incorrect collar-parent tree. Retrain under the current protocol; "
            "an old checkpoint may only be used through --init-from, which resets its "
            "pose/traj layers."
            % (saved_joints, state_pose_dim, saved_protocol,
               saved_checksum[:12] if saved_checksum else "missing")
        )
    d_model = int(saved_config.get("d_model", config["d_model"]))
    # tw always comes from the checkpoint, so evaluating an older model stays
    # correct even after anysole/configs/v1.yaml has moved on.
    tw = int(saved_config.get("tw", config["tw"]))
    modal = str(saved_config.get("modal", MODEL_ANYSOLEV1))
    if modal not in MODEL_NAMES:
        raise ValueError("Unknown checkpoint modal %r" % modal)
    use_drift = modal == MODEL_ANYSOLEV1_INSOLE_DRIFT
    model_kw = {}
    if use_drift:
        template_path = saved_config.get("template_path", config.get("template_path"))
        if not template_path:
            raise ValueError("template_path is required for drift-enabled evaluation")
        templates, subject_map = load_template_bank(template_path)
        model_kw.update(templates=templates, subject_to_index=subject_map)
    dropout = float(saved_config.get("dropout", config.get("dropout", 0.1)))
    pose_layers = int(saved_config.get("pose_layers", 6))
    # F5 part9 is archived (fix_plan_v3.md supersedes v2 §F5): its checkpoints
    # are kept on disk but the V3 codebase does not load them.
    if str(saved_config.get("decoder", "v1")) == "part9":
        raise ValueError(
            "F5 part9 checkpoint is archived (v2 §F5 作废，fix_plan_v3 取代)；"
            "V3 代码不回载。基线用 F4a_footconv / F2p4_combo。"
        )
    if modal == MODEL_ANYSOLEV2:
        # Registered models are rebuilt from their entry script — the
        # structure is fixed by the recorded model_name, and the saved
        # structure flags are validated against the entry (2026-09-24
        # independence restructure; no warm-start, no flag reconstruction).
        model_name = str(saved_config.get("model_name", "") or "")
        if not model_name:
            raise RuntimeError(
                "checkpoint config has no model_name; registered-model "
                "checkpoints must record it (train --model-name writes it)"
            )
        entry = get_builder(model_name)
        # 2026-11-01 回填：Ctx4/Ctx5 旧 ckpt 的 STRUCTURE 曾漏 v7_ctx3
        # （隐含键——模型 build 侧 ctx3 已生效，权重语义正确；entry 补键
        # 后旧 ckpt 校验不匹配）。回填后再比较。
        if saved_config.get("v7_ctx5") and saved_config.get("v7_ctx3") is None:
            saved_config["v7_ctx3"] = True
        if saved_config.get("v7_ctx4") and saved_config.get("v7_ctx3") is None:
            saved_config["v7_ctx3"] = True
        saved_struct = {k: saved_config.get(k) for k in entry.STRUCTURE}
        if saved_config.get("part_joints") is not None and "pose_parts" in saved_struct:
            # V4B: pose_parts is derived from the partition at build time.
            saved_struct["pose_parts"] = len(saved_config["part_joints"])
        if saved_struct != dict(entry.STRUCTURE):
            raise RuntimeError(
                "checkpoint structure does not match the %s entry: saved=%s entry=%s"
                % (model_name, saved_struct, dict(entry.STRUCTURE))
            )
        for key, value in getattr(entry, "CONFIG_EXTRA", {}).items():
            if saved_config.get(key) != value:
                raise RuntimeError(
                    "checkpoint flag %s=%r does not match the %s entry (%r)"
                    % (key, saved_config.get(key), model_name, value)
                )
        build_cfg = dict(config)
        # 结构标志进 build_cfg（2026-10-25 修订）：V7 的
        # v7_contact_root/v7_reliability_route 决定模型构造（route_mlp 的
        # 存在与否）——此前只校验了 ckpt 与 entry 的结构一致，却没有把
        # 结构键传给 build，C 类模型的 state_dict 装载必然 mismatch。
        build_cfg.update(entry.STRUCTURE)
        build_cfg["d_model"] = d_model
        build_cfg["tw"] = tw
        build_cfg["obs_context"] = int(saved_config.get("obs_context", tw))
        build_cfg["dropout"] = dropout
        build_cfg["pose_layers"] = pose_layers
        # V3_Fv1: the denoiser depth is a hyperparameter — read it from the
        # checkpoint like pose_layers (the yaml config only carries the
        # default).
        build_cfg["denoise_layers"] = int(saved_config.get("denoise_layers", 4))
        # V3_Gv1: the frozen carrier is structural provenance recorded in the
        # ckpt config at training time (2026-10-18 incident: building from the
        # yaml config dropped it and the formal eval died on the missing key).
        if saved_config.get("carrier_from"):
            build_cfg["carrier_from"] = str(saved_config["carrier_from"])
        if saved_config.get("g_floor_src"):
            build_cfg["g_floor_src"] = str(saved_config["g_floor_src"])
        # model_name is structural provenance for entry-based builds
        # (2026-10-19 V3_Hv2 incident: V3_Hv2/v3 heads are selected by
        # the model_name prefix — dropping it built a headless model and the
        # strict load died on unexpected keys).
        if saved_config.get("model_name"):
            build_cfg["model_name"] = str(saved_config["model_name"])
        # V3_Bv2/Bv3: the query pressure representation and its frozen
        # train-cell-mean prior are structural init values recorded in the
        # ckpt config at training time (same class as carrier_from above).
        if saved_config.get("pressure_head"):
            build_cfg["pressure_head"] = str(saved_config["pressure_head"])
        if saved_config.get("pressure_cell_mean"):
            build_cfg["pressure_cell_mean"] = saved_config["pressure_cell_mean"]
        # 维度 2/3 结构旗标（2026-10-21）：T_phys 12→20 与 d8 三头是
        # 训练时结构 provenance，必须从 ckpt config 回放（yaml 缺省 False）。
        build_cfg["t_phys_v2"] = bool(saved_config.get("t_phys_v2", False))
        build_cfg["d8_heads"] = bool(saved_config.get("d8_heads", False))
        # 修改 04 改动 B（2026-11-03）：低维反馈是训练时结构 provenance
        # （ar2_fb 输入维 = 10+k vs 148），必须从 ckpt config 回放。
        if saved_config.get("v7_ar2_lowdim"):
            build_cfg["v7_ar2_lowdim"] = True
        if saved_config.get("v7_ar2_fb_dim") is not None:
            build_cfg["v7_ar2_fb_dim"] = int(saved_config["v7_ar2_fb_dim"])
        part_joints = saved_config.get("part_joints")
        model = entry.build(
            build_cfg,
            part_joints=tuple(tuple(int(j) for j in g) for g in part_joints)
            if part_joints else None,
        ).to(device)
    else:
        model = AnySoleModel(
            d=d_model, tw=tw, modal=modal, use_insole_drift=use_drift, dropout=dropout,
            pose_layers=pose_layers,
            tactile_input=str(saved_config.get("tactile_input", "raw108")),
            tactile_direct=bool(saved_config.get("tactile_direct", False)),
            no_imu=bool(saved_config.get("no_imu", False)),
            v_input=str(saved_config.get("v_input", "hrnet")),
            t_encoder=str(saved_config.get("t_encoder", "linear")),
            f2_repr=bool(saved_config.get("f2_repr", False)),
            **model_kw,
        ).to(device)
    # Quasi-strict load: the only tolerated missing keys are the hygiene-only
    # additions (foot_encoder stream_norm, fix_plan_v3 §3.1) that pre-date the
    # checkpoint.  Anything else — missing OR unexpected — is a real mismatch.
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if unexpected or not set(missing) <= _MISSING_OK:
        raise RuntimeError(
            "state_dict mismatch: missing=%s unexpected=%s" % (missing, unexpected)
        )
    # E4: sampling init scale must match the checkpoint's training noise scale
    # (train.py saves noise_scaled=True when it scales q_sample noise by
    # pose_std; older checkpoints lack the flag and sample unscaled).
    model.noise_scaled = bool(saved_config.get("noise_scaled", False))
    # V4A: the assignment softmax temperature must match the checkpoint's
    # final annealed value (default 1.0 = the plain V3-3 softmax).
    if getattr(getattr(model, "pose_head", None), "soft_parts", False) and bool(saved_config.get("assign_cluster", False)):
        model.pose_head.set_assign_temp(float(saved_config.get("assign_temp_final", 1.0)))
    model.eval()
    return model


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config_values = _parse_config_values(args.config_id)
    config = load_config(args.config)
    if args.modal is not None:
        config["modal"] = args.modal
    if args.ckpt is not None:
        targets: List[tuple[Optional[str], Optional[str], Path]] = [
            (args.modal, args.contact_method, args.ckpt)
        ]
    elif args.model_name is not None and args.sweep:
        if args.split != "test":
            print("--sweep ignores --split; using test")
            args.split = "test"
        targets = _sweep_targets(args.model_name, which=args.which)
    elif args.model_name is not None and args.contact_method is not None:
        if args.variant and args.model_name is None:
            raise ValueError("--variant requires --model-name (the dir identifier, e.g. V3_4a)")
        dir_name = args.model_name
        targets = [
            (
                None,
                args.contact_method,
                infer_anysole_paths(dir_name, variant=args.variant,
                                    contact=args.contact_method, which=args.which)["ckpt"],
            )
        ]
    else:
        # Bare `python -m anysole.eval`: auto-scan results/AnySole and
        # evaluate every model whose dir is named <modal>_<contact_method>.
        targets = _discover_model_targets(args.modal, args.contact_method)
        if not targets:
            raise ValueError(
                "--ckpt is required unless results/AnySole contains model dirs named "
                "<modal>_<contact_method> (e.g. anysolev1_tactile_abs) with "
                "checkpoints/ckpt_last.pt"
            )
    multi = len(targets) > 1
    if multi and args.metrics_out is not None:
        raise ValueError(
            "--metrics-out is per-model; omit it when evaluating several models "
            "(each model's metrics go to its own metrics/test_brief.json)"
        )
    if multi and args.protocol_out is not None:
        raise ValueError(
            "--protocol-out is per-model; omit it when evaluating several models"
        )
    device = resolve_device(args.device)
    for modal, contact_method, ckpt in targets:
        _evaluate_one(args, config, config_values, modal, contact_method, ckpt, device, multi=multi)
    return 0


def _positions_to_6d_batch(pred_pos: torch.Tensor, batch: dict, device: torch.device) -> torch.Tensor:
    """E6.1 export conversion: root-local positions -> WORLD 6D, per sample.

    positions_to_6d_np recovers rest-frame rotations; pre-multiplying the root
    by the session frame-0 rotation (dataset root_rot_init) yields world
    rotations for FK metrics and standard SMPL motion export.
    """
    out = []
    for i in range(pred_pos.shape[0]):
        sixd = positions_to_6d_np(
            pred_pos[i].cpu().numpy(),
            batch["offsets"][i].cpu().numpy(),
            batch["parents"][i].cpu().numpy(),
        )
        root_init = batch["root_rot_init"][i].cpu().numpy()  # (3,3)
        root_local = rot6d_to_rotmat_np(sixd.reshape(-1, N_JOINTS, 6)[:, 0:1, :])[:, 0]
        world_root = np.einsum("ij,tjk->tik", root_init, root_local)
        sixd[:, 0:6] = rotmat_to_6d_np(world_root)
        out.append(torch.from_numpy(sixd))
    return torch.stack(out, dim=0).to(device)


def _session_window_groups(dataset: AnySoleDataset) -> dict:
    """{session_id: [dataset indices]} in dataset order (consecutive windows)."""
    groups: dict = {}
    for i in range(len(dataset)):
        groups.setdefault(dataset[i]["session_id"], []).append(i)
    return groups


def _sample_x_t_init(model, bsz, tw, device, prior=None):
    """Sampling init: pure noise, or E6.5 marginal start with a prior."""
    x_T = torch.randn(bsz, tw, model.pose_head.pose_dim, device=device)
    if getattr(model, "noise_scaled", False):
        x_T = x_T * model.pose_head.pose_std.to(device).view(1, 1, -1)
    if prior is not None:
        sqrt_abar = float(np.sqrt(model.diffusion_abar_top))
        x_T = sqrt_abar * prior.to(device) + float(np.sqrt(1.0 - model.diffusion_abar_top)) * x_T
    return x_T


def _evaluate_one(
    args: argparse.Namespace,
    config: dict,
    config_values: List[int],
    modal: Optional[str],
    contact_method: Optional[str],
    ckpt: Path,
    device: torch.device,
    multi: bool = False,
) -> None:
    """Evaluate one checkpoint under each requested conditioning mode."""
    ckpt_parent = ckpt.parent
    model_root = ckpt_parent.parent if ckpt_parent.name == "checkpoints" else ckpt_parent
    print("== evaluating %s (ckpt %s)" % (model_root.name, ckpt))
    motion_out = None
    if not args.no_write_motion:
        motion_out = args.write_motion or model_root / "predictions" / "eval_motion"
        if multi and args.write_motion is not None:
            # Scan mode with an explicit export dir: nest per model so
            # different checkpoints don't clobber each other's exports.
            motion_out = args.write_motion / model_root.name
    checkpoint = torch.load(ckpt, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Checkpoint must contain a 'model' state dict: %s" % ckpt)
    checkpoint_modal = str(checkpoint.get("config", {}).get("modal", MODEL_ANYSOLEV1))
    if modal is not None and modal != checkpoint_modal:
        raise ValueError("legacy architecture override %s does not match checkpoint architecture %s" % (modal, checkpoint_modal))
    ckpt_contact = str(checkpoint.get("config", {}).get("contact_method") or "")
    contact_method = contact_method or ckpt_contact or str(config["contact_method"])
    if not contact_method:
        raise ValueError("contact_method must be explicit in --contact-method, checkpoint, or config")
    from anysole.data.contact_package import VERSIONED_CONTACT_METHODS, corrected_contact_provenance
    if ckpt_contact in VERSIONED_CONTACT_METHODS or contact_method in VERSIONED_CONTACT_METHODS:
        if contact_method != ckpt_contact:
            raise ValueError("Corrected contact evaluation requires a checkpoint trained with the same label version")
        provenance = corrected_contact_provenance(contact_method)
        if checkpoint["config"].get("contact_label_package_sha256") != provenance["contact_label_package_sha256"]:
            raise ValueError("Checkpoint corrected contact package hash missing or mismatched")
    model = _load_model(checkpoint, config, device)
    # --no-imu checkpoints: the eval dataset must build the 38-dim T_s2m
    # (IMU channels deleted) to match the checkpoint's encoder input width.
    no_imu = bool(checkpoint.get("config", {}).get("no_imu", False))
    tw = int(checkpoint.get("config", {}).get("tw", config["tw"]))
    diffusion = GaussianDiffusion(n_train_steps=int(checkpoint.get("config", {}).get("diffusion_train_steps", config["diffusion_train_steps"])))
    sample_steps = int(args.sample_steps or config["diffusion_sample_steps"])
    model.diffusion_abar_top = float(diffusion.alphas_cumprod[diffusion.n_train_steps - 1])
    pos_mode = checkpoint_modal == MODEL_ANYSOLEV1_POS
    regress_mode = checkpoint_modal == MODEL_ANYSOLEV2 \
        and not bool(checkpoint.get("config", {}).get("v3f", False))
    v3e_mode = bool(checkpoint.get("config", {}).get("v3e", False))
    v3g_mode = bool(checkpoint.get("config", {}).get("v3g", False))
    v3h_mode = bool(checkpoint.get("config", {}).get("v3h", False))
    v5_1_mode = bool(checkpoint.get("config", {}).get("v5_1", False))
    v4_1_mode = bool(checkpoint.get("config", {}).get("v4_1", False))
    # V5_Av1 判据 ③ T 参与干预（--z-src）：VT 行的 z_PoE 换成指定来源
    # （V = 禁 T 专家；T = 反向对照）。与常规 VT 行逐会话配对比较 =
    # 干预实验读数。非 V5_Av1 模型忽略（打印提示）。
    z_src_override = getattr(args, "z_src", None)
    if z_src_override and not v5_1_mode:
        print("NOTE: --z-src %s ignored for non-V5_Av1 checkpoint" % z_src_override)
        z_src_override = None
    # V4_Av1 判据 ③ null-T 注入（--null-t）：同上口径，T 输入整窗置零。
    # 修改 03（2026-11-03）：解除 v4_1 限定，V7 家族（标准/ar/ar2/ctx，
    # model_v7 的 v7=True 属性）同样生效——T2M + null-T = 全输入为零 =
    # 无条件先验读数。不传开关时行为逐位不变（默认 False）。
    null_t = bool(getattr(args, "null_t", False))
    v7_family = bool(getattr(model, "v7", False))
    if null_t and not (v4_1_mode or v7_family):
        print("NOTE: --null-t ignored for non-V4_Av1/V7 checkpoint")
        null_t = False
    v3f_mode = bool(checkpoint.get("config", {}).get("v3f", False))
    # F2a: heading/tilt representation — world pose/trans recovered via
    # geometry.f2_to_world before FK metrics and SMPL export.
    f2_repr = bool(checkpoint.get("config", {}).get("f2_repr", False))
    # F1: hmr_gvhmr checkpoints consume the GVHMR channel (V_hmr).
    v_hmr_mode = str(checkpoint.get("config", {}).get("v_input", "hrnet")) == "hmr_gvhmr"
    continuation = bool(config.get("continuation", False))
    warm_start = bool(config.get("warm_start", False))
    # 续写协议 §4（V7_续写协议方案）：ckpt 带 v7_prev_ctx → 评估用
    # stride=tw//2 重叠窗 + 逐窗链式 ctx（导出/协议两路）。
    # 续写结构标志一律读模型属性（2026-11-01：Ctx4/Ctx5 entry STRUCTURE
    # 曾漏 v7_ctx3——checkpoint config 缺失导致导出走 v1 平铺 ctx 分支
    # IndexError + V/VT 行误用重叠数据集；模型属性才是真值）。
    v7_ctx = bool(getattr(model, "prev_ctx", False))
    v7_ctx_vcond = v7_ctx and bool(getattr(model, "ctx_vcond", False))
    v7_ctx3 = bool(getattr(model, "ctx3", False))
    v7_ctx4 = bool(getattr(model, "ctx4", False))
    v7_ctx4b = bool(getattr(model, "ctx4b", False))
    v7_ctx5 = bool(getattr(model, "ctx5", False))
    v7_ar = bool(getattr(model, "ar", False))
    v7_ar2 = bool(getattr(model, "ar2", False))

    # 修改 02（2026-11-03）：声明口径的重叠窗评估（--overlap-stride）。
    # N = tw 为等价性对照（走同一新代码路径，须与现协议逐位一致）。
    overlap_assemble = args.overlap_stride is not None
    if overlap_assemble:
        if not 1 <= args.overlap_stride <= tw:
            raise ValueError("--overlap-stride must be in [1, tw] (tw=%d)" % tw)
        if v7_ctx or v7_ar or pos_mode or f2_repr:
            raise NotImplementedError(
                "--overlap-stride 目前支持标准路径与 AR2 模型（不支持续写"
                "链式/帧级 AR/位置模态/f2 表示）")

    # --split selects the evaluated session set even without --limit-sessions.
    # (AnySoleDataset mode="eval" otherwise defaults to the val column, which
    # silently made every non-val metrics file report val numbers.)
    session_ids = load_split_ids(Path(config["split_csv"]), args.split)
    if args.limit_sessions is not None:
        if args.limit_sessions <= 0:
            raise ValueError("--limit-sessions must be positive")
        session_ids = session_ids[: args.limit_sessions]
    dataset = AnySoleDataset(
        mode="eval",
        seq_root=Path(config["seq_root"]),
        split_csv=Path(config["split_csv"]),
        cache_root=Path(config["cache_root"]),
        window_length=tw,
        session_ids=session_ids,
        contact_method=contact_method,
        v7_ar2=v7_ar2,
        obs_context=getattr(model, "obs_context", tw),
        smpl_support_policy=(args.smpl_support_policy or checkpoint.get("config", {}).get("smpl_support_policy", "legacy_clip")),
        tactile_input=str(checkpoint.get("config", {}).get("tactile_input", "raw108")),
        no_imu=no_imu,
        v_input=str(checkpoint.get("config", {}).get("v_input", "hrnet")),
        f2_repr=bool(checkpoint.get("config", {}).get("f2_repr", False)),
        smpl_roots=checkpoint.get("config", {}).get("smpl_roots", config.get("smpl_roots")),
        cam_id=int(config.get("cam_id", 3)),
        # E6.4: continuation eval uses overlapping windows (stride = tw/2);
        # without it the stride stays non-overlapping (E3 behavior).
        # 修改 02：--overlap-stride 声明的重叠窗（协议/导出同口径）。
        stride=(args.overlap_stride if overlap_assemble
                else ((tw // 2) if ((pos_mode and continuation) or v7_ctx)
                      else None)),
                        t_phys_v2=bool(checkpoint.get("config", {}).get("t_phys_v2", False)),
                        d8_heads=bool(checkpoint.get("config", {}).get("d8_heads", False)),
                        tactile_weight_norm=bool(checkpoint.get("config", {}).get("tactile_weight_norm", False)),
    )
    dataset_alt = None
    if v7_ctx3 or v7_ctx_vcond or v7_ctx5:
        # ctx3：V/VT 行回非重叠协议（回归 Cv5 口径），T 行走重叠链式——
        # 双数据集按条件选用（run_protocol/导出循环同口径）。
        dataset_alt = AnySoleDataset(
            mode="eval",
            seq_root=Path(config["seq_root"]),
            split_csv=Path(config["split_csv"]),
            cache_root=Path(config["cache_root"]),
            window_length=tw,
            session_ids=session_ids,
            contact_method=contact_method,
            smpl_support_policy=(args.smpl_support_policy or checkpoint.get("config", {}).get("smpl_support_policy", "legacy_clip")),
            tactile_input=str(checkpoint.get("config", {}).get("tactile_input", "raw108")),
            no_imu=no_imu,
            v_input=str(checkpoint.get("config", {}).get("v_input", "hrnet")),
            f2_repr=bool(checkpoint.get("config", {}).get("f2_repr", False)),
            smpl_roots=checkpoint.get("config", {}).get("smpl_roots", config.get("smpl_roots")),
            cam_id=int(config.get("cam_id", 3)),
            stride=None,
            t_phys_v2=bool(checkpoint.get("config", {}).get("t_phys_v2", False)),
            d8_heads=bool(checkpoint.get("config", {}).get("d8_heads", False)),
            tactile_weight_norm=bool(checkpoint.get("config", {}).get("tactile_weight_norm", False)),
        )
    if len(dataset) == 0:
        raise RuntimeError("Evaluation dataset contains no valid windows")
    if overlap_assemble and dataset.stride != tw:
        # GT 泄漏硬约束（修改 02 §B4）：锚点网格 = 现协议非重叠窗网格。
        # tile 首窗无效时该 tile 的平移重叠窗一并剔除——否则会引入现协议
        # 从未消费的 GT 锚（n_gt_anchors_used 不再逐位相同）并扩大评分
        # 帧集（2026-11-03 s5 扫描发现）。
        from anysole.utils.eval_protocol import prune_overlap_windows
        before = len(dataset.valid_windows)
        dataset.valid_windows = prune_overlap_windows(dataset.valid_windows, tw)
        print("overlap-stride %d: prune %d -> %d valid windows (tile grid)"
              % (dataset.stride, before, len(dataset.valid_windows)))
    window_session_ids = {
        dataset.sessions[file_index]["session_id"]
        for file_index, _left, _right in dataset.valid_windows
    }
    skipped_no_window = [sid for sid in session_ids if sid not in window_session_ids]
    if skipped_no_window:
        print(
            "skip sessions with no valid evaluation windows: "
            + ", ".join(skipped_no_window)
        )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size or config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        collate_fn=collate_windows,
        pin_memory=device.type == "cuda",
    )

    if overlap_assemble:
        # 导出侧 tile 锚覆盖（修改 02 §B4）需要会话级 trans_global 查询；
        # 打包段会再次构建同名查询（覆盖前值，无副作用）。
        session_by_id = {s["session_id"]: s for s in loader.dataset.sessions}
    for config_value in config_values:
        exports = {}
        with torch.inference_mode():
            if v7_ctx and motion_out is not None:
                # 续写协议 §4：逐会话逐窗链式生成（ctx = 上一窗预测在 half
                # 处的根状态），全窗写入 exports → _pack_motion_export
                # 同时间淡化组装（eval.py 顶部修复版）。val 不写 motion
                # （motion_out=None）时跳过（指标走 run_protocol）。
                half = tw // 2
                hard_prefix = bool(checkpoint.get("config", {}).get("v7_ctx_hardprefix", False))
                ds_for = dataset_alt if ((v7_ctx3 or v7_ctx_vcond or v7_ctx5) and config_value != CONFIG_T) else dataset
                for session_id, idxs in _session_window_groups(ds_for).items():
                    ctx = None
                    prev_start = None
                    prev_pose = None  # G3 硬前缀（与 run_protocol 同款拼接）
                    prev_v = None
                    R_prev = None    # v4-a：锚 R_a 链式携带
                    for i in idxs:
                        raw = ds_for[i]   # 2026-11-01 修复：曾恒取 dataset[i]
                        batch = move_batch(collate_windows([raw]), device)
                        batch_size = 1
                        config_id = torch.full((batch_size,), config_value,
                                               device=device, dtype=torch.long)
                        batch["config_id"] = config_id
                        assert_batch_shapes(batch, batch_size, tw=tw,
                            t_phys_v2=bool(checkpoint.get("config", {}).get("t_phys_v2", False)))
                        v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, config_id)
                        if null_t:
                            # 修改 03（2026-11-03）：null-T 对 V7 ctx 导出
                            # 同口径生效（与评分路径一致）。
                            t_raw = torch.zeros_like(t_raw)
                            t_phys = torch.zeros_like(t_phys)
                            t_s2m = torch.zeros_like(t_s2m)
                        v_kw = {"V_hmr": batch.get("V_hmr")} if v_hmr_mode else {}
                        start = int(raw["frame_start"])
                        if ctx is None or prev_start is None or start != prev_start + half:
                            ctx = (torch.zeros(1, half, 148, device=device) if v7_ctx5
                                   else torch.zeros(1, half, 10, device=device) if v7_ctx3
                                   else torch.zeros(1, 10, device=device))
                        out = model(v_feat, t_raw, t_phys, config_id, [session_id],
                                    T_s2m=t_s2m, **v_kw, prev_ctx=ctx)
                        anchor = batch["trans_anchor"][:, None, :]
                        reset = (prev_start is None or start != prev_start + half)
                        if v7_ctx4 or v7_ctx4b:
                            # v4-a：残差输出组合 R_a 回世界系（首窗/复位 I）。
                            # v4-b：T 行 = 链锚（同 v4-a）；V/VT 行 = 锚估计头。
                            if v7_ctx4b and config_value != CONFIG_T:
                                R_a = out["anchor_rot"][0]
                            else:
                                R_a = torch.eye(3, device=device) if reset else R_prev
                            pose_w = rot_root6d(R_a[None], out["x0_hat"][0])
                            v_w = torch.einsum("ij,tj->ti", R_a, out["v_hat"][0])
                            trans_w = torch.cumsum(v_w, dim=0) / float(FPS) + anchor[0]
                            if config_value == CONFIG_T:
                                R_prev = torch.matmul(R_a, rot6d_to_rotmat(out["x0_hat"][0, half, :6]))
                        elif hard_prefix and prev_pose is not None:
                            pose_w = out["x0_hat"][0].clone()
                            v_w = out["v_hat"][0].clone()
                            pose_w[:half] = prev_pose[half:]
                            v_w[:half] = prev_v[half:]
                            trans_w = torch.cumsum(v_w, dim=0) / float(FPS) + anchor[0]
                        else:
                            pose_w = out["x0_hat"][0]
                            v_w = out["v_hat"][0]
                            trans_w = (out["trans_hat"] + anchor)[0]
                        exports.setdefault(session_id, []).append(
                            (
                                start,
                                pose_w.cpu().numpy(),
                                trans_w.cpu().numpy(),
                                (batch["trans_gt"] + anchor)[0].cpu().numpy(),
                            )
                        )
                        if v7_ctx3:
                            if v7_ctx5:
                                block = torch.cat([pose_w[half:, :144], v_w[half:, :3],
                                                   torch.ones(half, 1, device=device)], dim=-1)
                            else:
                                block = torch.cat([pose_w[half:, :6], v_w[half:, :3],
                                                   torch.ones(half, 1, device=device)], dim=-1)
                            ctx = block.unsqueeze(0)
                        else:
                            root6 = pose_w[half, :6].detach()
                            vel3 = v_w[half, :3].detach()
                            ctx = torch.cat([root6, vel3, torch.ones(1, device=device)],
                                            dim=0).unsqueeze(0)
                        prev_start = start
                        prev_pose = pose_w
                        prev_v = v_w
            elif v7_ar2 and motion_out is not None:
                # AR2 自由生成（2026-11-01，V7_AR方案 §6）：与协议同源的
                # ar2_session_chunks 组装（同一序列口径，§6.5），世界姿态
                # + 自由积分轨迹写入 exports；无每窗 GT 锚。
                # ar2_regression（2026-11-02）：窗口并行前向（V7_Cv5AR2R）。
                from anysole.utils.eval_protocol import ar2_session_chunks
                ar2_reg = (bool(getattr(model, "ar2_regression", False))
                            or bool(checkpoint.get("config", {}).get(
                                "ar2_regression", False)))
                # 与评分同口径的增益下限/先验高度（run_protocol 同键同默认）。
                ar2_gain_floor = float(config.get("ar2_vis_gain_floor", 0.0))
                ar2_init_height = float(config.get("ar2_init_height", 0.0))
                for session_id, idxs in _session_window_groups(dataset).items():
                    v_kw = ({"V_hmr": dataset[idxs[0]].get("V_hmr")}
                            if v_hmr_mode else {})
                    chunks, _meta = ar2_session_chunks(
                        dataset, model, session_id, idxs, config_value,
                        device, tw, v_kw, regression=ar2_reg,
                        gain_floor=ar2_gain_floor, null_t=null_t,
                        dyaw_base=str(checkpoint.get("config", {}).get(
                            "ar2_dyaw_base", "zero")),
                        init_height=ar2_init_height)
                    for chunk in chunks:
                        start = chunk["start"]
                        exports.setdefault(session_id, []).append(
                            (start, chunk["pred_pose"].numpy(),
                             chunk["pred_trans"].numpy(),
                             chunk["gt_trans"].numpy()))
            elif v7_ar and motion_out is not None:
                # 帧级 AR（全条件，2026-11-01 用户裁定）：逐帧串行生成，
                # 整窗写入 exports（窗口间携带末帧状态）。
                for session_id, idxs in _session_window_groups(dataset).items():
                        prev = torch.zeros(147, device=device)
                        prev_start = None
                        for i in idxs:
                            raw = dataset[i]
                            batch = move_batch(collate_windows([raw]), device)
                            batch_size = 1
                            config_id = torch.full((batch_size,), config_value,
                                                   device=device, dtype=torch.long)
                            batch["config_id"] = config_id
                            assert_batch_shapes(batch, batch_size, tw=tw,
                                t_phys_v2=bool(checkpoint.get("config", {}).get("t_phys_v2", False)))
                            v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, config_id)
                            if null_t:
                                # 修改 03（2026-11-03）：null-T 对 V7 帧级
                                # AR 导出同口径生效（与评分路径一致）。
                                t_raw = torch.zeros_like(t_raw)
                                t_phys = torch.zeros_like(t_phys)
                                t_s2m = torch.zeros_like(t_s2m)
                            v_kw = {"V_hmr": batch.get("V_hmr")} if v_hmr_mode else {}
                            start = int(raw["frame_start"])
                            if prev_start is None or start != prev_start + tw:
                                prev = torch.zeros(147, device=device)
                            anchor = batch["trans_anchor"][:, None, :]
                            ps = torch.zeros(1, tw, 147, device=device)
                            ps[0, 0] = prev
                            poses, vels = [], []
                            for t in range(tw):
                                out = model(v_feat, t_raw, t_phys, config_id,
                                            [session_id], T_s2m=t_s2m, **v_kw,
                                            prev_state=ps, causal=True)
                                poses.append(out["x0_hat"][0, t].detach())
                                vels.append(out["v_hat"][0, t].detach())
                                if t + 1 < tw:
                                    ps[0, t + 1] = torch.cat([poses[-1], vels[-1]])
                            pred_pose = torch.stack(poses, dim=0)
                            v_hat = torch.stack(vels, dim=0)
                            pred_trans = torch.cumsum(v_hat, dim=0) / float(FPS) + anchor[0]
                            exports.setdefault(session_id, []).append(
                                (start, pred_pose.cpu().numpy(),
                                 pred_trans.cpu().numpy(),
                                 (batch["trans_gt"] + anchor)[0].cpu().numpy()))
                            prev = torch.cat([poses[-1], vels[-1]])
                            prev_start = start
            elif pos_mode:
                # E6.1 position mode: sample positions, convert to world 6D
                # per window (R_init composition), then reuse all 6D metrics.
                if continuation:
                    # E6.4: each session's windows share one DDIM chain.
                    half = tw // 2
                    for session_id, idxs in _session_window_groups(dataset).items():
                        raw = [dataset[i] for i in idxs]
                        batch = move_batch(collate_windows(raw), device)
                        bsz = batch["pose_gt"].shape[0]
                        config_id = torch.full((bsz,), config_value, device=device, dtype=torch.long)
                        batch["config_id"] = config_id
                        assert_batch_shapes(batch, bsz, tw=tw,
                                   t_phys_v2=bool(checkpoint.get("config", {}).get("t_phys_v2", False)))
                        v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, config_id)
                        v_kw = {"V_hmr": batch.get("V_hmr")} if v_hmr_mode else {}
                        cond = {"V_feat": v_feat, "T_raw": t_raw, "T_phys": t_phys,
                                "T_s2m": t_s2m, "V_hmr": batch.get("V_hmr") if v_hmr_mode else None,
                                "config_id": config_id,
                                "session_id": batch.get("session_id")}
                        prior = model.pose_head.pose_mean.view(1, 1, -1).expand(bsz, tw, -1) if warm_start else None
                        x_T = _sample_x_t_init(model, bsz, tw, device, prior=prior)
                        pred_pose = diffusion.ddim_sample_loop_continue(
                            model, x_T=x_T, tau_related_kwargs=cond,
                            steps=sample_steps, half=half)[0]
                        tau_zero = torch.zeros(bsz, device=device, dtype=torch.long)
                        out = model(v_feat, t_raw, t_phys, pred_pose, tau_zero, config_id,
                                    batch.get("session_id"), T_s2m=t_s2m, **v_kw)
                        anchor = batch["trans_anchor"][:, None, :]
                        pred_trans = out["trans_hat"] + anchor
                        gt_trans = batch["trans_gt"] + anchor
                        pred6d = _positions_to_6d_batch(pred_pose, batch, device)
                        for w in range(bsz):
                            sl = slice(0, tw) if w == 0 else slice(half, tw)
                            p6 = pred6d[w : w + 1, sl]
                            pt = pred_trans[w : w + 1, sl]
                            gtt = gt_trans[w : w + 1, sl]
                            kp = fk_pose6d(p6, pt, batch["offsets"][w : w + 1], batch["parents"][w : w + 1])
                            kp_gt = batch["kp_gt"][w : w + 1, sl]
                            if motion_out is not None:
                                exports.setdefault(session_id, []).append(
                                    (
                                        int(raw[w]["frame_start"]) + (0 if w == 0 else half),
                                        p6[0].cpu().numpy(),
                                        pt[0].cpu().numpy(),
                                        gtt[0].cpu().numpy(),
                                    )
                                )
                else:
                    for raw_batch in loader:
                        batch = move_batch(raw_batch, device)
                        batch_size = batch["pose_gt"].shape[0]
                        config_id = torch.full((batch_size,), config_value, device=device, dtype=torch.long)
                        batch["config_id"] = config_id
                        assert_batch_shapes(batch, batch_size, tw=tw,
                            t_phys_v2=bool(checkpoint.get("config", {}).get("t_phys_v2", False)))
                        v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, config_id)
                        v_kw = {"V_hmr": batch.get("V_hmr")} if v_hmr_mode else {}
                        cond = {"V_feat": v_feat, "T_raw": t_raw, "T_phys": t_phys,
                                "T_s2m": t_s2m, "V_hmr": batch.get("V_hmr") if v_hmr_mode else None,
                                "config_id": config_id,
                                "session_id": batch.get("session_id")}
                        prior = model.pose_head.pose_mean.view(1, 1, -1).expand(batch_size, tw, -1) if warm_start else None
                        x_T = _sample_x_t_init(model, batch_size, tw, device, prior=prior)
                        pred_pose = diffusion.ddim_sample_loop(
                            model, x_T=x_T, tau_related_kwargs=cond,
                            steps=sample_steps, eta=0.0, device=device)
                        tau_zero = torch.zeros(batch_size, device=device, dtype=torch.long)
                        out = model(v_feat, t_raw, t_phys, pred_pose, tau_zero, config_id,
                                    batch.get("session_id"), T_s2m=t_s2m, **v_kw)
                        anchor = batch["trans_anchor"][:, None, :]
                        pred_trans = out["trans_hat"] + anchor
                        gt_trans = batch["trans_gt"] + anchor
                        pred6d = _positions_to_6d_batch(pred_pose, batch, device)
                        pred_kp = fk_pose6d(pred6d, pred_trans, batch["offsets"], batch["parents"])
                        if motion_out is not None:
                            for idx, session_id in enumerate(raw_batch["session_id"]):
                                exports.setdefault(session_id, []).append(
                                    (
                                        int(raw_batch["frame_start"][idx]),
                                        pred6d[idx].cpu().numpy(),
                                        pred_trans[idx].cpu().numpy(),
                                        gt_trans[idx].cpu().numpy(),
                                    )
                                )
            else:
                for raw_batch in loader:
                    batch = move_batch(raw_batch, device)
                    batch_size = batch["pose_gt"].shape[0]
                    config_id = torch.full((batch_size,), config_value, device=device, dtype=torch.long)
                    batch["config_id"] = config_id
                    assert_batch_shapes(batch, batch_size, tw=tw,
                            t_phys_v2=bool(checkpoint.get("config", {}).get("t_phys_v2", False)))
                    v_feat, t_raw, t_phys, t_s2m = condition_inputs(batch, config_id)
                    if v3f_mode:
                        # V3_Fv1: T stays live under every condition (the
                        # contact head reads the real stream; conditions are
                        # sampler switches).  condition_inputs already zeroed
                        # V for T2M — exactly the unconditional pass.
                        t_raw, t_phys = batch["T_raw"], batch["T_phys"]
                    if v5_1_mode or v4_1_mode:
                        # V5_Av1/V4_Av1：T 流输入三条件恒定（g_T/接触头/XAttn
                        # 恒读真实压力，判据 ③ 干预实验要求同输入比
                        # 来源）；V 零化保留（T2M 行）。
                        t_raw, t_phys = batch["T_raw"], batch["T_phys"]
                    if null_t:
                        # V4_Av1 判据 ③ null-T 注入（--null-t）：T 输入整窗
                        # 置零（三条件），VT 行修正分支应输出 ≈0。
                        t_raw = torch.zeros_like(t_raw)
                        t_phys = torch.zeros_like(t_phys)
                        t_s2m = torch.zeros_like(t_s2m)
                    v_kw = {"V_hmr": batch.get("V_hmr")} if v_hmr_mode else {}
                    if v3f_mode:
                        out = sample_v3f(
                            model, v_feat, t_raw, t_phys,
                            anchor=batch["trans_anchor"][:, None, :],
                            floor_y=batch["floor_y"],
                            offsets=batch["offsets"], parents=batch["parents"],
                            use_t=config_value != CONFIG_V,
                            refine=config_value != CONFIG_V,
                            diffusion=diffusion, steps=sample_steps, eta=0.0,
                            device=device)
                        pred_pose = out["x0_hat"]
                    elif regress_mode:
                        # F0b: single regression forward — no DDIM chain.
                        # E6.4 continuation is diffusion-era and never applies
                        # here (F3 replaces it with Hann blending); the FADE=4
                        # crossfade below stays.
                        z_kw = {"z_src": v5_1_z_src(config_value, z_src_override)} if v5_1_mode else {}
                        out = model(v_feat, t_raw, t_phys, config_id,
                                    batch.get("session_id"), T_s2m=t_s2m, **v_kw, **z_kw)
                        pred_pose = out["x0_hat"]
                    else:
                        cond = {
                            "V_feat": v_feat,
                            "T_raw": t_raw,
                            "T_phys": t_phys,
                            "T_s2m": t_s2m,
                            "V_hmr": batch.get("V_hmr") if v_hmr_mode else None,
                            "config_id": config_id,
                            "session_id": batch.get("session_id"),
                        }
                        prior = model.pose_head.pose_mean.view(1, 1, -1).expand(batch_size, tw, -1) if warm_start else None
                        pred_pose = diffusion.ddim_sample_loop(
                            model,
                            tau_related_kwargs=cond,
                            shape=(batch_size, tw, POSE_DIM),
                            steps=sample_steps,
                            eta=0.0,
                            device=device,
                            prior=prior,
                        )
                        tau_zero = torch.zeros(batch_size, device=device, dtype=torch.long)
                        out = model(v_feat, t_raw, t_phys, pred_pose, tau_zero, config_id,
                                    batch.get("session_id"), T_s2m=t_s2m, **v_kw)
                    pred_trans = out["trans_hat"]
                    anchor = batch["trans_anchor"][:, None, :]
                    if overlap_assemble:
                        # GT 泄漏硬约束（修改 02 §B4）：导出与协议同口径——
                        # 锚点网格保持原非重叠网格，重叠窗继承所属 tile 的
                        # 锚（stride=tw 时与现导出逐位相同）。
                        tile_anchors = []
                        for j in range(batch_size):
                            s = session_by_id[batch["session_id"][j]]
                            left = int(batch["frame_start"][j])
                            tile_anchors.append(np.asarray(s["trans_global"])[
                                max((left // tw) * tw - 1, 0)])
                        anchor = torch.from_numpy(
                            np.stack(tile_anchors)).float().to(device)[:, None, :]
                    if f2_repr:
                        # F2a: world pose/trans from the tilt pose + heading
                        # trajectory; the GT world pose is recovered the same
                        # way for the rotation metric.
                        pred_pose_w, pred_trans_world = f2_to_world(
                            pred_pose, out["v_hat"], batch["psi_anchor"], batch["trans_anchor"]
                        )
                        gt_pose_w, _ = f2_to_world(
                            batch["pose_gt"], batch["traj_gt_f2"], batch["psi_anchor"],
                            batch["trans_anchor"],
                        )
                    else:
                        if v3e_mode:
                            # V3_Ev1: VT exports the constraint-processed
                            # prediction; V/T export the two raw streams.
                            if config_value == CONFIG_VT:
                                pred_pose_w, pred_trans_world = apply_v3e_constraints(
                                    out["x0_hat"], pred_trans + anchor,
                                    batch["floor_y"],
                                    batch["offsets"], batch["parents"])
                            elif config_value == CONFIG_V:
                                pred_pose_w = pred_pose
                                pred_trans_world = pred_trans + anchor
                            else:
                                pred_pose_w = out["x0_prior"]
                                pred_trans_world = out["trans_prior"] + anchor
                        elif v3g_mode:
                            # V3_Gv1: VT = 分部观测层 on the V stream; V = raw
                            # carrier (= E); T = T prior + 观测层.  floor_src
                            # stays the registered default "gt" (E v2.1 口径).
                            if config_value == CONFIG_VT:
                                pred_pose_w, pred_trans_world = apply_v3g_constraints(
                                    out["x0_hat"], pred_trans + anchor,
                                    out["contact_logits"],
                                    batch["offsets"], batch["parents"],
                                    floor_y=batch["floor_y"])
                            elif config_value == CONFIG_V:
                                pred_pose_w = pred_pose
                                pred_trans_world = pred_trans + anchor
                            else:
                                pred_pose_w, pred_trans_world = apply_v3g_constraints(
                                    out["x0_prior"], out["trans_prior"] + anchor,
                                    out["contact_logits"],
                                    batch["offsets"], batch["parents"],
                                    floor_y=batch["floor_y"])
                        elif v3h_mode:
                            # V3_Hv1: VT = 残差通道 on the V stream; V = raw
                            # V tower; T = null-V 先验 + 残差通道 (H 无 T 塔
                            # ——同一 V 塔、V 已在上游零化).  floor_src stays
                            # the registered default "gt" (E v2.1 口径).
                            # V3_Hv3：VT/T 行先应用一阶足部残差。
                            pose_vt = apply_foot_residual(
                                out["x0_hat"], out["foot_res"],
                                landing=str(getattr(model, "res_landing", "foot"))) if out.get("foot_res") is not None else out["x0_hat"]
                            if config_value == CONFIG_VT:
                                pred_pose_w, pred_trans_world = apply_v3h(
                                    bool(getattr(model, "no_anchor", False)),
                                    pose_vt, pred_trans + anchor,
                                    out["contact_logits"],
                                    batch["offsets"], batch["parents"],
                                    floor_y=batch["floor_y"])
                            elif config_value == CONFIG_V:
                                pred_pose_w = pred_pose
                                pred_trans_world = pred_trans + anchor
                            else:
                                pred_pose_w, pred_trans_world = apply_v3h(
                                    bool(getattr(model, "no_anchor", False)),
                                    pose_vt, pred_trans + anchor,
                                    out["contact_logits"],
                                    batch["offsets"], batch["parents"],
                                    floor_y=batch["floor_y"])
                        elif v4_1_mode:
                            # V4_Av1: 姿态输出/渲染用 pose_v（修正只进评估
                            # 软骨架）；T2M 行导出 T 先验流。
                            if config_value == CONFIG_T:
                                pred_pose_w = out["x0_prior"]
                                pred_trans_world = out["trans_prior"] + anchor
                            else:
                                pred_pose_w = pred_pose
                                pred_trans_world = pred_trans + anchor
                        elif getattr(model, "ctx4b", False):
                            # v4-b：V/VT 行出口组合（锚估计头）；T 行走链式
                            # 导出路径（本分支防御：GT 锚 teacher-forced）。
                            if config_value == CONFIG_T:
                                R_a = rot6d_to_rotmat(batch["pose_gt"][:, 0, :6])
                            else:
                                R_a = out["anchor_rot"]
                            pred_pose_w = rot_root6d(R_a[:, None, :, :], pred_pose)
                            v_w = torch.einsum("bij,btj->bti", R_a, out["v_hat"])
                            pred_trans_world = torch.cumsum(v_w, dim=1) / float(FPS) + anchor
                        else:
                            pred_trans_world = pred_trans + anchor
                            pred_pose_w = pred_pose
                        gt_pose_w = batch["pose_gt"]
                    gt_trans_world = batch["trans_gt"] + anchor
                    pred_kp = fk_pose6d(pred_pose_w, pred_trans_world, batch["offsets"], batch["parents"])


                    if motion_out is not None:
                        for idx, session_id in enumerate(raw_batch["session_id"]):
                            exports.setdefault(session_id, []).append(
                                (
                                    int(raw_batch["frame_start"][idx]),
                                    pred_pose_w[idx].cpu().numpy(),
                                    pred_trans_world[idx].cpu().numpy(),
                                    gt_trans_world[idx].cpu().numpy(),
                                )
                            )

        if motion_out is not None:
            beta_by_session = {
                s["session_id"]: np.asarray(s.get("betas", np.zeros(10)), dtype=np.float32)
                for s in loader.dataset.sessions
            }
            session_by_id = {
                s["session_id"]: s for s in loader.dataset.sessions
            }
            for session_id, windows in exports.items():
                session = session_by_id[session_id]
                packed = _pack_motion_export(session, windows, pos_mode,
                                             overlap_blend=overlap_assemble)
                pose = packed["pose"]
                trans = packed["trans"]
                gt_trans = packed["gt_trans"]
                # Re-orthonormalize blended 6D vectors before writing the
                # Standard SMPL motion archive.
                R = rot6d_to_rotmat(torch.from_numpy(pose.reshape(-1, N_JOINTS, 6)).float())
                pose = rotmat_to_6d(R).reshape(-1, POSE_DIM).numpy()
                joints_native = fk_pose6d(
                    torch.from_numpy(pose).float().unsqueeze(0),
                    torch.from_numpy(trans).float().unsqueeze(0),
                    torch.from_numpy(np.asarray(session["offsets"])).float(),
                    torch.from_numpy(np.asarray(session["parents"])).long(),
                )[0].numpy()
                packed["joint_xyz_world"] = _smpl_yup_to_display(joints_native)
                output_path = motion_out / ("%s_%s.npz" % (session_id, CONFIG_MODE_NAMES[config_value]))
                motion_out.mkdir(parents=True, exist_ok=True)
                smpl_poses = smpl24_pose6d_to_poses(pose)
                np.savez_compressed(
                    output_path,
                    poses=smpl_poses,
                    trans=pelvis_to_smpl_trans(pose, trans, beta_by_session.get(session_id)),
                    # Standard SMPL ``trans`` above is model-origin
                    # translation.  Keep the exact pelvis-space arrays used
                    # by traj_ATE separately so visualization never compares
                    # model origin against pelvis GT.
                    pred_pelvis_trans=trans.astype(np.float32),
                    gt_pelvis_trans=gt_trans.astype(np.float32),
                    gt_trans=gt_trans.astype(np.float32),  # legacy alias
                    betas=beta_by_session.get(session_id, np.zeros((10,), dtype=np.float32)),
                    betas_source=np.asarray("ground_truth_session"),
                    joint_xyz_world=packed["joint_xyz_world"].astype(np.float32),
                    joint_names=np.asarray(JOINT_NAMES),
                    valid_mask=packed["valid"].astype(bool),
                    frame_indices=np.arange(len(pose), dtype=np.int64),
                    joint_coordinate_system=np.asarray("world_z_up"),
                    session_id=np.asarray(session_id),
                    root_orient=smpl_poses[:, :3],
                    pose_body=smpl_poses[:, 3:],
                    mocap_frame_rate=np.asarray(FPS, dtype=np.float32),
                    source_frame_times_s=np.arange(len(pose), dtype=np.float32) / float(FPS),
                    **smpl_archive_metadata(),
                )
                print("wrote %s" % output_path)
    metrics_out = args.metrics_out
    if metrics_out is None:
        metrics_out = model_root / "metrics" / ("%s_brief.json" % args.split)
    from anysole.utils.eval_protocol import run_protocol
    protocol_out = args.protocol_out
    if protocol_out is None:
        protocol_out = model_root / "metrics" / ("%s.json" % args.split)
    final_metrics = run_protocol(
        checkpoint=checkpoint,
        config=config,
        model=model,
        dataset=dataset,
        device=device,
        checkpoint_path=str(ckpt),
        split=args.split,
        config_values=config_values,
        regress_mode=regress_mode,
        diffusion=diffusion,
        sample_steps=sample_steps,
        warm_start=warm_start,
        seed=args.protocol_seed,
        robustness=not args.no_robustness,
        contact_method=contact_method,
        tw=tw,
        out_path=protocol_out,
        brief_out=metrics_out,
        v2t_out_dir=motion_out,
        dump_detail=args.dump_detail,
        z_src_override=z_src_override,
        null_t=null_t,
        dataset_alt=dataset_alt,
        overlap_assemble=overlap_assemble,
    )
    # Bind exported scores to the actual weights and contact version.
    from anysole.data.contact_package import sha256 as _artifact_sha256
    provenance_fields = {"checkpoint_sha256": _artifact_sha256(ckpt),
                         "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
                         "contact_label_package_sha256": checkpoint.get("config", {}).get("contact_label_package_sha256"),
                         "contact_label_definition": checkpoint.get("config", {}).get("contact_label_definition")}
    for exported_path in (protocol_out, metrics_out):
        if exported_path is not None and Path(exported_path).is_file():
            payload = json.loads(Path(exported_path).read_text())
            payload.update(provenance_fields)
            Path(exported_path).write_text(json.dumps(payload, indent=2) + "\n")
    if not final_metrics:
        raise RuntimeError(
            "canonical session evaluation requires non-overlapping windows; "
            "this checkpoint/config uses an unsupported continuation stride"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
