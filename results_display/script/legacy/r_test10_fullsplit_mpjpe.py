"""Full-split MPJPE with eval.py's exact path (DDIM -> tau0 polish -> pred_trans FK).

Answers: why probe batch 0 of eval split gives 98.4mm but test.json reports
177mm on the test split.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(REPO))

from anysole.data.dataset import AnySoleDataset, collate_windows, load_split_ids
from anysole.utils.diffusion import GaussianDiffusion
from anysole.utils.geometry import fk_pose6d
from anysole.models import AnySoleModel, MODEL_ANYSOLEV1
from anysole.train import condition_inputs
from anysole.types import POSE_DIM

CKPT = REPO / "results/AnySole/anysolev1_joint_and/checkpoints/ckpt_last.pt"
device = torch.device("cuda")
torch.manual_seed(0)

ckpt = torch.load(CKPT, map_location="cpu")
cfg = ckpt["config"]
model = AnySoleModel(
    d=int(cfg["d_model"]), tw=int(cfg["tw"]), modal=MODEL_ANYSOLEV1,
    dropout=float(cfg["dropout"]), pose_layers=int(cfg.get("pose_layers", 6)),
).to(device)
model.load_state_dict(ckpt["model"], strict=True)
model.eval()
diffusion = GaussianDiffusion(n_train_steps=int(cfg["diffusion_train_steps"]))

for split in ("val", "test"):
    session_ids = load_split_ids(Path(cfg["split_csv"]), split)
    ds = AnySoleDataset(
        mode="eval", seq_root=Path(cfg["seq_root"]), split_csv=Path(cfg["split_csv"]),
        cache_root=Path(cfg["cache_root"]), window_length=int(cfg["tw"]),
        session_ids=session_ids, contact_method=str(cfg.get("contact_method", "joint_and")),
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_windows)
    sums = torch.zeros(()).to(device)
    per_session = {}
    with torch.inference_mode():
        for raw_batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in raw_batch.items()}
            B = batch["pose_gt"].shape[0]
            config_id = torch.zeros(B, dtype=torch.long, device=device)  # VT2M
            v_feat, t_raw, t_phys = condition_inputs(batch, config_id)
            pred_pose = diffusion.ddim_sample_loop(
                model,
                tau_related_kwargs={"V_feat": v_feat, "T_raw": t_raw, "T_phys": t_phys,
                                    "config_id": config_id, "session_id": batch.get("session_id")},
                shape=(B, int(cfg["tw"]), POSE_DIM), steps=int(cfg["diffusion_sample_steps"]),
                eta=0.0, device=device,
            )
            tau_zero = torch.zeros(B, device=device, dtype=torch.long)
            out = model(v_feat, t_raw, t_phys, pred_pose, tau_zero, config_id, batch.get("session_id"))
            pred_trans_world = out["trans_hat"] + batch["trans_anchor"][:, None, :]
            pred_kp = fk_pose6d(pred_pose, pred_trans_world, batch["offsets"], batch["parents"])
            err = torch.linalg.vector_norm(pred_kp - batch["kp_gt"], dim=-1) * 1000.0  # (B,20)
            sums += err.sum()
            for i, sid in enumerate(raw_batch["session_id"]):
                per_session.setdefault(sid, []).append(float(err[i].mean().item()))
    n_windows = len(ds)
    mean = float(sums.item() / (n_windows * int(cfg["tw"])))
    sess_means = sorted((sum(v) / len(v), k) for k, v in per_session.items())
    print("%s split: %d windows, MPJPE = %.1f mm | per-session: min %.1f med %.1f max %.1f"
          % (split, n_windows, mean, sess_means[0][0], sess_means[len(sess_means) // 2][0], sess_means[-1][0]))
    print("   worst sessions:", [(k, round(m)) for m, k in sess_means[-3:]])
