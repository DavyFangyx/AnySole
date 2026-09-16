"""Per-group pose variation: body / left leg / right leg.

GT vs head output (baseline ckpt): within-window (time) std and across-window
(batch) std, per group, in raw 6D units. Explains "legs frozen, arms flailing".
"""
import sys
from pathlib import Path

import torch

GAIT = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(GAIT))

from anysole.data.dataset import AnySoleDataset, collate_windows
from anysole.models import AnySoleModel
from anysole.train import condition_inputs, load_config, move_batch
from anysole.types import CONFIG_VT, BODY_SLICE, LEFT_LEG_SLICE, RIGHT_LEG_SLICE

DEVICE = torch.device("cuda:6")


def main():
    config = load_config(GAIT / "configs/v1.yaml")
    config["contact_method"] = "joint_and"
    ckpt = torch.load(GAIT / "results/AnySole/anysolev1_joint_and/checkpoints/ckpt_last.pt",
                      map_location="cpu")
    saved = ckpt["config"]
    model = AnySoleModel(
        d=int(saved["d_model"]), tw=int(saved["tw"]), modal=str(saved["modal"]),
        dropout=float(saved["dropout"]), pose_layers=int(saved["pose_layers"]),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    ds = AnySoleDataset(
        mode="eval", seq_root=Path(config["seq_root"]), split_csv=Path(config["split_csv"]),
        cache_root=Path(config["cache_root"]), window_length=int(config["tw"]),
        contact_method="joint_and",
    )
    # 64 windows for stable stats
    batch = move_batch(collate_windows([ds[i] for i in range(64)]), DEVICE)
    bsz = batch["pose_gt"].shape[0]
    t0 = torch.zeros(bsz, device=DEVICE, dtype=torch.long)
    c = torch.full((bsz,), CONFIG_VT, device=DEVICE, dtype=torch.long)
    v, tr, tp = condition_inputs(batch, c)
    with torch.inference_mode():
        out = model(v, tr, tp, batch["pose_gt"], t0, c, batch.get("session_id"))["x0_hat"]

    gt = batch["pose_gt"]  # (B,20,138)
    print("group            GT std_time  GT std_batch   out std_time  out std_batch")
    for name, sl in (("body", BODY_SLICE), ("left_leg", LEFT_LEG_SLICE), ("right_leg", RIGHT_LEG_SLICE)):
        g_gt, g_out = gt[..., sl], out[..., sl]
        print("%-12s   %.4f        %.4f         %.4f         %.4f"
              % (name,
                 g_gt.std(dim=1).mean().item(),     # per-window temporal std, mean over batch
                 g_gt.mean(dim=1).std(dim=0).mean().item(),  # across-window std of window means
                 g_out.std(dim=1).mean().item(),
                 g_out.mean(dim=1).std(dim=0).mean().item()))


if __name__ == "__main__":
    main()
