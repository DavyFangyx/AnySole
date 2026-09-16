"""Follow-up probe: full-val mean-pose baseline, timestep-emb stats, self-attn.

Questions:
  A. Is the full-val tau0 readout (~145mm from wandb) just the mean-pose error?
  B. Does the trained timestep embedding distinguish tau=0 from tau=999?
  C. Self-attention: do pose tokens of the same frame stick together / does the
     timestep token dominate?
  D. Magnitude of F tokens vs decoder residual stream.
"""
import sys
from pathlib import Path

import torch

GAIT = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(GAIT))

from anysole.data.dataset import AnySoleDataset, collate_windows
from anysole.diffusion import GaussianDiffusion
from anysole.geometry import fk_pose6d
from anysole.models import AnySoleModel
from anysole.train import condition_inputs, load_config, move_batch
from anysole.types import CONFIG_VT

CKPT = GAIT / "results/AnySole/anysolev1_joint_and/checkpoints/ckpt_last.pt"
DEVICE = torch.device("cuda:6")


def main():
    config = load_config(GAIT / "configs/v1.yaml")
    config["contact_method"] = "joint_and"
    ckpt = torch.load(CKPT, map_location="cpu")
    saved = ckpt["config"]
    model = AnySoleModel(
        d=int(saved["d_model"]), tw=int(saved["tw"]), modal=str(saved["modal"]),
        dropout=float(saved["dropout"]), pose_layers=int(saved["pose_layers"]),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # ---- B: timestep embedding at tau=0 vs 999 ----
    with torch.inference_mode():
        e0 = model.embeddings.timestep(torch.zeros(8, dtype=torch.long, device=DEVICE))
        e999 = model.embeddings.timestep(torch.full((8,), 999, dtype=torch.long, device=DEVICE))
    d = (e0 - e999).abs().mean().item()
    print("[timestep emb] |e(0)-e(999)| mean abs = %.4f ; |e0| = %.4f, |e999| = %.4f"
          % (d, e0.abs().mean().item(), e999.abs().mean().item()))

    # ---- A: full-val mean-pose baseline vs head tau0 readout ----
    ds = AnySoleDataset(
        mode="eval", seq_root=Path(config["seq_root"]), split_csv=Path(config["split_csv"]),
        cache_root=Path(config["cache_root"]), window_length=int(config["tw"]),
        contact_method="joint_and",
    )
    print("\n[full val set: %d windows]" % len(ds))
    pm = model.pose_head.pose_mean
    sums = {"meanpose": 0.0, "head": 0.0, "head_mean_in": 0.0, "n": 0}
    cid = None
    with torch.inference_mode():
        for start in range(0, len(ds), 64):
            raw = [ds[i] for i in range(start, min(start + 64, len(ds)))]
            batch = move_batch(collate_windows(raw), DEVICE)
            bsz = batch["pose_gt"].shape[0]
            pose_gt = batch["pose_gt"]
            anchor = batch["trans_anchor"][:, None, :]
            gt_trans = batch["trans_gt"] + anchor
            kp_gt = batch["kp_gt"]
            mean_pose = pm.view(1, 1, -1).expand(bsz, pose_gt.shape[1], -1)
            kp_mean = fk_pose6d(mean_pose, gt_trans, batch["offsets"], batch["parents"])
            t0 = torch.zeros(bsz, device=DEVICE, dtype=torch.long)
            c = torch.full((bsz,), CONFIG_VT, device=DEVICE, dtype=torch.long)
            v, tr, tp = condition_inputs(batch, c)
            out = model(v, tr, tp, pose_gt, t0, c, batch.get("session_id"))
            kp_head = fk_pose6d(out["x0_hat"], gt_trans, batch["offsets"], batch["parents"])
            out2 = model(v, tr, tp, mean_pose, t0, c, batch.get("session_id"))
            kp_head2 = fk_pose6d(out2["x0_hat"], gt_trans, batch["offsets"], batch["parents"])
            n = bsz
            sums["meanpose"] += float(torch.linalg.vector_norm(kp_mean - kp_gt, dim=-1).sum().item())
            sums["head"] += float(torch.linalg.vector_norm(kp_head - kp_gt, dim=-1).sum().item())
            sums["head_mean_in"] += float(torch.linalg.vector_norm(kp_head2 - kp_gt, dim=-1).sum().item())
            sums["n"] += n
    total = 23.0 * 20.0 * sums["n"]
    print("[full-val MPJPE] mean-pose baseline      = %.1f mm" % (sums["meanpose"] / total * 1000))
    print("[full-val MPJPE] head tau0 (GT input)    = %.1f mm" % (sums["head"] / total * 1000))
    print("[full-val MPJPE] head tau0 (meanPose in) = %.1f mm" % (sums["head_mean_in"] / total * 1000))

    # ---- C/D: self-attention + magnitudes, one batch ----
    import torch.nn as nn

    batch = move_batch(collate_windows([ds[i] for i in range(8)]), DEVICE)
    bsz = batch["pose_gt"].shape[0]
    t0 = torch.zeros(bsz, device=DEVICE, dtype=torch.long)
    c = torch.full((bsz,), CONFIG_VT, device=DEVICE, dtype=torch.long)
    v, tr, tp = condition_inputs(batch, c)

    class MHAWrap(nn.Module):
        def __init__(self, mha, idx, name):
            super().__init__()
            self.mha = mha
            self.idx = idx
            self.name = name
            for attr in ("batch_first", "embed_dim", "num_heads"):
                if hasattr(mha, attr):
                    setattr(self, attr, getattr(mha, attr))

        def forward(self, *args, **kwargs):
            kwargs["need_weights"] = True
            out = self.mha(*args, **kwargs)
            w = out[1]
            if kwargs.get("tgt_mask") is not None or args[1].shape[1] == 461:
                print("  %s L%d self: mean=%.4f max=%.4f | t-token->pose sum=%.3f | pose->t-token mean=%.4f"
                      % (self.name, self.idx, w.mean().item(), w.max().item(),
                         w[0, 0, 1:].sum().item(), w[0, 1:, 0].mean().item()))
            return out

    wraps = []
    for li, layer in enumerate(model.pose_head.decoder.layers):
        w = MHAWrap(layer.self_attn, li, "posehead")
        layer.self_attn = w
        wraps.append(w)
    with torch.inference_mode():
        model(v, tr, tp, batch["pose_gt"], t0, c, batch.get("session_id"))

    # magnitudes: F tokens vs x tokens vs timestep token
    with torch.inference_mode():
        v_tok, t_tok = model.encoders(v, tr, tp, c)
        fused = model.fusion(v_tok, t_tok)
        x_tau = batch["pose_gt"]
        xn = (x_tau - model.pose_head.pose_mean) / model.pose_head.pose_std
        h = model.pose_head._embed(xn)
        t_tok_emb = model.embeddings.timestep(t0).unsqueeze(1) + model.pose_head.timestep_token
    print("\n[magnitudes] |F| = %.3f |x_tokens| = %.3f |t_tok| = %.3f |t_tok+learned| = %.3f"
          % (fused.abs().mean().item(), h.abs().mean().item(),
             model.embeddings.timestep(t0).abs().mean().item(),
             t_tok_emb.abs().mean().item()))
    print("[F per-token std over batch] %.3f (is F informative across samples?)"
          % fused.std(dim=0).mean().item())


if __name__ == "__main__":
    main()
