"""Test: does the huge timestep token erase x_tau from the residual stream?

Checks on baseline ckpt, E1 ckpt, and a fresh (untrained) model:
  1. |timestep_emb(tau)| magnitude: untrained vs trained (did it grow unboundedly?)
  2. attention of pose tokens -> timestep token in layer-0 self-attention
  3. cosine similarity between pose-token residual stream after layer 0 and the
     embedded input tokens (is x_tau information destroyed at layer 0?)
"""
import sys
from pathlib import Path

import torch

GAIT = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(GAIT))

from anysole.models import AnySoleModel
from anysole.train import load_config
from anysole.data.dataset import AnySoleDataset, collate_windows
from anysole.train import condition_inputs, move_batch
from anysole.types import CONFIG_VT

DEVICE = torch.device("cuda:6")


def magnitude_and_erasure(ckpt_path, tag):
    print("\n==== %s (%s) ====" % (tag, ckpt_path))
    config = load_config(GAIT / "configs/v1.yaml")
    config["contact_method"] = "joint_and"
    if ckpt_path is None:
        model = AnySoleModel(d=256, tw=20, modal="anysolev1", dropout=0.1, pose_layers=6).to(DEVICE)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        saved = ckpt["config"]
        model = AnySoleModel(
            d=int(saved["d_model"]), tw=int(saved["tw"]), modal=str(saved["modal"]),
            dropout=float(saved["dropout"]), pose_layers=int(saved["pose_layers"]),
        ).to(DEVICE)
        model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    with torch.inference_mode():
        for t in (0, 100, 999):
            e = model.embeddings.timestep(torch.full((8,), t, dtype=torch.long, device=DEVICE))
            print("  |timestep_emb(tau=%d)| = %.2f" % (t, e.abs().mean().item()))
    if ckpt_path is None:
        return

    ds = AnySoleDataset(
        mode="eval", seq_root=Path(config["seq_root"]), split_csv=Path(config["split_csv"]),
        cache_root=Path(config["cache_root"]), window_length=int(config["tw"]),
        contact_method="joint_and",
    )
    batch = move_batch(collate_windows([ds[i] for i in range(8)]), DEVICE)
    bsz = batch["pose_gt"].shape[0]
    t0 = torch.zeros(bsz, device=DEVICE, dtype=torch.long)
    c = torch.full((bsz,), CONFIG_VT, device=DEVICE, dtype=torch.long)
    v, tr, tp = condition_inputs(batch, c)
    with torch.inference_mode():
        v_tok, t_tok = model.encoders(v, tr, tp, c)
        fused = model.fusion(v_tok, t_tok)
        x_tau = batch["pose_gt"]
        xn = (x_tau - model.pose_head.pose_mean) / model.pose_head.pose_std
        h_in = model.pose_head._embed(xn)  # embedded input tokens (B,460,256)
        t_tok_emb = model.embeddings.timestep(t0).unsqueeze(1) + model.pose_head.timestep_token
        print("  |t_tok(full)| = %.2f  |x_tokens| = %.3f  |F| = %.3f"
              % (t_tok_emb.abs().mean().item(), h_in.abs().mean().item(),
                 fused.abs().mean().item()))

    # step the decoder one layer at a time manually to inspect the residual stream
    import torch.nn as nn
    h = torch.cat([t_tok_emb, h_in], dim=1)
    layer0 = model.pose_head.decoder.layers[0]
    # need attention weights from layer 0 self-attn: wrap
    class W(nn.Module):
        def __init__(self, mha):
            super().__init__()
            self.mha = mha
            self.batch_first = mha.batch_first
            self.embed_dim = mha.embed_dim
            self.num_heads = mha.num_heads
            self.head_dim = mha.head_dim
            self.w = None
        def forward(self, *a, **kw):
            kw["need_weights"] = True
            out = self.mha(*a, **kw)
            self.w = out[1]
            return out
    wrap = W(layer0.self_attn)
    layer0.self_attn = wrap
    with torch.inference_mode():
        h_out = layer0(h, fused)
    attn = wrap.w  # (B, 461, 461)
    print("  L0 self-attn: pose->t-token mean=%.4f (max=%.4f), t-token->pose sum=%.3f"
          % (attn[0, 1:, 0].mean().item(), attn[0, 1:, 0].max().item(),
             attn[0, 0, 1:].sum().item()))
    # cosine similarity of pose tokens before/after layer 0
    before = torch.nn.functional.normalize(h_in.reshape(-1, 256), dim=-1)
    after = torch.nn.functional.normalize(h_out[:, 1:].reshape(-1, 256), dim=-1)
    sim = (before * after).sum(dim=-1).mean().item()
    print("  cos-sim(embedded x_tau, post-L0 tokens) = %.3f  (<0.1 => input erased)" % sim)


if __name__ == "__main__":
    magnitude_and_erasure(None, "fresh untrained")
    magnitude_and_erasure(GAIT / "results/AnySole/anysolev1_joint_and/checkpoints/ckpt_last.pt", "baseline 2000ep")
    magnitude_and_erasure(GAIT / "results/AnySole/E1_taumax100/checkpoints/ckpt_last.pt", "E1 taumax100 330ep")
