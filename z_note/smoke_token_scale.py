"""Token-magnitude smoke check (fix_plan_v3 §3.2, hygiene item).

Guards the F5 failure mode from recurring in any form: token streams that are
"shape-correct but scale-wrong" slip through shape asserts while poisoning
cross-attention (|t_tok| ~1400x |v_tok| -> one-hot softmax -> LN erasure).

Checks on the F4a checkpoint, VT conditioning, one eval-mode batch:
 1. |v_tok| / |t_tok| in [1, 100]  (forward() stream, the fusion input);
 2. |encode_stream| / |forward| output in [0.5, 50]  (the archived 3-token
    stream stays in the same magnitude class after the LayerNorm fix);
 3. per-token magnitude of encode_stream ~ sqrt(dim) (LayerNorm contract,
    |x|_2 ~ 16 for dim=256), NOT the ~22000 the bug produced.

Usage (touch_gait env):
  python z_note/smoke_token_scale.py --ckpt results/AnySole/F4a_footconv/checkpoints/ckpt_last.pt --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path("/data/fangyuxuan/projects/gait")
sys.path.insert(0, str(REPO))

from anysole.data.dataset import AnySoleDataset, collate_windows
from anysole.eval import _load_model, load_config, resolve_device


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path,
                    default=Path("/data/fangyuxuan/projects/gait/AnysoleWorkspace/splits/default/splits.csv"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args(argv)
    device = resolve_device(args.device)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    base_cfg = load_config(Path("/data/fangyuxuan/projects/gait/configs/v1.yaml"))
    model = _load_model(ck, base_cfg, device).eval()

    ds = AnySoleDataset(
        mode="eval", seq_root=cfg["seq_root"], split_csv=args.split_csv,
        cache_root=cfg["cache_root"], window_length=int(cfg.get("tw", 20)),
        contact_method=cfg.get("contact_method", "tactile_abs"),
        no_imu=bool(cfg.get("no_imu", False)),
        v_input=cfg.get("v_input", "hrnet"), f2_repr=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_windows)
    with torch.inference_mode():
        raw = next(iter(loader))
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in raw.items()}
        B = batch["pose_gt"].shape[0]
        cid = torch.zeros(B, dtype=torch.long, device=device)
        t_tac = torch.cat([batch["T_raw"], batch["T_phys"]], dim=-1)
        v_tok = model.encoders.v_enc(batch["V_feat"])
        t_tok = model.encoders.t_enc(t_tac)
        stream = model.encoders.t_enc.encode_stream(t_tac)

    v_mag = v_tok.norm(dim=-1).mean().item()
    t_mag = t_tok.norm(dim=-1).mean().item()
    ratio_vt = v_mag / max(t_mag, 1e-6)
    stream_mag = stream.norm(dim=-1).mean().item()
    ratio_stream_fwd = stream_mag / max(t_mag, 1e-6)
    dim = v_tok.shape[-1]

    print("|v_tok| = %.3f  |t_tok| = %.3f  ratio v/t = %.3f" % (v_mag, t_mag, ratio_vt))
    print("|stream| = %.3f  ratio stream/forward = %.3f  (sqrt(dim)=%.2f)"
          % (stream_mag, ratio_stream_fwd, dim ** 0.5))
    assert 1.0 <= ratio_vt <= 100.0, "v/t token magnitude ratio out of [1,100]: %.3f" % ratio_vt
    assert 0.5 <= ratio_stream_fwd <= 50.0, \
        "encode_stream/forward magnitude ratio out of [0.5,50]: %.3f" % ratio_stream_fwd
    assert 0.5 * dim ** 0.5 <= stream_mag <= 2.0 * dim ** 0.5, \
        "encode_stream magnitude %.3f not ~sqrt(dim) (LayerNorm contract broken)" % stream_mag
    print("smoke_token_scale: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
