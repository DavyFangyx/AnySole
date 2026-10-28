"""RGB-only B1 teacher with shared A3 encoding and ordered V/Q set fusion.

Camera geometry is an external, normalized cam1-relative condition. There are
no camera-index embeddings, pose labels, tactile measurements or student
parameters in this model. View dropout precedes encoding, so the returned
selection describes the RGB windows actually consumed.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from anysole.models.common.embeddings import relative_time_pe
from anysole.models.common.fusion import _transformer_encoder
from anysole.models.common.model_v2 import AnySoleModelV2
from anysole.types import POSE_DIM, V_FEAT_DIM


class CrossViewBlock(nn.Module):
    """One time/stream query reads one source stream across valid views."""

    def __init__(self, dim, nhead, dropout):
        super().__init__()
        self.stream_query = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.normal_(self.stream_query, std=.02)
        self.attention = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(4 * dim, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, content, geometry, valid, time):
        # content/valid are [G*T,C,d]/[G*T,C]. Never invoke attention on
        # all-missing timestamps, even if attention has learned bias terms.
        active = valid.any(-1)
        result = content.new_zeros(content.shape[0], content.shape[-1])
        if bool(active.any()):
            query = time[active, None] + self.stream_query.to(dtype=content.dtype)
            values = torch.where(valid[active, :, None], content[active],
                                 torch.zeros_like(content[active]))
            keys = values + geometry[active]
            attended, _ = self.attention(query, keys, values,
                                          key_padding_mask=~valid[active], need_weights=False)
            hidden = query + self.dropout(attended)
            hidden = hidden + self.dropout(self.ffn(self.norm1(hidden)))
            result[active] = self.norm2(hidden[:, 0])
        return result


class MultiviewMemoryFusion(nn.Module):
    """Per-stream cross-view attention followed by one temporal refinement."""

    def __init__(self, dim=256, tw=20, nhead=8, dropout=.1):
        super().__init__()
        self.dim, self.tw = dim, tw
        self.geometry_encoder = nn.Sequential(nn.Linear(16, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.streams = nn.ModuleList([CrossViewBlock(dim, nhead, dropout) for _ in range(2)])
        self.refinement = _transformer_encoder(dim, 1, nhead, 4 * dim, dropout)

    def forward(self, memory, frame_valid, timestamps, geometry):
        g, c, _, _ = memory.shape
        valid_time = frame_valid.any(1)
        view_valid = frame_valid.any(-1)
        safe_geometry = torch.where(view_valid[..., None], geometry, torch.zeros_like(geometry))
        geo = self.geometry_encoder(safe_geometry.to(dtype=memory.dtype))
        geo = torch.where(view_valid[..., None], geo, torch.zeros_like(geo))
        geo = geo[:, None].expand(g, self.tw, c, self.dim).reshape(g * self.tw, c, self.dim)
        valid = frame_valid.permute(0, 2, 1).reshape(g * self.tw, c)
        time = relative_time_pe(timestamps, self.dim, 40., memory.dtype).reshape(g * self.tw, self.dim)
        source = memory.reshape(g, c, 2, self.tw, self.dim)
        aggregated = []
        for stream_id, block in enumerate(self.streams):
            values = source[:, :, stream_id].permute(0, 2, 1, 3).reshape(g * self.tw, c, self.dim)
            aggregated.append(block(values, geo, valid, time).reshape(g, self.tw, self.dim))
        ordered = torch.cat(aggregated, dim=1)
        padding = torch.cat([~valid_time, ~valid_time], dim=1)
        fused = torch.zeros_like(ordered)
        active = valid_time.any(-1)
        if bool(active.any()):
            mask = padding[active]
            refined = self.refinement(ordered[active], src_key_padding_mask=mask if bool(mask.any()) else None)
            fused[active] = torch.where(mask[..., None], torch.zeros_like(refined), refined)
        return fused, aggregated[0], aggregated[1]


class AnySoleMultiviewTeacher(AnySoleModelV2):
    """A3 task heads run once on a genuinely fused multiview memory."""

    NEW_STATE_PREFIX = "multiview_fusion."

    def __init__(self, *args, three_view_probability=.7, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 <= float(three_view_probability) <= 1:
            raise ValueError("three_view_probability must be in [0,1]")
        self.three_view_probability = float(three_view_probability)
        self.multiview_fusion = MultiviewMemoryFusion(
            self.d, self.tw, kwargs.get("nhead", 8), kwargs.get("dropout", .1))

    def initialize_from_baseline(self, baseline):
        """Copy all and only compatible B0 tensors, without sharing objects.

        Deliberately reject partial/mismatched state dictionaries. Only the
        new fusion module may be absent; same-stage teacher resumes instead
        use normal strict ``load_state_dict`` including that module.
        """
        source = baseline.state_dict() if isinstance(baseline, nn.Module) else baseline
        if not isinstance(source, Mapping):
            raise TypeError("baseline must be a model or complete model state_dict")
        target = self.state_dict()
        expected = {k for k in target if not k.startswith(self.NEW_STATE_PREFIX)}
        missing, extra = expected - set(source), set(source) - expected
        mismatch = sorted(k for k in expected & set(source)
                          if not torch.is_tensor(source[k]) or source[k].shape != target[k].shape)
        if missing or extra or mismatch:
            raise ValueError("incompatible B0 initialization: missing=%s extra=%s shape=%s" %
                             (sorted(missing), sorted(extra), mismatch))
        result = self.load_state_dict(source, strict=False)
        allowed_missing = {k for k in target if k.startswith(self.NEW_STATE_PREFIX)}
        if set(result.missing_keys) != allowed_missing or result.unexpected_keys:
            raise RuntimeError("B0 copy did not match the explicit teacher fusion boundary")
        return {"copied_keys": sorted(expected), "new_keys": sorted(allowed_missing)}

    def select_views(self, frame_valid):
        """Draw one whole-window subset; callable before denominator counting."""
        if frame_valid.ndim != 3 or frame_valid.dtype != torch.bool or not 1 <= frame_valid.shape[1] <= 3:
            raise ValueError("frame_valid must be bool [G,C,T] with one to three views")
        selected = frame_valid.any(-1).clone()
        candidates = (selected.sum(-1) == 3).nonzero(as_tuple=True)[0]
        if self.training and candidates.numel():
            drop = torch.rand(candidates.numel(), device=frame_valid.device) >= self.three_view_probability
            rows = candidates[drop]
            if rows.numel():
                columns = torch.randint(3, (rows.numel(),), device=frame_valid.device)
                selected[rows, columns] = False
        return frame_valid & selected[..., None], selected

    def forward_multiview(self, V_feat, frame_valid, timestamps, geometry, selected_view_valid=None):
        """Predict in cam1 coordinates from synchronized RGB/geometry only."""
        if (V_feat.ndim != 4 or V_feat.shape[2:] != (self.tw, V_FEAT_DIM)
                or not V_feat.is_floating_point() or V_feat.shape[0] == 0):
            raise ValueError("V_feat must be nonempty floating [G,C,T,2051]")
        g, c, t, _ = V_feat.shape
        if not 1 <= c <= 3 or frame_valid.shape != (g, c, t) or frame_valid.dtype != torch.bool:
            raise ValueError("frame_valid must be bool [G,C,T] with one to three views")
        if timestamps.shape != (g, t) or not timestamps.is_floating_point():
            raise ValueError("timestamps must be floating [G,T]")
        if geometry.shape != (g, c, 16) or not geometry.is_floating_point():
            raise ValueError("geometry must be normalized floating [G,C,16] relative to cam1")
        if any(x.device != V_feat.device for x in (frame_valid, timestamps, geometry)):
            raise ValueError("teacher RGB, mask, timestamps and geometry must share a device")
        if (not bool(torch.isfinite(timestamps).all())
                or not bool((timestamps[:, 1:] >= timestamps[:, :-1]).all())):
            raise ValueError("target timestamps must be finite and non-decreasing")
        if selected_view_valid is None:
            selected_view_valid, selected_views = self.select_views(frame_valid)
        else:
            if (selected_view_valid.shape != frame_valid.shape or selected_view_valid.dtype != torch.bool
                    or selected_view_valid.device != frame_valid.device):
                raise ValueError("selected_view_valid must be bool [G,C,T] on the RGB device")
            selected_views = selected_view_valid.any(-1)
            if not torch.equal(selected_view_valid, frame_valid & selected_views[..., None]):
                raise ValueError("selected_view_valid must select whole views from frame_valid")
        if not bool(torch.isfinite(V_feat[selected_view_valid]).all()):
            raise ValueError("valid visual features contain NaN or Inf")
        if not bool(torch.isfinite(geometry[selected_views]).all()):
            raise ValueError("selected camera geometry contains NaN or Inf")

        flat_valid = selected_view_valid.reshape(g * c, t)
        active = flat_valid.any(-1)
        if bool(active.any()):
            flat_times = timestamps[:, None].expand(g, c, t).reshape(g * c, t)
            encoded = self.encode_visual(V_feat.reshape(g * c, t, -1)[active], flat_valid[active], flat_times[active])
            memory = encoded["F"].new_zeros(g * c, 2 * t, self.d)
            memory[active] = encoded["F"]
        else:
            memory = V_feat.new_zeros(g * c, 2 * t, self.d)
        fused, v_tok, t_tok = self.multiview_fusion(
            memory.reshape(g, c, 2 * t, self.d), selected_view_valid, timestamps, geometry)
        union = selected_view_valid.any(1)
        groups = union.any(-1)
        out = self._empty_outputs(fused)
        if bool(groups.any()):
            decoded = self.decode_memory(fused[groups], union[groups], timestamps[groups],
                                         v_tok=v_tok[groups], t_tok=t_tok[groups])
            static = {"part_logits", "part_assignment", "pose_mean", "pose_std", "sigma_frozen", "x0_residual"}
            for key, value in decoded.items():
                if key in static:
                    out[key] = value
                elif torch.is_tensor(value):
                    axis = 1 if key == "part_hypotheses" else 0
                    shape = list(value.shape)
                    shape[axis] = g
                    full = value.new_zeros(shape)
                    if axis == 1:
                        full[:, groups] = value
                    else:
                        full[groups] = value
                    out[key] = full
        # Decoder query/output biases are never observations at a missing
        # timestamp. Mask every batch/time output after the final learned op.
        for key, value in out.items():
            if key == "part_hypotheses":
                out[key] = torch.where(union[None, :, :, None, None], value, torch.zeros_like(value))
            elif key not in ("part_logits", "part_assignment", "pose_mean", "pose_std") and torch.is_tensor(value):
                if value.ndim >= 2 and value.shape[:2] == union.shape:
                    mask = union.reshape(*union.shape, *((1,) * (value.ndim - 2)))
                    out[key] = torch.where(mask, value, torch.zeros_like(value))
        out.update(F=fused, v_tok=v_tok, t_tok=t_tok, frame_valid=union,
                   memory_padding_mask=torch.cat([~union, ~union], dim=1),
                   teacher_valid=selected_view_valid.sum(1) >= 2,
                   selected_view_valid=selected_view_valid, selected_views=selected_views,
                   view_counts=selected_view_valid.sum(1), rgb_window_counts=selected_views.sum(1))
        return out

    def _empty_outputs(self, memory):
        g = memory.shape[0]
        zeros = lambda *shape: memory.new_zeros(*shape)
        out = {"x0_hat": zeros(g, self.tw, POSE_DIM), "v_hat": zeros(g, self.tw, 3),
               "trans_hat": zeros(g, self.tw, 3), "pressure_hat": zeros(g, self.tw, 96),
               "vfeat_hat": zeros(g, self.tw, V_FEAT_DIM),
               "part_logits": self.pose_head.part_logits if self.soft_parts else None,
               "part_assignment": self.pose_head.part_assignment() if self.soft_parts else None,
               "x0_residual": None}
        if self.gate == "sigma":
            p = self.pose_head.n_parts
            out.update(gate_sigma=zeros(g, self.tw, p, 2), gate_g=zeros(g, self.tw, p, 3),
                       part_hypotheses=zeros(p, g, self.tw, 24, 6),
                       pose_mean=self.pose_head.pose_mean, pose_std=self.pose_head.pose_std,
                       sigma_frozen=bool(self.pose_head.decoder.layers[-1].sigma_frozen))
        if hasattr(self, "pressure_readout"):
            out.update(pressure_query_valid=torch.zeros(g, self.tw, 2, dtype=torch.bool, device=memory.device),
                       pressure_contact_logits=zeros(g, self.tw, 2), pressure_total=zeros(g, self.tw, 2),
                       pressure_allocation=zeros(g, self.tw, 2, 48))
        return out
