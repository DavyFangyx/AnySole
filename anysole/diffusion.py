"""Handwritten cosine Gaussian diffusion and DDIM sampling (no diffusers)."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from anysole.types import DIFFUSION_SAMPLE_STEPS, DIFFUSION_TRAIN_STEPS, POSE_DIM, TW


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    raise NotImplementedError("unknown beta schedule: %s" % schedule_name)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def _extract(arr, tau, x_shape):
    """Gather a 1-D schedule at batch timesteps and broadcast to x_shape.

    Returns float32 of shape (B, 1, 1, ...) matching ``x_shape``.
    """
    if isinstance(arr, np.ndarray):
        values = torch.from_numpy(arr).to(device=tau.device, dtype=torch.float32)
    else:
        values = arr.to(device=tau.device, dtype=torch.float32)
    out = values[tau.long()]
    while out.ndim < len(x_shape):
        out = out[..., None]
    return out.expand(x_shape)


def _timestep_schedule(n_train_steps, steps):
    """Uniformly pick ``steps`` indices from ``n_train_steps-1`` down, then keep 0."""
    if steps <= 1:
        return [0]
    start = n_train_steps - 1
    ts = np.linspace(start, 0, num=steps, dtype=np.float64)
    ts = np.round(ts).astype(np.int64)
    seen = set()
    ordered = []
    for t in ts.tolist():
        if t not in seen:
            seen.add(t)
            ordered.append(int(t))
    if ordered[-1] != 0:
        ordered.append(0)
    return ordered


class GaussianDiffusion:
    def __init__(self, n_train_steps=DIFFUSION_TRAIN_STEPS, schedule="cosine"):
        self.n_train_steps = int(n_train_steps)
        self.schedule = schedule
        betas = np.array(get_named_beta_schedule(schedule, self.n_train_steps), dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

    def q_sample(self, x0, tau, noise=None) -> Tensor:
        """x0 (B,20,138), tau (B,) long in [0, n_train_steps-1] -> x_tau same shape."""
        if noise is None:
            noise = torch.randn_like(x0)
        if noise.shape != x0.shape:
            raise ValueError("noise shape %s != x0 shape %s" % (tuple(noise.shape), tuple(x0.shape)))
        sqrt_abar = _extract(self.sqrt_alphas_cumprod, tau, x0.shape)
        sqrt_one_minus = _extract(self.sqrt_one_minus_alphas_cumprod, tau, x0.shape)
        return sqrt_abar * x0 + sqrt_one_minus * noise

    def predict_eps_from_x0(self, x_tau, tau, x0_hat) -> Tensor:
        sqrt_abar = _extract(self.sqrt_alphas_cumprod, tau, x_tau.shape)
        sqrt_one_minus = _extract(self.sqrt_one_minus_alphas_cumprod, tau, x_tau.shape)
        return (x_tau - sqrt_abar * x0_hat) / sqrt_one_minus

    def training_losses(self, model, batch, config_id) -> dict:
        """Add noise to pose_gt and run the model. Geometry losses live in losses.py."""
        pose_gt = batch["pose_gt"]
        batch_size = pose_gt.shape[0]
        device = pose_gt.device
        tau = torch.randint(0, self.n_train_steps, (batch_size,), device=device, dtype=torch.long)
        noise = torch.randn_like(pose_gt)
        x_tau = self.q_sample(pose_gt, tau, noise)
        out = model(
            batch["V_feat"],
            batch["T_raw"],
            batch["T_phys"],
            x_tau,
            tau,
            config_id,
        )
        out = dict(out)
        out["tau"] = tau
        out["noise"] = noise
        out["x_tau"] = x_tau
        return out

    def _call_model(self, model, x_tau, tau, cond):
        if all(k in cond for k in ("V_feat", "T_raw", "T_phys", "config_id")):
            return model(
                cond["V_feat"],
                cond["T_raw"],
                cond["T_phys"],
                x_tau,
                tau,
                cond["config_id"],
            )
        if "F" in cond:
            return model(x_tau, tau, cond["F"])
        return model(x_tau, tau)

    def _unpack_x0(self, out, x_tau):
        if isinstance(out, dict):
            if "x0_hat" not in out:
                raise KeyError("model output dict is missing x0_hat")
            x0_hat = out["x0_hat"]
        else:
            x0_hat = out
        if x0_hat.shape != x_tau.shape:
            raise ValueError(
                "x0_hat shape %s != x_tau shape %s" % (tuple(x0_hat.shape), tuple(x_tau.shape))
            )
        return x0_hat

    def ddim_step(self, x_tau, tau, tau_prev, x0_hat, eta=0.0, noise=None) -> Tensor:
        """One DDIM update from tau -> tau_prev (MDM / DDIM equation 12)."""
        eps = self.predict_eps_from_x0(x_tau, tau, x0_hat)
        abar = _extract(self.alphas_cumprod, tau, x_tau.shape)
        abar_prev = _extract(self.alphas_cumprod, tau_prev, x_tau.shape)
        sigma = (
            eta
            * torch.sqrt((1.0 - abar_prev) / (1.0 - abar))
            * torch.sqrt(1.0 - abar / abar_prev)
        )
        dir_coeff = torch.sqrt(torch.clamp(1.0 - abar_prev - sigma ** 2, min=0.0))
        x_prev = torch.sqrt(abar_prev) * x0_hat + dir_coeff * eps
        if float(eta) != 0.0:
            if noise is None:
                noise = torch.randn_like(x_tau)
            nonzero = (tau != 0).float().view(-1, *([1] * (x_tau.ndim - 1)))
            x_prev = x_prev + nonzero * sigma * noise
        return x_prev

    def ddim_sample_loop(
        self,
        model,
        x_T=None,
        tau_related_kwargs=None,
        shape=None,
        steps=DIFFUSION_SAMPLE_STEPS,
        eta=0.0,
        device=None,
    ) -> Tensor:
        """Deterministic DDIM. model is called like AnySoleModel.forward."""
        cond = dict(tau_related_kwargs or {})
        if x_T is None:
            if shape is None:
                if "V_feat" in cond:
                    batch_size = cond["V_feat"].shape[0]
                    if device is None:
                        device = cond["V_feat"].device
                    shape = (batch_size, TW, POSE_DIM)
                else:
                    raise ValueError("ddim_sample_loop needs x_T or shape")
            if device is None:
                try:
                    device = next(model.parameters()).device
                except (StopIteration, AttributeError):
                    device = torch.device("cpu")
            x_T = torch.randn(*shape, device=device)
        x = x_T
        batch_size = x.shape[0]
        device = x.device

        timesteps = _timestep_schedule(self.n_train_steps, int(steps))
        for i, t in enumerate(timesteps):
            tau = torch.full((batch_size,), int(t), device=device, dtype=torch.long)
            out = self._call_model(model, x, tau, cond)
            x0_hat = self._unpack_x0(out, x)
            if int(t) == 0:
                return x0_hat
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            tau_prev = torch.full((batch_size,), int(t_prev), device=device, dtype=torch.long)
            x = self.ddim_step(x, tau, tau_prev, x0_hat, eta=eta)
        return x

    def ddim_sample(self, model, F_or_inputs, shape, steps=DIFFUSION_SAMPLE_STEPS, eta=0.0) -> Tensor:
        """Return x0 (B,20,138). eta defaults to 0."""
        if isinstance(F_or_inputs, dict):
            cond = dict(F_or_inputs)
        else:
            cond = {"F": F_or_inputs}
        device = None
        if "V_feat" in cond:
            device = cond["V_feat"].device
        elif "F" in cond and torch.is_tensor(cond["F"]):
            device = cond["F"].device
        if device is None:
            try:
                device = next(model.parameters()).device
            except (StopIteration, AttributeError):
                device = torch.device("cpu")
        x_T = torch.randn(*shape, device=device)
        return self.ddim_sample_loop(
            model,
            x_T=x_T,
            tau_related_kwargs=cond,
            shape=shape,
            steps=steps,
            eta=eta,
            device=device,
        )
