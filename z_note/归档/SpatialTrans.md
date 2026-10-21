"""
insole_icstn.py  —  SE(2) Inverse-Compositional alignment for 4 × 12 tactile insoles

Drop-in before your main encoder:
    compensator = InsoleDriftCompensator(template_L, template_R)
    T_aligned, theta_L, theta_R = compensator(T_raw)   # (B,T,96) → (B,T,96)

What changed vs stock IC-STN library (Lin & Lucey CVPR 2017):
  - Warp: affine grid_sample → SE(2) + Gaussian resampling in mm-space
  - Localization input: single frame → temporal max-envelope
  - Parameter space: 6-DOF affine → 3-DOF SE(2) with physical limits
  - Jacobian: fixed (IC property), finite-differenced from calibration template
The IC update loop structure (precomputed J, GN step, compose) is preserved.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════
# §1  Canonical taxel layout
# ═══════════════════════════════════════════════════════════════════════════

def make_insole_coords(
    n_rows: int         = 4,
    n_cols: int         = 12,
    sp_ml_mm: float     = 26.7,   # medial-lateral taxel pitch  ← measure yours
    sp_ap_mm: float     = 22.7,   # anterior-posterior taxel pitch
    device: str | torch.device = 'cpu',
) -> torch.Tensor:
    """
    Returns (N=48, 2) canonical positions [x_ml, y_ap] in mm, centred at 0.

    Assumed CSV layout (columns 1..48, row-major, left foot shown):
        row 0  lateral edge   ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●
        row 1                 ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●
        row 2                 ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●
        row 3  medial edge    ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●
                     col 0 (heel) ─────────────────── col 11 (toe)

    Measure sp_ml_mm and sp_ap_mm from the physical insole PCB or datasheet.
    Flip axis signs here if your row/col orientation differs from above.
    """
    ml = (torch.arange(n_rows, dtype=torch.float32) - (n_rows - 1) / 2.0) * sp_ml_mm
    ap = (torch.arange(n_cols, dtype=torch.float32) - (n_cols - 1) / 2.0) * sp_ap_mm
    g_ml, g_ap = torch.meshgrid(ml, ap, indexing='ij')    # each (4, 12)
    return torch.stack([g_ml.flatten(), g_ap.flatten()], dim=-1).to(device)  # (48, 2)


# ═══════════════════════════════════════════════════════════════════════════
# §2  SE(2) warp with differentiable Gaussian resampling
# ═══════════════════════════════════════════════════════════════════════════

class SE2Warp(nn.Module):
    """
    Physics model: the insole has slipped by theta = (tx_mm, ty_mm, alpha_rad)
    relative to the foot.  Taxel i, nominally at canonical position c_i,
    now sits at anatomical location  R(alpha)·c_i + t  and reads pressure there.
    We Gaussian-average those slipped readings back onto the canonical grid:

        output_k = Σ_i  ker(c_k, w_i) · p_i  /  Σ_i  ker(c_k, w_i)
        where  w_i = R·c_i + t  (slipped sensor position in anatomical frame)
        and    ker(a, b) = exp(−||a − b||² / 2σ²)

    sigma_mm ≈ 1× taxel spacing is a good starting point.
    The kernel keeps gradients non-zero even for sub-taxel offsets.
    """

    def __init__(self, sigma_mm: float = 22.0, **grid_kw):
        super().__init__()
        c = make_insole_coords(**grid_kw)           # (N, 2)
        self.register_buffer('c', c)
        self.inv2s2  = 1.0 / (2.0 * sigma_mm ** 2)
        self.sigma_mm = sigma_mm

    # ------------------------------------------------------------------
    def forward(self, p: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        p     : (B, N)  raw sensor readings  (any positive normalisation)
        theta : (B, 3)  [tx_mm, ty_mm, alpha_rad]
        →       (B, N)  pressure at canonical anatomical positions
        """
        c          = self.c
        tx, ty, a  = theta[:, 0], theta[:, 1], theta[:, 2]
        ca, sa     = torch.cos(a), torch.sin(a)

        # Slipped position of sensor i in anatomical frame: w_i = R·c_i + t
        wx    = ca[:, None] * c[:, 0] - sa[:, None] * c[:, 1] + tx[:, None]  # (B, N)
        wy    = sa[:, None] * c[:, 0] + ca[:, None] * c[:, 1] + ty[:, None]
        w_pos = torch.stack([wx, wy], dim=-1)                                 # (B, N, 2)

        # Kernel weights:  ker[b, k, i] = exp(−||c_k − w_i||² / 2σ²)
        diff  = c[None, :, None, :] - w_pos[:, None, :, :]   # (B, N_k, N_i, 2)
        d2    = diff.pow(2).sum(-1)                           # (B, N_k, N_i)
        ker   = torch.exp(-d2 * self.inv2s2)                  # (B, N_k, N_i)
        num   = (ker * p[:, None, :]).sum(-1)                 # (B, N_k)
        den   = ker.sum(-1) + 1e-6
        return num / den                                      # (B, N_k)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def jacobian_at_zero(
        self,
        p_tpl: torch.Tensor,                           # (N,)  normalised template
        eps: tuple[float, float, float] = (1.0, 1.0, 5e-3),
    ) -> torch.Tensor:
        """
        Finite-difference Jacobian  d(output)/d(theta)  at theta=0.
        eps units: (mm, mm, rad).  Called once at init with calibration template.
        Returns J : (N, 3).

        This is the IC "steepest-descent image": computed from the template
        and frozen for the life of the module.  Changing the template
        (new subject) requires re-instantiating FootAlignmentSTN.
        """
        p = p_tpl.unsqueeze(0)  # (1, N)
        J = torch.empty(p_tpl.shape[0], 3, device=p_tpl.device)
        for j, e in enumerate(eps):
            tp = torch.zeros(1, 3, device=p_tpl.device); tp[0, j] =  e
            tm = torch.zeros(1, 3, device=p_tpl.device); tm[0, j] = -e
            J[:, j] = (self.forward(p, tp) - self.forward(p, tm)).squeeze(0) / (2.0 * e)
        return J  # (N, 3)


# ═══════════════════════════════════════════════════════════════════════════
# §3  Precomputed IC Gauss-Newton step
# ═══════════════════════════════════════════════════════════════════════════

class ICStep(nn.Module):
    """
    One GN step with the frozen Jacobian J (the IC trick):
        Δθ = −(JᵀJ + λ·diag(JᵀJ))⁻¹ · Jᵀ · residual

    The matrix solve is done once at init; forward is a single matmul.
    lam provides Levenberg-Marquardt damping (keeps (JᵀJ)⁻¹ stable when
    some warp directions have very small gradients, e.g. near the arch).
    """

    def __init__(self, J: torch.Tensor, lam: float = 1e-2):
        super().__init__()
        H   = J.T @ J                                   # (3, 3)
        reg = lam * H.diag().mean() * torch.eye(3, device=J.device)
        SD  = torch.linalg.solve(H + reg, J.T)          # (3, N)  steepest descent
        self.register_buffer('SD', SD)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        """
        residual : (B, N)   =  warp(observation, theta) − template
        →          (B, 3)   Δθ  (add to current theta to reduce residual)
        """
        return -(residual @ self.SD.T)                  # (B, N) @ (N, 3) → (B, 3)


# ═══════════════════════════════════════════════════════════════════════════
# §4  SE(2) forward composition
# ═══════════════════════════════════════════════════════════════════════════

def se2_compose(theta: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    theta_new = theta ⊕ delta   (SE(2) group multiplication).
    delta is expressed in theta's rotated frame (forward composition).
    Both tensors: (B, 3)  [tx_mm, ty_mm, alpha_rad].
    """
    tx, ty, a  = theta[:, 0], theta[:, 1], theta[:, 2]
    dx, dy, da = delta[:, 0], delta[:, 1], delta[:, 2]
    ca, sa     = torch.cos(a), torch.sin(a)
    return torch.stack([
        tx + ca * dx - sa * dy,
        ty + sa * dx + ca * dy,
        a  + da,
    ], dim=1)


# ═══════════════════════════════════════════════════════════════════════════
# §5  Per-foot alignment module
# ═══════════════════════════════════════════════════════════════════════════

class FootAlignmentSTN(nn.Module):
    """
    Full IC-STN pipeline for one foot (48 taxels, T frames per window):

      ① Max-envelope   — drift anchor insensitive to COP excursions
      ② Localization MLP  → coarse θ
      ③ N_iter IC GN steps → refined θ  (frozen Jacobian, O(48×3) per step)
      ④ Shared θ applied frame-by-frame in the window

    Key design constraints:
      - theta is shared across all T frames (drift is a slow variable)
      - tanh clamping enforces physical limits at every step
      - Jacobian is from the TEMPLATE (IC trick), never retrained
    """

    def __init__(
        self,
        p_template: torch.Tensor,      # (48,)  canonical footprint
        n_iter: int          = 4,
        tx_lim_mm: float     = 15.0,
        ty_lim_mm: float     = 8.0,
        alpha_lim_deg: float = 8.0,
        lam_ic: float        = 1e-2,
        **warp_kw,                     # forwarded to SE2Warp: sigma_mm, sp_*_mm
    ):
        super().__init__()
        a_lim = math.radians(alpha_lim_deg)
        self.register_buffer('lim', torch.tensor([tx_lim_mm, ty_lim_mm, a_lim]))
        self.n_iter = n_iter

        self.warp = SE2Warp(**warp_kw)

        p_tpl = p_template / (p_template.max() + 1e-6)
        self.register_buffer('template', p_tpl)

        # Jacobian precomputed from template, never retrained
        J = self.warp.jacobian_at_zero(p_tpl.to(self.warp.c.device))
        self.ic_step = ICStep(J, lam=lam_ic)

        N = p_tpl.shape[0]
        self.loc_net = nn.Sequential(
            nn.LayerNorm(N),
            nn.Linear(N, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32,  3),
        )
        # Zero-init last layer → IC refines from near-zero start in early training
        nn.init.zeros_(self.loc_net[-1].weight)
        nn.init.zeros_(self.loc_net[-1].bias)

    # ------------------------------------------------------------------
    def _clamp(self, th: torch.Tensor) -> torch.Tensor:
        """Smooth physical constraint: tanh saturation at each limit."""
        return torch.tanh(th / self.lim) * self.lim

    # ------------------------------------------------------------------
    def forward(self, pressure_seq: torch.Tensor):
        """
        pressure_seq : (B, T, 48)
        Returns (aligned_seq (B,T,48), theta (B,3)).
        theta is in mm (tx, ty) and radians (alpha).
        """
        B, T, N = pressure_seq.shape

        # ① Max over window — slow drift variable; COP oscillation cancels out
        env   = pressure_seq.amax(dim=1)                          # (B, 48)
        env_n = env / (env.amax(dim=1, keepdim=True) + 1e-6)     # normalise [0,1]

        # ② Coarse theta from localization MLP
        theta = self._clamp(self.loc_net(env_n))                  # (B, 3)

        # ③ IC GN refinement (frozen Jacobian, fast; 4 iter usually sufficient)
        for _ in range(self.n_iter):
            aligned_env = self.warp(env_n, theta)                 # (B, 48)
            residual    = aligned_env - self.template             # (B, 48)
            delta       = self.ic_step(residual)                  # (B, 3)
            theta       = self._clamp(se2_compose(theta, delta))

        # ④ Apply shared theta to all T frames
        p_flat   = pressure_seq.reshape(B * T, N)
        th_exp   = theta.unsqueeze(1).expand(B, T, 3).reshape(B * T, 3)
        aligned  = self.warp(p_flat, th_exp).reshape(B, T, N)

        return aligned, theta


# ═══════════════════════════════════════════════════════════════════════════
# §6  Dual-foot wrapper  —  (B, T, 96) → (B, T, 96)
# ═══════════════════════════════════════════════════════════════════════════

class InsoleDriftCompensator(nn.Module):
    """
    Drop-in before your main tactile encoder.

    Usage:
        coords = make_insole_coords(sp_ml_mm=..., sp_ap_mm=...)
        tpl_L  = build_template(cal_seq_L)   # from calibration trial
        tpl_R  = build_template(cal_seq_R)
        model  = InsoleDriftCompensator(tpl_L, tpl_R, sigma_mm=22.0,
                                        sp_ml_mm=..., sp_ap_mm=...)
        T_aligned, theta_L, theta_R = model(T_raw)

    template_L, template_R : (48,) from build_template().
    **foot_kw is forwarded to FootAlignmentSTN and SE2Warp.
    """

    def __init__(
        self,
        template_L: torch.Tensor,
        template_R: torch.Tensor,
        **foot_kw,
    ):
        super().__init__()
        self.stn_L = FootAlignmentSTN(template_L, **foot_kw)
        self.stn_R = FootAlignmentSTN(template_R, **foot_kw)

    def forward(self, T_raw: torch.Tensor):
        """
        T_raw : (B, T, 96)  left 48 columns + right 48 columns at 40 Hz
        Returns T_aligned (B,T,96), theta_L (B,3), theta_R (B,3).
        Log theta_L / theta_R during training as a sanity check.
        """
        L, R          = T_raw[..., :48], T_raw[..., 48:]
        L_al, theta_L = self.stn_L(L)
        R_al, theta_R = self.stn_R(R)
        return torch.cat([L_al, R_al], dim=-1), theta_L, theta_R


# ═══════════════════════════════════════════════════════════════════════════
# §7  Calibration — build canonical footprint template
# ═══════════════════════════════════════════════════════════════════════════

def build_template(
    cal_seq: torch.Tensor,   # (T_cal, N)  3–5 s static standing, insole freshly aligned
    q: float = 0.95,         # high quantile captures foot outline, suppresses noise
) -> torch.Tensor:
    """
    Returns (N,) normalised canonical footprint for one foot.
    Collect calibration data immediately after donning and marking the insole
    position.  If the subject walks during calibration, use the max-envelope
    of a full gait cycle instead (q=1.0).
    """
    tpl = torch.quantile(cal_seq.float(), q, dim=0)      # (N,)
    return (tpl / (tpl.max() + 1e-6)).float()


# ═══════════════════════════════════════════════════════════════════════════
# §8  Synthetic drift injection  (supervised pre-training utility)
# ═══════════════════════════════════════════════════════════════════════════

def inject_drift(
    p_clean: torch.Tensor,              # (B, T, N)  clean canonical data
    coords: torch.Tensor,               # (N, 2)  from make_insole_coords()
    sigma_mm: float,
    tx_range:  tuple[float, float] = (-12.0, 12.0),  # mm
    ty_range:  tuple[float, float] = ( -6.0,  6.0),  # mm
    a_range:   tuple[float, float] = (-0.12, 0.12),   # rad  ≈ ±6.9°
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simulates random insole slippage on clean data.

    After drift by theta, sensor k sits at anatomical position R·c_k + t
    and samples the pressure field at that location.  This is the forward
    model of what the sensors physically record after slippage.

    Returns (drifted (B,T,N), gt_theta (B,3)).
    Invariant: SE2Warp.forward(drifted, gt_theta)  ≈  p_clean  (roundtrip check).

    Pre-training recipe:
        1. Inject drift → supervised regression to gt_theta  (fast convergence)
        2. Freeze loc_net, unfreeze IC step  → template alignment fine-tune
        3. Plug into full system, unfreeze all  → task loss end-to-end
    """
    B, T, N = p_clean.shape
    dev     = p_clean.device
    inv2s2  = 1.0 / (2.0 * sigma_mm ** 2)

    tx = torch.empty(B, device=dev).uniform_(*tx_range)
    ty = torch.empty(B, device=dev).uniform_(*ty_range)
    a  = torch.empty(B, device=dev).uniform_(*a_range)
    gt_theta = torch.stack([tx, ty, a], dim=-1)            # (B, 3)

    ca, sa = torch.cos(a), torch.sin(a)
    # Slipped anatomical positions: s_k = R·c_k + t
    sx = ca[:, None] * coords[:, 0] - sa[:, None] * coords[:, 1] + tx[:, None]  # (B,N)
    sy = sa[:, None] * coords[:, 0] + ca[:, None] * coords[:, 1] + ty[:, None]
    slipped = torch.stack([sx, sy], dim=-1)                # (B, N, 2)

    # Gaussian interp of clean pressure at slipped positions
    # diff[b, k, j] = slipped[b,k] − coords[j]
    diff  = slipped[:, :, None, :] - coords[None, None, :, :]  # (B, N_k, N_j, 2)
    d2    = diff.pow(2).sum(-1)                                 # (B, N_k, N_j)
    ker   = torch.exp(-d2 * inv2s2)                            # (B, N_k, N_j)

    # Apply shared theta to all T frames
    # num[b, t, k] = Σ_j ker[b,k,j] * p_clean[b,t,j]
    num = (ker[:, None, :, :] * p_clean[:, :, None, :]).sum(-1)  # (B, T, N_k)
    den = ker.sum(-1)[:, None, :] + 1e-6                         # (B, 1, N_k)
    return num / den, gt_theta                                   # (B,T,N), (B,3)


# ═══════════════════════════════════════════════════════════════════════════
# §9  Training losses
# ═══════════════════════════════════════════════════════════════════════════

def loss_template_soft_iou(
    aligned_env: torch.Tensor,     # (B, N)  (will be re-normalised internally)
    template:    torch.Tensor,     # (N,)
) -> torch.Tensor:
    """
    1 − soft-IoU between aligned observation and canonical template.
    Amplitude-invariant: robust to pressure differences across subjects.
    Use in place of MSE for the template alignment term.
    """
    a = aligned_env / (aligned_env.amax(dim=1, keepdim=True) + 1e-6)
    t = template.unsqueeze(0)
    inter = (a * t).sum(-1)
    union = (a + t - a * t).sum(-1) + 1e-6
    return (1.0 - inter / union).mean()


def loss_theta_smooth(
    theta_cur:  torch.Tensor,      # (B, 3)  current window
    theta_prev: torch.Tensor,      # (B, 3)  previous window
) -> torch.Tensor:
    """Temporal smoothness: penalises jump between consecutive window estimates."""
    return (theta_cur - theta_prev).pow(2).mean()


def loss_theta_amplitude(theta: torch.Tensor) -> torch.Tensor:
    """Amplitude prior: 'drift is usually small'.  Prevents large-offset solutions."""
    return theta.pow(2).mean()


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test  —  python insole_icstn.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    torch.manual_seed(0)
    B, T = 2, 20

    # Fake calibration data (replace with real standing-trial CSV)
    tpl_L = build_template(torch.rand(100, 48))
    tpl_R = build_template(torch.rand(100, 48))

    model = InsoleDriftCompensator(
        template_L   = tpl_L,
        template_R   = tpl_R,
        sigma_mm     = 22.0,    # ≈ 1× taxel spacing; tune down if over-smoothed
        sp_ml_mm     = 26.7,    # replace with measured hardware value
        sp_ap_mm     = 22.7,
    )

    T_raw = torch.rand(B, T, 96)
    T_al, thL, thR = model(T_raw)
    assert T_al.shape == T_raw.shape
    print(f"Input  shape : {T_raw.shape}")
    print(f"Output shape : {T_al.shape}")
    print(f"theta_L (mm, mm, deg): "
          f"tx={thL[:,0].tolist()}  "
          f"ty={thL[:,1].tolist()}  "
          f"α={[round(x*57.3,2) for x in thL[:,2].tolist()]}")

    # ── Roundtrip check ──────────────────────────────────────────────────
    # Inject known drift → compensate with GT theta → measure recovery error.
    # Expect error << 0.05 for drifts within training range.
    coords  = make_insole_coords()
    clean_L = torch.rand(B, T, 48).clamp(0, 1)

    drifted, gt = inject_drift(
        clean_L, coords, sigma_mm=22.0,
        tx_range=(-5, 5), ty_range=(-3, 3), a_range=(-0.05, 0.05),
    )
    warp    = model.stn_L.warp
    p_flat  = drifted.reshape(B * T, 48)
    gt_rep  = gt.unsqueeze(1).expand(B, T, 3).reshape(B * T, 3)
    recovered = warp(p_flat, gt_rep).reshape(B, T, 48)

    err = (recovered - clean_L).abs().mean()
    print(f"\nRoundtrip MAE with GT theta : {err:.4f}  (expect < 0.05)")
    print(f"Drift injected (sample 0)   : "
          f"tx={gt[0,0]:.1f}mm  ty={gt[0,1]:.1f}mm  α={gt[0,2]*57.3:.1f}°")