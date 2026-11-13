"""文档 16 v1（SE(2) 区间平滑器）数学与合成场景验收。

运行：python -m pytest tests/test_vtbridge.py -q
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VT_DIR = os.path.join(HERE, "..", "results", "diagnostics",
                      "20261110_vtbridge")
sys.path.insert(0, os.path.abspath(VT_DIR))

import se2_smoother as S  # noqa: E402

DT = S.DT


def _assemble_vec(v, psi):
    """hybrid_sceneocc_eval._assemble_vec 的独立拷贝。"""
    return np.array([v[0] * np.sin(psi) - v[1] * np.cos(psi),
                     v[0] * np.cos(psi) + v[1] * np.sin(psi)])


def _make_truth(K, speed=1.2, turn_rate=0.0, lat=0.0, psi_start=0.3,
                seed=0):
    """生成精确 GT 与观测（无噪声）。返回 d, dyaw, p, psi。"""
    rng = np.random.RandomState(seed)
    psi = np.empty(K + 1)
    psi[0] = psi_start
    for j in range(K):
        psi[j + 1] = psi[j] + turn_rate * DT
    # 每步速度有小幅变化，保证不是纯直线
    sp = speed * (1.0 + 0.15 * np.sin(np.arange(K) * 0.7 + rng.rand()))
    d = np.stack([sp * DT, np.full(K, lat * DT)], -1)
    p = np.empty((K + 1, 2))
    p[0] = np.array([0.2, -0.4])
    for j in range(K):
        p[j + 1] = p[j] + S.frame_to_world(d[j], psi[j])
    dyaw = np.diff(psi)
    return d, dyaw, p, psi


def _cfg(**kw):
    base = dict(opt_yaw=True, opt_bias=True, opt_scale=True,
                sigma_mode="iso", iso_var=1e-4, q_psi=1e-4,
                eps_psi=1e-10, sigma_b=1.0, sigma_a=1.0, init="ramp")
    base.update(kw)
    return S.SolverConfig(**base)


class TestGeometry(unittest.TestCase):
    def test_A_matches_assemble_vec(self):
        rng = np.random.RandomState(0)
        for _ in range(100):
            v = rng.normal(size=2)
            psi = rng.uniform(-np.pi, np.pi)
            a = S.A(psi) @ v
            b = _assemble_vec(v, psi)
            self.assertLessEqual(float(np.max(np.abs(a - b))), 1e-14)

    def test_orthonormal(self):
        rng = np.random.RandomState(1)
        psi = rng.uniform(-10, 10, size=50)
        M = np.einsum("...ji,...jk->...ik", S.A(psi), S.A(psi))
        I = np.broadcast_to(np.eye(2), M.shape)
        self.assertLessEqual(float(np.max(np.abs(M - I))), 1e-12)
        M2 = np.einsum("...ij,...j->...i", S.AT(psi),
                       S.A(psi)[..., 0])
        np.testing.assert_allclose(M2, np.stack(
            [np.ones(50), np.zeros(50)], -1), atol=1e-12)

    def test_wrap(self):
        self.assertAlmostEqual(S.wrap(0.1), 0.1)
        self.assertAlmostEqual(S.wrap(np.pi + 0.2), -np.pi + 0.2)
        self.assertAlmostEqual(S.wrap(-np.pi - 0.2), np.pi - 0.2)
        self.assertAlmostEqual(S.wrap(3 * np.pi + 0.1), -np.pi + 0.1)
        for a in np.linspace(-20, 20, 101):
            w = S.wrap(a)
            self.assertGreaterEqual(w, -np.pi - 1e-12)
            self.assertLessEqual(w, np.pi + 1e-12)
            self.assertAlmostEqual(np.sin(w), np.sin(a), places=10)
            self.assertAlmostEqual(np.cos(w), np.cos(a), places=10)

    def test_jacobian_fd(self):
        rng = np.random.RandomState(2)
        K = 8
        d = rng.normal(size=(K, 2)) * 0.3
        dyaw = rng.normal(size=K) * 0.05
        p0 = rng.normal(size=2)
        p1 = rng.normal(size=2)
        psi0, psi1 = 0.2, -0.1
        cfg = _cfg(sigma_mode="physical", q_f=2e-3, q_l=5e-3, eps_l=0.02)
        cov = S.Covariance("physical", q_f=2e-3, q_l=5e-3, eps_l=0.02)
        psi_init = psi0 + np.concatenate([[0.0], np.cumsum(dyaw)])
        npi, nvar, off_psi, off_b, off_a = S._assemble_arrays(
            K, p0, p1, psi0, psi1, d, dyaw, cfg, None)
        x = rng.normal(size=nvar) * 0.05
        args = (d, dyaw, DT, p0, p1, psi0, psi1, psi_init, cfg, cov,
                npi, nvar, off_psi, off_b, off_a)
        r, J, _, _, _, _ = S.residuals_jacobian(x, *args, jac=True)
        eps = 1e-7
        Jfd = np.zeros_like(J)
        for k in range(nvar):
            xp = x.copy(); xp[k] += eps
            xm = x.copy(); xm[k] -= eps
            rp, _, _, _, _, _ = S.residuals_jacobian(xp, *args, jac=False)
            rm, _, _, _, _, _ = S.residuals_jacobian(xm, *args, jac=False)
            Jfd[:, k] = (rp - rm) / (2 * eps)
        err = np.max(np.abs(J - Jfd)) / max(np.max(np.abs(Jfd)), 1e-12)
        self.assertLessEqual(float(err), 1e-6)


class TestV1a(unittest.TestCase):
    def _ramp_reference(self, d, p0, p1, psi):
        K = len(d)
        p = np.empty((K + 1, 2))
        p[0] = p0
        for j in range(K):
            p[j + 1] = p[j] + S.frame_to_world(d[j], psi[j])
        delta = p1 - p[-1]
        tau = np.arange(K + 1)[:, None] / float(K)
        out = p + tau * delta
        out[0] = p0
        out[-1] = p1
        return out

    def test_linear_ramp_equivalence(self):
        rng = np.random.RandomState(3)
        for trial in range(20):
            K = int(rng.randint(2, 60))
            d = rng.normal(size=(K, 2)) * 0.05
            psi0 = rng.uniform(-np.pi, np.pi)
            dyaw = rng.normal(size=K) * 0.05
            psi_init = psi0 + np.concatenate([[0.0], np.cumsum(dyaw)])
            p0 = rng.normal(size=2)
            p1 = rng.normal(size=2) * 0.5
            cfg = _cfg(opt_yaw=False, opt_bias=False, opt_scale=False,
                       sigma_mode="iso", check_condition=False)
            res = S.solve_interval(d, dyaw, p0, p1, psi0, psi_init[-1],
                                   cfg=cfg, psi_init=psi_init)
            ref = self._ramp_reference(d, p0, p1, psi_init[:-1])
            self.assertLessEqual(float(np.max(np.abs(res.p - ref))), 1e-6)

    def test_hard_endpoints_exact(self):
        d, dyaw, p, psi = _make_truth(30)
        cfg = _cfg(init="raw")
        res = S.solve_interval(d, dyaw, p[0], p[-1], psi[0], psi[-1],
                               cfg=cfg)
        self.assertEqual(float(np.max(np.abs(res.p[0] - p[0]))), 0.0)
        self.assertEqual(float(np.max(np.abs(res.p[-1] - p[-1]))), 0.0)
        self.assertLessEqual(res.endpoint_res, 1e-12)
        self.assertLessEqual(res.yaw_boundary_res, 1e-12)


class TestSyntheticScenarios(unittest.TestCase):
    def test_zero_error_straight(self):
        d, dyaw, p, psi = _make_truth(40, turn_rate=0.0)
        res = S.solve_interval(d, dyaw, p[0], p[-1], psi[0], psi[-1],
                               cfg=_cfg())
        self.assertLessEqual(float(np.max(np.abs(res.p - p))), 1e-8)
        self.assertLessEqual(abs(S.wrap(res.psi[-1] - psi[-1])), 1e-10)

    def test_constant_gyro_bias(self):
        d, dyaw, p, psi = _make_truth(50, turn_rate=0.05)
        b_true = 0.03
        dyaw_obs = dyaw + b_true * DT
        res = S.solve_interval(d, dyaw_obs, p[0], p[-1], psi[0], psi[-1],
                               cfg=_cfg(sigma_b=1.0))
        self.assertAlmostEqual(res.b, b_true, delta=1e-5)
        self.assertAlmostEqual(res.a, 0.0, places=8)

    def test_constant_scale(self):
        d, dyaw, p, psi = _make_truth(45)
        s_true = 1.2
        d_obs = d / s_true
        res = S.solve_interval(d_obs, dyaw, p[0], p[-1], psi[0], psi[-1],
                               cfg=_cfg(sigma_a=1.0))
        self.assertAlmostEqual(res.a, np.log(s_true), delta=1e-3)

    def test_bias_and_scale(self):
        d, dyaw, p, psi = _make_truth(60, turn_rate=-0.04)
        b_true, s_true = -0.02, 0.85
        res = S.solve_interval(d / s_true, dyaw + b_true * DT,
                               p[0], p[-1], psi[0], psi[-1],
                               cfg=_cfg(sigma_b=1.0, sigma_a=1.0))
        self.assertAlmostEqual(res.b, b_true, delta=1e-4)
        self.assertAlmostEqual(res.a, np.log(s_true), delta=1e-3)
        self.assertLessEqual(float(np.max(np.abs(res.p - p))), 1e-5)

    def test_static_interval(self):
        K = 30
        d = np.zeros((K, 2))
        dyaw = np.zeros(K)
        p0 = np.array([0.3, 0.1])
        cfg = _cfg(sigma_mode="physical", q_f=1e-3, q_l=1e-3, eps_l=0.01)
        cov = S.Covariance("physical", q_f=1e-3, q_l=1e-3, eps_l=0.01)
        res = S.solve_interval(d, dyaw, p0, p0, 0.4, 0.4, cfg=cfg, cov=cov)
        self.assertLessEqual(float(np.max(np.abs(res.p - p0))), 1e-9)
        self.assertAlmostEqual(res.b, 0.0, places=9)
        self.assertAlmostEqual(res.a, 0.0, places=9)

    def test_side_step(self):
        d, dyaw, p, psi = _make_truth(40, turn_rate=0.0, lat=0.8)
        res = S.solve_interval(d, dyaw, p[0], p[-1], psi[0], psi[-1],
                               cfg=_cfg())
        self.assertLessEqual(float(np.max(np.abs(res.p - p))), 1e-8)
        lat = np.diff(p, axis=0)
        self.assertGreater(float(np.mean(np.abs(lat[:, 0]))), 0.01)

    def test_pi_turn(self):
        d, dyaw, p, psi = _make_truth(
            80, speed=0.8, turn_rate=0.12, psi_start=np.pi - 0.2)
        self.assertLessEqual(float(psi.min()), np.pi - 0.2 + 1e-12)
        self.assertGreater(float(psi.max()), np.pi)
        res = S.solve_interval(d, dyaw, p[0], p[-1], psi[0], psi[-1],
                               cfg=_cfg())
        self.assertLessEqual(float(np.max(np.abs(res.p - p))), 1e-7)

    def test_short_and_near_zero(self):
        for K in (1, 2, 3):
            d = np.full((K, 2), 1e-6)
            dyaw = np.zeros(K)
            res = S.solve_interval(d, dyaw, np.zeros(2), np.full(2, 1e-6),
                                   0.0, 0.0, cfg=_cfg())
            self.assertTrue(res.success)
            self.assertLessEqual(res.endpoint_res, 1e-12)

    def test_bad_init_same_solution(self):
        d, dyaw, p, psi = _make_truth(50, turn_rate=0.03)
        cfg_raw = _cfg(init="raw")
        cfg_ramp = _cfg(init="ramp")
        r1 = S.solve_interval(d, dyaw, p[0], p[-1], psi[0], psi[-1],
                              cfg=cfg_raw)
        r2 = S.solve_interval(d, dyaw, p[0], p[-1], psi[0], psi[-1],
                              cfg=cfg_ramp)
        self.assertLessEqual(float(np.max(np.abs(r1.p - r2.p))), 1e-6)
        self.assertLessEqual(abs(r1.b - r2.b), 1e-5)
        self.assertLessEqual(abs(r1.a - r2.a), 1e-5)

    def test_singular_covariance_rejected(self):
        with self.assertRaises(ValueError):
            S.Covariance("physical", q_f=0.0, q_l=1e-3, eps_l=0.01)
        with self.assertRaises(ValueError):
            S.Covariance("physical", q_f=1e-3, q_l=-1e-3, eps_l=0.01)
        with self.assertRaises(ValueError):
            S.Covariance("physical", q_f=1e-3, q_l=1e-3, eps_l=-0.1)

    def test_physical_covariance_bounds(self):
        cov = S.Covariance("physical", q_f=1e-2, q_l=1e-1, eps_l=0.0)
        d = np.array([[0.0, 0.0], [1e-9, 0.0], [0.5, 0.0], [1e6, 0.0]])
        var = cov.variance(d)
        self.assertTrue(np.all(var > 0))
        self.assertTrue(np.all(var <= S.VAR_MAX))
        self.assertTrue(np.all(var >= S.VAR_MIN))


if __name__ == "__main__":
    unittest.main()


class TestV1Prime(unittest.TestCase):
    """v1′：冻结 d̂，只解 (b,a)，端点用 VT 位置。"""

    def test_endpoint_jacobian_fd(self):
        import bias_scale_solver as B
        rng = np.random.RandomState(11)
        K = 30
        d = rng.normal(size=(K, 2)) * 0.03
        dyaw = rng.normal(size=K) * 0.02
        psi_in = 0.4
        b0, a0 = 0.03, -0.1
        J = B.endpoint_jacobian(d, dyaw, psi_in, b0, a0)
        eps = 1e-7
        Jfd = np.zeros_like(J)
        p0 = np.zeros(2)
        for k, (b, a) in enumerate([(b0 + eps, a0), (b0 - eps, a0),
                                     (b0, a0 + eps), (b0, a0 - eps)]):
            p, _ = B.rollout(d, dyaw, psi_in, p0, b, a)
            col = k // 2
            sign = 1.0 if k % 2 == 0 else -1.0
            Jfd[:, col] += sign * p[-1] / (2 * eps)
        err = np.max(np.abs(J - Jfd)) / max(np.max(np.abs(Jfd)), 1e-12)
        self.assertLessEqual(float(err), 1e-6)

    def _setup(self, K=80, b_true=0.0, s_true=1.0):
        import bias_scale_solver as B
        d, dyaw, p, psi = _make_truth(K, turn_rate=0.02)
        d_obs = d / s_true
        dyaw_obs = dyaw + b_true * DT
        S0 = B.endpoint_cov(d_obs, B.raw_yaw(dyaw_obs, psi[0]),
                            3e-3, 8e-3, 0.01)
        return d_obs, dyaw_obs, p, psi, S0

    def test_bias_only_recovers(self):
        import bias_scale_solver as B
        d, dyaw, p, psi, S0 = self._setup(b_true=0.04)
        cfg = B.BSConfig(solve_b=True, solve_a=False, sigma_b=0.2,
                         sigma_a=0.5)
        r = B.solve(d, dyaw, psi[0], p[0], p[-1], S0, cfg)
        self.assertAlmostEqual(r.b, 0.04, delta=5e-3)
        self.assertEqual(r.a, 0.0)
        self.assertLess(r.endpoint_res_after, 0.05)
        self.assertTrue(np.isfinite(r.post_std_b))
        self.assertTrue(np.isnan(r.post_std_a))

    def test_scale_only_recovers(self):
        import bias_scale_solver as B
        d, dyaw, p, psi, S0 = self._setup(s_true=1.2)
        cfg = B.BSConfig(solve_b=False, solve_a=True, sigma_b=0.2,
                         sigma_a=0.5)
        r = B.solve(d, dyaw, psi[0], p[0], p[-1], S0, cfg)
        self.assertAlmostEqual(r.a, np.log(1.2), delta=2e-3)
        self.assertEqual(r.b, 0.0)
        self.assertLess(r.endpoint_res_after, 0.05)
        self.assertTrue(np.isnan(r.post_std_b))
        self.assertTrue(np.isfinite(r.post_std_a))

    def test_bias_scale_recovers(self):
        import bias_scale_solver as B
        d, dyaw, p, psi, S0 = self._setup(b_true=-0.03, s_true=0.85)
        cfg = B.BSConfig(solve_b=True, solve_a=True, sigma_b=0.2,
                         sigma_a=0.5)
        r = B.solve(d, dyaw, psi[0], p[0], p[-1], S0, cfg)
        self.assertAlmostEqual(r.b, -0.03, delta=5e-3)
        self.assertAlmostEqual(r.a, np.log(0.85), delta=5e-3)
        self.assertLess(r.endpoint_res_after, 0.05)
        self.assertTrue(np.isfinite(r.cond))
        self.assertTrue(np.isfinite(r.post_std_b))
        self.assertTrue(np.isfinite(r.post_std_a))

    def test_weak_endpoint_prior_dominates(self):
        import bias_scale_solver as B
        d, dyaw, p, psi, S0 = self._setup(b_true=0.05, s_true=1.2)
        cfg = B.BSConfig(solve_b=True, solve_a=True, sigma_b=0.01,
                         sigma_a=0.01)
        r = B.solve(d, dyaw, psi[0], p[0], p[-1], S0 * 1e8, cfg)
        self.assertLess(abs(r.b), 1e-3)
        self.assertLess(abs(r.a), 1e-3)
        raw, _ = B.rollout(d, dyaw, psi[0], p[0], 0.0, 0.0)
        self.assertLessEqual(float(np.max(np.abs(r.p_hat - raw))), 1e-5)

    def test_linear_ramp_is_not_raw(self):
        """LinearRamp 是位置 ramp；v1′ b=a=0 是 raw rollout，两者不同。"""
        d, dyaw, p, psi = _make_truth(30)
        p_out = p[-1] + np.array([0.20, -0.10])
        cfg = _cfg(opt_yaw=False, opt_bias=False, opt_scale=False,
                   sigma_mode="iso", check_condition=False)
        psi_init = psi[0] + np.concatenate([[0.0], np.cumsum(dyaw)])
        res = S.solve_interval(d, dyaw, p[0], p_out, psi[0],
                               psi_init[-1], cfg=cfg, psi_init=psi_init)
        self.assertGreater(float(np.max(np.abs(res.p - p))), 1e-2)
        self.assertLessEqual(float(np.max(np.abs(res.p[-1] - p_out))),
                             1e-12)


class TestSplineBias(unittest.TestCase):
    """v1s：B-spline/多项式时变 yaw 零偏 + 常数尺度 + 端点闭合。"""

    def test_basis_and_roughness(self):
        import spline_bias_solver as SP
        tau = np.linspace(-1, 1, 9)
        B = SP.basis(tau, 4)
        np.testing.assert_allclose(B[:, 0], 1.0)
        np.testing.assert_allclose(B[:, 1], tau)
        np.testing.assert_allclose(B[:, 2], tau ** 2)
        R = SP.roughness_matrix(4)
        self.assertAlmostEqual(R[2, 2], 8.0)
        self.assertAlmostEqual(R[3, 3], 24.0)
        self.assertEqual(R[0, 0], 0.0)
        self.assertEqual(R[1, 1], 0.0)

    def test_jacobian_fd(self):
        import spline_bias_solver as SP
        rng = np.random.RandomState(21)
        K = 25
        M = 3
        d = rng.normal(size=(K, 2)) * 0.03
        dyaw = rng.normal(size=K) * 0.02
        tau = SP.tau_grid(K)
        c = rng.normal(size=M) * 0.02
        a = -0.05
        J = SP.endpoint_jacobian(d, dyaw, 0.3, tau, c, a)
        eps = 1e-7
        Jfd = np.zeros_like(J)
        p0 = np.zeros(2)
        for k in range(M):
            cp = c.copy(); cp[k] += eps
            cm = c.copy(); cm[k] -= eps
            pp, _ = SP.rollout(d, dyaw, 0.3, p0, tau, cp, a)
            pm, _ = SP.rollout(d, dyaw, 0.3, p0, tau, cm, a)
            Jfd[:, k] = (pp[-1] - pm[-1]) / (2 * eps)
        pp, _ = SP.rollout(d, dyaw, 0.3, p0, tau, c, a + eps)
        pm, _ = SP.rollout(d, dyaw, 0.3, p0, tau, c, a - eps)
        Jfd[:, M] = (pp[-1] - pm[-1]) / (2 * eps)
        err = np.max(np.abs(J - Jfd)) / max(np.max(np.abs(Jfd)), 1e-12)
        self.assertLessEqual(float(err), 1e-6)

    def _setup(self, K=100, c_true=None, s_true=1.0):
        import spline_bias_solver as SP
        if c_true is None:
            c_true = np.array([0.02, -0.01, 0.03])
        d, dyaw, p, psi = _make_truth(K, turn_rate=0.02)
        tau = SP.tau_grid(K)
        b = SP.basis(tau, len(c_true)) @ c_true
        dyaw_obs = dyaw + b * DT
        d_obs = d / s_true
        return d_obs, dyaw_obs, p, psi, tau, c_true

    def test_recover_time_varying_bias_and_scale(self):
        import spline_bias_solver as SP
        d, dyaw, p, psi, tau, c_true = self._setup(s_true=1.2)
        cfg = SP.SplineConfig(M=3, lam=1e-2, sigma_a=1.0, sigma_c=0.2)
        r = SP.solve(d, dyaw, psi[0], p[0], p[-1],
                     np.eye(2) * 1e-4, tau, cfg)
        self.assertTrue(r.success)
        self.assertLessEqual(float(np.max(np.abs(r.c - c_true))), 3e-2)
        self.assertAlmostEqual(r.a, np.log(1.2), delta=5e-3)
        self.assertLessEqual(float(np.max(np.abs(r.p_hat - p))), 1e-2)

    def test_without_magnitude_prior_rank_deficient(self):
        """M=3 只有粗糙度罚时 constant/linear 方向无信息，H 秩亏。"""
        import spline_bias_solver as SP
        d, dyaw, p, psi, tau, _ = self._setup()
        cfg = SP.SplineConfig(M=3, lam=0.0, sigma_a=1.0, sigma_c=0.0)
        r = SP.solve(d, dyaw, psi[0], p[0], p[-1],
                     np.eye(2) * 1e-4, tau, cfg)
        self.assertFalse(r.success)

    def test_constant_bias_M1(self):
        import spline_bias_solver as SP
        d, dyaw, p, psi, tau, _ = self._setup(c_true=np.array([0.04]), s_true=1.0)
        cfg = SP.SplineConfig(M=1, lam=0.0, sigma_a=1.0)
        r = SP.solve(d, dyaw, psi[0], p[0], p[-1],
                     np.eye(2) * 1e-4, tau, cfg)
        self.assertTrue(r.success)
        self.assertAlmostEqual(r.c[0], 0.04, delta=5e-3)
        self.assertLessEqual(float(np.max(np.abs(r.p_hat - p))), 1e-3)

    def test_roughness_suppresses_curvature(self):
        import spline_bias_solver as SP
        d, dyaw, p, psi, tau, _ = self._setup()
        cfg0 = SP.SplineConfig(M=3, lam=0.0, sigma_a=1.0, sigma_c=0.2)
        cfg1 = SP.SplineConfig(M=3, lam=1e3, sigma_a=1.0, sigma_c=0.2)
        r0 = SP.solve(d, dyaw, psi[0], p[0], p[-1],
                      np.eye(2) * 1e-4, tau, cfg0)
        r1 = SP.solve(d, dyaw, psi[0], p[0], p[-1],
                      np.eye(2) * 1e-4, tau, cfg1)
        self.assertLess(abs(r1.c[2]), abs(r0.c[2]) + 1e-12)
        self.assertTrue(r1.success)
