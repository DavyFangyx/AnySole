import unittest

import numpy as np
import torch

from results_display.script.visual_warmup_compare import rollout, score


class VisualWarmupTests(unittest.TestCase):
    def test_scoring_reconstructs_relative_gt_and_scores_only_T_suffix(self):
        from anysole.utils.geometry import fk_pose6d_np, rotmat_to_6d_np, yaw_rotmat_np
        n = 12
        psi = np.linspace(0.2, 0.5, n)
        identity = np.tile([1., 0., 0., 0., 1., 0.], (n, 24, 1))
        world = identity.copy()
        world[:, 0] = rotmat_to_6d_np(yaw_rotmat_np(psi))
        world = world.reshape(n, 144)
        offsets = np.stack([np.cos(np.arange(24)), np.sin(np.arange(24)),
                            np.arange(24) / 20], axis=-1) / 10
        parents = np.array([-1] + [0] * 23)
        positions = np.zeros((n, 3))
        gt = fk_pose6d_np(world, positions, offsets, parents)
        generated = {"pose": np.repeat(world[None], 3, axis=0),
                     "trans": np.repeat(positions[None], 3, axis=0),
                     "frame_ids": np.arange(n), "segments": np.zeros(n),
                     "score_mask": np.arange(n) >= 4}
        session = {"pose_gt": identity.reshape(n, 144), "psi": psi,
                   "kp_gt": gt, "offsets": offsets, "parents": parents}
        result = score(session, generated, relative=True)
        self.assertEqual(result["frame_ids"], list(range(4, n)))
        self.assertEqual(result["frames"], n - 4)
        for row in result["arms"].values():
            for key in ("pa_mpjpe_mm", "mpjpe_mm", "root_orientation_deg",
                        "root_orientation_drift_deg"):
                self.assertLess(row[key], 1e-4)

    def test_switch_keeps_predicted_state_and_ignores_gt_history(self):
        class Dataset:
            def __getitem__(self, index):
                return {"session_id": "S", "frame_start": torch.tensor(index * 2),
                        "V_feat": torch.ones(2, 2051), "T_raw": torch.ones(2, 96),
                        "T_phys": torch.ones(2, 12), "T_s2m": torch.zeros(2, 50),
                        "prev_state": torch.full((2, 148), 999.)}

        class Model(torch.nn.Module):
            ar2 = ar2_fb = True
            tw = 2

            def forward(self, v, tr, tp, cid, subjects, *, T_s2m, prev_state, causal):
                if not causal:
                    raise AssertionError("Expected causal inference")
                dyaw = prev_state[..., :1] + (cid != 2)[:, None, None] * 0.1
                identity = v.new_tensor([1., 0., 0., 0., 1., 0.]).repeat(24)
                return {"x0_hat": identity.expand(3, 2, 144),
                        "v_hat": v.new_zeros(3, 2, 3), "dyaw_hat": dyaw,
                        "vis_yaw": v.new_zeros(3, 2, 1), "vis_gain": v.new_zeros(3, 2, 1)}

        result = rollout(Dataset(), Model().eval(), [0, 1, 2], torch.device("cpu"),
                         prefix_windows=1)
        np.testing.assert_array_equal(result["frame_ids"], np.arange(6))
        np.testing.assert_array_equal(result["score_mask"], [False, False, True, True, True, True])
        # Cold T remains at yaw=0; V/VT prefixes produce nonzero state that
        # persists after switching to T instead of restarting from zero/GT.
        self.assertLess(float(np.abs(result["pose"][0, :, :6] -
                                    np.array([1., 0., 0., 0., 1., 0.])).max()), 1e-6)
        self.assertGreater(float(np.abs(result["pose"][1, 2, :6] -
                                       result["pose"][0, 2, :6]).max()), 0.1)
        np.testing.assert_allclose(result["pose"][1], result["pose"][2])

    def test_real_gap_starts_a_new_prefix(self):
        class Dataset:
            def __getitem__(self, index):
                return {"session_id": "S", "frame_start": torch.tensor([0, 2, 8, 10][index]),
                        "V_feat": torch.ones(2, 2051), "T_raw": torch.ones(2, 96),
                        "T_phys": torch.ones(2, 12), "T_s2m": torch.zeros(2, 50)}

        class Model(torch.nn.Module):
            ar = True
            tw = 2

            def forward(self, v, tr, tp, cid, subjects, *, T_s2m, prev_state, causal):
                identity = v.new_tensor([1., 0., 0., 0., 1., 0.]).repeat(24)
                return {"x0_hat": identity.expand(3, 2, 144),
                        "v_hat": v.new_zeros(3, 2, 3)}

        result = rollout(Dataset(), Model().eval(), [0, 1, 2, 3], torch.device("cpu"), 1)
        np.testing.assert_array_equal(result["score_mask"], [False, False, True, True] * 2)
        np.testing.assert_array_equal(result["segments"], [0] * 4 + [1] * 4)
