"""V7 C 变体族（2026-10-26 统一编号 V7_Cv*）：结构隔离与枚举训练。

历史：本文件原名所指的 V1/V2/V3/V1_enum 编号已随统一编号改为
V7_Cv1（单门基准）/ V7_Cv2（双门）/ V7_Cv3（T-only 适配器）/
V7_Cv4（枚举训练），2026-10-26 消融组新增 V7_Cv6（双门+始终 VT）/
V7_Cv7（无门控+枚举）。2026-10-30 门控修改方案新增 V7_Cv5T（逐帧双门）/
V7_Cv5LR（窗口四组门）/ V7_Cv5TLR（逐帧四组门）。
"""
import copy
import io
import unittest
import warnings

import torch

from anysole.eval import _load_model
from anysole.registry import get_builder
from anysole.train import backward_v7_enum, condition_inputs, parse_args
from anysole.types import CONFIG_T, CONFIG_V, JOINT_PROTOCOL_CHECKSUM, MOTION_PROTOCOL
from anysole.utils.losses import compute_losses
from configs.tools.common import load_registry
from configs.tools.runner import _evaluation_splits, _train_command


def config(name):
    entry = get_builder(name)
    return {"model_name": name, "modal": "anysolev2", "d_model": 16, "tw": 4,
            "pose_layers": 1, "dropout": 0., "config_probs": [.5, .25, .25],
            "lambda_pose": 3., "lambda_traj": 1., "lambda_contact": 1.,
            "lambda_air": 2., "traj_deltas": [1],
            "motion_n_joints": 24, "motion_protocol": MOTION_PROTOCOL,
            "joint_protocol_checksum": JOINT_PROTOCOL_CHECKSUM,
            **entry.STRUCTURE, **entry.CONFIG_EXTRA}


def build(name):
    torch.manual_seed(17)
    return get_builder(name).build(config(name)).eval()


def batch():
    torch.manual_seed(19)
    return {"V_feat": torch.randn(2, 4, 2051), "T_raw": torch.rand(2, 4, 96),
            "T_phys": torch.rand(2, 4, 12), "pose_gt": torch.randn(2, 4, 144),
            "vel_gt": torch.randn(2, 4, 3), "trans_gt": torch.zeros(2, 4, 3),
            "contact_gt": torch.tensor([[[1., 0.]] * 4, [[0., 1.]] * 4])}


def forward(model, data, mode):
    cid = torch.full((2,), mode, dtype=torch.long)
    return model(data["V_feat"], data["T_raw"], data["T_phys"], cid)


class NewModelTests(unittest.TestCase):
    def test_cv6_cv7_reuse_v5_and_a_skeletons(self):
        # Cv6 = 双门结构（同 Cv5/Cv2）；Cv7 = 无门控 A 骨架（同 Av1）。
        for variant, base in (("V7_Cv6", "V7_Cv2"), ("V7_Cv7", "V7_Av1")):
            a, b = build(variant), build(base)
            self.assertEqual(set(a.state_dict()), set(b.state_dict()))
            data = batch()
            with torch.no_grad():
                for mode in range(3):
                    x, y = forward(a, data, mode), forward(b, data, mode)
                    for key in ("x0_hat", "v_hat", "trans_hat"):
                        torch.testing.assert_close(x[key], y[key], rtol=0, atol=0)

    def test_v2_starts_as_v1_and_has_independent_task_gates(self):
        a, b = build("V7_Cv1"), build("V7_Cv2")
        data = batch()
        with torch.no_grad():
            x, y = forward(a, data, 0), forward(b, data, 0)
            for key in ("x0_hat", "v_hat"):
                torch.testing.assert_close(x[key], y[key], rtol=0, atol=0)
            b.route_root_mlp[-1].bias.add_(2.)
            z = forward(b, data, 0)
        gate = z["route_gate"]
        self.assertEqual(tuple(gate.shape), (2, 4, 10, 1))
        self.assertGreater((gate[:, :, 0] - gate[:, :, 1]).abs().max().item(), .01)
        self.assertGreater((z["v_hat"] - y["v_hat"]).abs().max().item(), 0.)
        self.assertFalse(any(k.startswith("t_only_") for k in b.state_dict()))

    def test_v3_adapters_start_at_identity_and_only_affect_t_only(self):
        a, b = build("V7_Cv1"), build("V7_Cv3")
        data = batch()
        with torch.no_grad():
            for mode in range(3):
                x, y = forward(a, data, mode), forward(b, data, mode)
                torch.testing.assert_close(x["x0_hat"], y["x0_hat"], rtol=0, atol=0)
                torch.testing.assert_close(x["v_hat"], y["v_hat"], rtol=0, atol=0)
            b.t_only_root_adapter[-1].bias.fill_(.2)
            b.t_only_pose_adapter[-1].bias.fill_(.2)
            for mode in (0, CONFIG_V):
                x, y = forward(a, data, mode), forward(b, data, mode)
                torch.testing.assert_close(x["x0_hat"], y["x0_hat"], rtol=0, atol=0)
                torch.testing.assert_close(x["v_hat"], y["v_hat"], rtol=0, atol=0)
            x, y = forward(a, data, CONFIG_T), forward(b, data, CONFIG_T)
        self.assertGreater((x["x0_hat"] - y["x0_hat"]).abs().max().item(), 0.)
        self.assertGreater((x["v_hat"] - y["v_hat"]).abs().max().item(), 0.)
        self.assertFalse(hasattr(b, "route_root_mlp"))

    def test_adapter_gradients_are_private_to_t_only(self):
        model = build("V7_Cv3")
        data = batch()
        for mode in range(3):
            model.zero_grad(set_to_none=True)
            out = forward(model, data, mode)
            (out["x0_hat"].square().mean() + out["v_hat"].square().mean()).backward()
            for adapter in (model.t_only_root_adapter, model.t_only_pose_adapter):
                norm = sum(p.grad.abs().sum().item() for p in adapter.parameters() if p.grad is not None)
                if mode == CONFIG_T:
                    self.assertGreater(norm, 0.)
                else:
                    self.assertEqual(norm, 0.)

    def test_missing_inputs_do_not_leak_and_all_gradients_are_finite(self):
        data = batch()
        for name in ("V7_Cv1", "V7_Cv2", "V7_Cv3", "V7_Cv6", "V7_Cv7",
                     "V7_Cv5T", "V7_Cv5LR", "V7_Cv5TLR", "V7_Cv5Q"):
            model = build(name)
            for mode, keys in ((CONFIG_V, ("T_raw", "T_phys")), (CONFIG_T, ("V_feat",))):
                changed = dict(data)
                changed.update({key: data[key] * 10 for key in keys})
                with torch.no_grad():
                    a, b = forward(model, data, mode), forward(model, changed, mode)
                torch.testing.assert_close(a["x0_hat"], b["x0_hat"], rtol=0, atol=0)
                torch.testing.assert_close(a["v_hat"], b["v_hat"], rtol=0, atol=0)
            model.zero_grad(set_to_none=True)
            losses = compute_losses(forward(model, data, 0), data, torch.zeros(2, dtype=torch.long), config(name))
            losses["loss"].backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_checkpoint_reload_preserves_each_new_structure(self):
        data = batch()
        for name in ("V7_Cv1", "V7_Cv2", "V7_Cv3", "V7_Cv6", "V7_Cv7",
                     "V7_Cv5T", "V7_Cv5LR", "V7_Cv5TLR", "V7_Cv5Q"):
            model = build(name)
            stream = io.BytesIO()
            torch.save({"model": model.state_dict(), "config": config(name), "epoch": 1}, stream)
            stream.seek(0)
            checkpoint = torch.load(stream, weights_only=False)
            restored = _load_model(checkpoint, config(name), torch.device("cpu"))
            with torch.no_grad():
                for mode in range(3):
                    a, b = forward(model, data, mode), forward(restored, data, mode)
                    torch.testing.assert_close(a["x0_hat"], b["x0_hat"], rtol=0, atol=0)
                    torch.testing.assert_close(a["v_hat"], b["v_hat"], rtol=0, atol=0)

    def test_t_reuses_cv5_gate_mlps_per_frame(self):
        # T = Cv5 双门逐帧化：参数集合与初始值逐项相同（同种子），
        # 逐帧门公式 = sigmoid(MLP(cat(v_tok_t, t_tok_t)))，Cv5 门窗内恒定。
        a, b = build("V7_Cv5"), build("V7_Cv5T")
        self.assertEqual(set(a.state_dict()), set(b.state_dict()))
        for key in a.state_dict():
            torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], rtol=0, atol=0)
        data = batch()
        with torch.no_grad():
            x, y = forward(a, data, 0), forward(b, data, 0)
            t_tac = torch.cat([data["T_raw"], data["T_phys"]], dim=-1)
            v_tok, t_tok = b.encoders(data["V_feat"], t_tac,
                                      torch.zeros(2, dtype=torch.long))
            x_t = torch.cat([v_tok, t_tok], dim=-1)             # (B, tw, 2d)
            g_vel = torch.sigmoid(b.route_root_mlp(x_t))        # (B, tw, 1)
            g_pose = torch.sigmoid(b.route_mlp(x_t))
            expected = torch.cat([g_vel, g_pose.expand(-1, -1, 9)], dim=-1)[..., None]
            torch.testing.assert_close(y["route_gate"], expected, rtol=0, atol=0)
            g_win = torch.sigmoid(a.route_mlp(x_t.mean(dim=1)))  # (B, 1)
            self.assertEqual(
                (x["route_gate"][:, 1:, 1:, 0] - g_win[:, None, :]).abs().max().item(), 0.)
        self.assertEqual(tuple(y["route_gate"].shape), (2, 4, 10, 1))
        # 随机输入：T 门逐帧变化，Cv5 门窗内恒定。
        self.assertGreater(
            (y["route_gate"][:, 1:, 0, 0] - y["route_gate"][:, :1, 0, 0]).abs().max().item(), 0.)
        self.assertEqual(
            (x["route_gate"][:, 1:, 0, 0] - x["route_gate"][:, :1, 0, 0]).abs().max().item(), 0.)
        # 根速度门与姿态门独立（root MLP 为姿态门深拷贝，扰动后分离）。
        with torch.no_grad():
            b.route_root_mlp[-1].bias.add_(2.)
        y = forward(b, data, 0)
        self.assertGreater(
            (y["route_gate"][:, 0, 0, 0] - y["route_gate"][:, 0, 1, 0]).abs().max().item(), 0.)
        # 姿态门在同一帧内对九姿态 query 同值。
        self.assertEqual(
            (y["route_gate"][:, 0, 1:, 0] - y["route_gate"][:, 0, 1:2, 0]).abs().max().item(), 0.)

    def test_lr_reproduces_b_and_tlr_reproduces_t_at_init(self):
        # 左右门 = 姿态门深拷贝 → 初始化等价：LR 复现 B、TLR 复现 T（同种子，
        # 三条件）；共享模块随机初始化不受新增参数影响。
        b, lr = build("V7_Cv5"), build("V7_Cv5LR")
        t, tlr = build("V7_Cv5T"), build("V7_Cv5TLR")
        data = batch()
        with torch.no_grad():
            for mode in range(3):
                x, y = forward(b, data, mode), forward(lr, data, mode)
                torch.testing.assert_close(x["x0_hat"], y["x0_hat"], rtol=0, atol=0)
                torch.testing.assert_close(x["v_hat"], y["v_hat"], rtol=0, atol=0)
                x, y = forward(t, data, mode), forward(tlr, data, mode)
                torch.testing.assert_close(x["x0_hat"], y["x0_hat"], rtol=0, atol=0)
                torch.testing.assert_close(x["v_hat"], y["v_hat"], rtol=0, atol=0)
        # 共享键与基座逐参数一致；route_left/route_right 为新增键。
        shared = {k: v for k, v in lr.state_dict().items()
                  if not k.startswith(("route_left", "route_right"))}
        self.assertEqual(set(shared), set(b.state_dict()))
        for key in shared:
            torch.testing.assert_close(shared[key], b.state_dict()[key], rtol=0, atol=0)
        for key in ("route_left_mlp.0.weight", "route_left_mlp.0.bias",
                    "route_left_mlp.2.weight", "route_left_mlp.2.bias"):
            torch.testing.assert_close(lr.state_dict()[key], lr.state_dict()["route_mlp." + key.split(".", 1)[1]], rtol=0, atol=0)
        self.assertFalse(any(k.startswith(("route_left", "route_right")) for k in b.state_dict()))

    def test_part_group_mapping_follows_part_names(self):
        # 分组严格按 PART_NAMES：query 0 = vel；1-5（root/torso/headneck/
        # l_arm/r_arm）= other；6/8（l_leg/l_foot）= left；7/9 = right。
        lr, tlr = build("V7_Cv5LR"), build("V7_Cv5TLR")
        data = batch()
        for model, per_frame in ((lr, False), (tlr, True)):
            with torch.no_grad():
                gate = forward(model, data, 0)["route_gate"]     # (B, tw, 10, 1)
            self.assertEqual(tuple(gate.shape), (2, 4, 10, 1))
            if per_frame:
                self.assertGreater(
                    (gate[:, 1:, 0, 0] - gate[:, :1, 0, 0]).abs().max().item(), 0.)
            else:
                self.assertEqual(
                    (gate[:, 1:, :, :] - gate[:, :1, :, :]).abs().max().item(), 0.)
            # 组内广播：other 组五 query 同值；left/right 组两 query 同值。
            self.assertEqual(
                (gate[:, :, 1:6, 0] - gate[:, :, 1:2, 0]).abs().max().item(), 0.)
            self.assertEqual(
                (gate[:, :, 6:7, 0] - gate[:, :, 8:9, 0]).abs().max().item(), 0.)
            self.assertEqual(
                (gate[:, :, 7:8, 0] - gate[:, :, 9:10, 0]).abs().max().item(), 0.)
        # 左右门独立学习：扰动左门只改 left 组。
        with torch.no_grad():
            before = forward(tlr, data, 0)
            tlr.route_left_mlp[-1].bias.add_(2.)
            after = forward(tlr, data, 0)
        gate = after["route_gate"]
        self.assertGreater(
            (gate[:, :, 6:7, 0] - gate[:, :, 7:8, 0]).abs().max().item(), 0.)
        # other/right 组不受左门扰动影响（与扰动前同值 = 姿态门深拷贝值）。
        self.assertEqual(
            (gate[:, :, 1:6, 0] - gate[:, :, 7:8, 0]).abs().max().item(), 0.)
        torch.testing.assert_close(gate[:, :, 1:6, 0], before["route_gate"][:, :, 1:6, 0])
        # 扰动左门后模型输出变化（门实际参与调制）。
        self.assertGreater((after["x0_hat"] - before["x0_hat"]).abs().max().item(), 0.)

    def test_q_conditional_gate_reads_first_layer_context(self):
        # Q = TLR 结构（同参数集合、同初始化），门输入改为第一层解码上下文：
        # 与 TLR 的 token 门不同值、形状/组广播不变、梯度经上下文回传。
        tlr, q = build("V7_Cv5TLR"), build("V7_Cv5Q")
        self.assertEqual(set(tlr.state_dict()), set(q.state_dict()))
        for key in tlr.state_dict():
            torch.testing.assert_close(tlr.state_dict()[key], q.state_dict()[key], rtol=0, atol=0)
        data = batch()
        with torch.no_grad():
            x, y = forward(tlr, data, 0), forward(q, data, 0)
        self.assertEqual(tuple(y["route_gate"].shape), (2, 4, 10, 1))
        # 门值不同（条件输入 ≠ token 输入）但组广播结构一致。
        self.assertGreater((x["route_gate"] - y["route_gate"]).abs().max().item(), 0.)
        self.assertGreater(
            (y["route_gate"][:, 1:, 0, 0] - y["route_gate"][:, :1, 0, 0]).abs().max().item(), 0.)
        self.assertEqual(
            (y["route_gate"][:, :, 1:6, 0] - y["route_gate"][:, :, 1:2, 0]).abs().max().item(), 0.)
        self.assertEqual(
            (y["route_gate"][:, :, 6:7, 0] - y["route_gate"][:, :, 8:9, 0]).abs().max().item(), 0.)
        # 梯度经第一层上下文回传到门 MLP（有效条件下有限非零）。
        q.zero_grad(set_to_none=True)
        out = forward(q, data, 0)
        (out["x0_hat"].square().mean() + out["v_hat"].square().mean()).backward()
        for pname in ("route_mlp", "route_root_mlp", "route_left_mlp", "route_right_mlp"):
            for p in getattr(q, pname).parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(p.grad.abs().sum().item(), 0.)
        # gate_override 干预钩子：替换解码器门改变输出（§7 诊断用）。
        with torch.no_grad():
            ref = forward(q, data, 0)
            gate0 = ref["route_gate"].clone()
            gate0[..., 1:6, :] = 0.0   # 关掉 other 组（query 1-5）
            z = q(data["V_feat"], data["T_raw"], data["T_phys"],
                  torch.zeros(2, dtype=torch.long), gate_override=gate0)
        self.assertGreater((z["x0_hat"] - ref["x0_hat"]).abs().max().item(), 0.)


class EnumerationTests(unittest.TestCase):
    def test_streamed_enum_matches_weighted_objective_and_gradients(self):
        for name in ("V7_Cv4", "V7_Cv7", "V7_Cv5TLR", "V7_Cv5Q"):
            model = build(name)
            reference = copy.deepcopy(model)
            data = batch()
            cfg = config(name)
            original = {k: v.clone() for k, v in data.items()}
            _, actual = backward_v7_enum(model, data, cfg)
            objective = 0.
            t_probability = .75
            for mode, probability in enumerate(cfg["config_probs"]):
                cid = torch.full((2,), mode, dtype=torch.long)
                v, t, p, s = condition_inputs(data, cid)
                out = reference(v, t, p, cid, T_s2m=s)
                loss = compute_losses(out, data, cid, cfg)
                objective = objective + probability * (3 * loss["L_pose"] + loss["L_traj"])
                objective = objective + probability / t_probability * loss["L_contact"]
            objective.backward()
            torch.testing.assert_close(actual["loss"], objective.detach())
            for (n, p), (other, q) in zip(model.named_parameters(), reference.named_parameters()):
                self.assertEqual(n, other)
                if p.grad is None or q.grad is None:
                    self.assertIsNone(p.grad)
                    self.assertIsNone(q.grad)
                else:
                    torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-6)
            for key in original:
                torch.testing.assert_close(data[key], original[key], rtol=0, atol=0)
            self.assertFalse(actual["loss"].requires_grad)

    def test_invalid_enum_probabilities_are_rejected(self):
        model, data = build("V7_Cv1"), batch()
        for probs in ([0, 0, 0], [.5, -.1, .6], [float("nan"), 0, 1], [1, 0]):
            with self.assertRaises(ValueError):
                backward_v7_enum(model, data, {**config("V7_Cv1"), "config_probs": probs})

    def test_new_runs_have_matching_recipes_and_no_automatic_test(self):
        registry = load_registry()
        # 当前线（2026-10-29）：训练 cell 即评估 cell（val+test 一次跑齐）。
        # 门控组（2026-10-30，V7_C门控修改方案 §8.2）：T/LR/TLR 配方与
        # Cv5_s1 实际运行记录逐项一致（contact v1 / v1.yaml / 三条件枚举 /
        # ckpt_last / 740ep；条目定义见 z_gen/model_registry 同名块）。
        enum = {"V7_Cv4_s1", "V7_Cv5_s1", "V7_Cv7_s1",
                "V7_Cv5T_s1", "V7_Cv5LR_s1", "V7_Cv5TLR_s1", "V7_Cv5Q_s1"}
        # Cv6 = 双门 + 始终 VT（无模态缺失训练，消融组 2026-10-26）。
        probs = {"V7_Cv6_s1": [1.0, 0.0, 0.0]}
        for name in ("V7_Cv1_s1", "V7_Cv2_s1", "V7_Cv3_s1", "V7_Cv4_s1",
                     "V7_Cv5_s1", "V7_Cv6_s1", "V7_Cv7_s1",
                     "V7_Cv5T_s1", "V7_Cv5LR_s1", "V7_Cv5TLR_s1",
                     "V7_Cv5Q_s1"):
            spec = registry["models"][name]
            args = parse_args(_train_command(registry, name)[3:])
            self.assertEqual(args.config_probs, probs.get(name, [.5, .25, .25]))
            self.assertFalse(args.occ_mix)
            self.assertEqual(args.final_eval_split, "none")
            # Cv4/Cv5/Cv7 与门控组显式枚举；Cv2/Cv3/Cv6 显式 sample；
            # Cv1 缺省（sample）。
            self.assertEqual(args.condition_sampling,
                             "enum" if name in enum else (None if name == "V7_Cv1_s1" else "sample"))
            self.assertEqual(_evaluation_splits(registry, name),
                             ("val", "test"))
            self.assertEqual(spec["epochs"], 740)
            self.assertTrue(args.checkpoint_all)
        # 门控组结构入口与 Cv5 基线一致 + 各自的开关键（结构隔离）。
        base_struct = dict(get_builder("V7_Cv5").STRUCTURE)
        for variant, extra in (("V7_Cv5T", {"v7_route_temporal": True}),
                               ("V7_Cv5LR", {"v7_route_parts": True}),
                               ("V7_Cv5TLR", {"v7_route_temporal": True,
                                              "v7_route_parts": True}),
                               ("V7_Cv5Q", {"v7_route_temporal": True,
                                            "v7_route_parts": True,
                                            "v7_route_conditional": True})):
            struct = dict(get_builder(variant).STRUCTURE)
            self.assertEqual({**base_struct, **extra}, struct)


torch.set_num_threads(1)
warnings.filterwarnings("ignore", message="enable_nested_tensor")

if __name__ == "__main__":
    unittest.main()
