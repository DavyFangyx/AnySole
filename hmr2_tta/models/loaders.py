"""plain-torch 模型直载：官方 Lightning ckpt 的 state_dict → HMR2 实例（绕开 PL 加载链）。

用法：
  model, cfg = load_hmr2_plain(checkpoint=None, finetune_aug=True)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hmr2_tta.paths import CACHE_4DHUMANS, SMPL_MODEL_PATH  # noqa: E402

DEFAULT_CKPT = (CACHE_4DHUMANS /
                "logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt")
DEFAULT_CFG = CACHE_4DHUMANS / "logs/train/multiruns/hmr2/0/model_config.yaml"


def load_cfg(finetune_aug: bool = True):
    """官方 model_config.yaml → yacs cfg（路径改为本项目绝对路径；增强按固定机位口径）。"""
    from hmr2.configs import get_config
    cfg = get_config(str(DEFAULT_CFG)).clone()  # clone 解除 freeze（frozen 源的 clone 仍冻结）
    cfg.defrost()
    cfg.SMPL.MODEL_PATH = str(SMPL_MODEL_PATH)
    cfg.SMPL.JOINT_REGRESSOR_EXTRA = str(CACHE_4DHUMANS / "data/SMPL_to_J19.pkl")
    cfg.SMPL.MEAN_PARAMS = str(CACHE_4DHUMANS / "data/smpl_mean_params.npz")
    cfg.LOSS_WEIGHTS.ADVERSARIAL = 0
    # 官方推理路径（models/__init__.py:75-78）在加载时注入 [192,256]（"to crop bbox correctly"）；
    # 训练/评估统一按此口径（与官方 demo/eval 一致）。
    cfg.MODEL.BBOX_SHAPE = [192, 256]
    if finetune_aug:
        # 固定机位：旋转减半、关极端裁剪（决策可回退，notes 记录）
        cfg.DATASETS.CONFIG.ROT_FACTOR = 15
        cfg.DATASETS.CONFIG.EXTREME_CROP_AUG_RATE = 0.0
    cfg.freeze()
    return cfg


def load_hmr2_plain(checkpoint: str | None = None, finetune_aug: bool = True):
    """官方 ckpt state_dict 直载。返回 (model, cfg)。

    strict=False + 打印 unexpected/missing 键清单：预期 unexpected 仅 discriminator 相关。
    """
    from hmr2.models.hmr2 import HMR2
    cfg = load_cfg(finetune_aug=finetune_aug)
    model = HMR2(cfg, init_renderer=False)
    ckpt_path = Path(checkpoint) if checkpoint else DEFAULT_CKPT
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"官方 ckpt 未下载: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state.get("state_dict", state)
    if all(k.startswith("model.") for k in sd):
        sd = {k[len("model."):]: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "discriminator" not in k]
    unexpected = [k for k in unexpected if "discriminator" not in k]
    if missing or unexpected:
        print(f"[load_hmr2_plain] missing={missing}")
        print(f"[load_hmr2_plain] unexpected={unexpected}")
    return model, cfg


def load_hmr2_plain_with_token(checkpoint: str | None = None, finetune_aug: bool = True):
    """官方 ckpt 直载 + token_out 钩子（不改 4D-Humans 官方代码）。

    TTA 的优化变量 = smpl_head 的 token_out（learned SMPL token，论文 token_out）。
    官方 forward 不返回它 → 在本侧用子类覆盖 forward 透出（与官方逐行一致，
    只多一个返回值）。返回 (model, cfg)，model.forward_step 输出带 "token_out"。
    """
    from hmr2.models.hmr2 import HMR2
    from hmr2.models.heads.smpl_head import SMPLTransformerDecoderHead as SMPLHead

    class SMPLHeadWithToken(SMPLHead):
        """SMPLHead 子类：forward 额外返回 token_out（decpose 前瓶颈）。"""

        def forward(self, x, **kwargs):
            import einops
            from hmr2.utils.geometry import rot6d_to_rotmat, aa_to_rotmat
            batch_size = x.shape[0]
            x = einops.rearrange(x, "b c h w -> b (h w) c")
            init_body_pose = self.init_body_pose.expand(batch_size, -1)
            init_betas = self.init_betas.expand(batch_size, -1)
            init_cam = self.init_cam.expand(batch_size, -1)
            if self.joint_rep_type == "aa":
                raise NotImplementedError
            pred_body_pose = init_body_pose
            pred_betas = init_betas
            pred_cam = init_cam
            pred_body_pose_list, pred_betas_list, pred_cam_list = [], [], []
            last_token = None
            for _ in range(self.cfg.MODEL.SMPL_HEAD.get("IEF_ITERS", 1)):
                if self.input_is_mean_shape:
                    token = torch.cat([pred_body_pose, pred_betas, pred_cam],
                                      dim=1)[:, None, :]
                else:
                    token = torch.zeros(batch_size, 1, 1).to(x.device)
                token_out = self.transformer(token, context=x)
                token_out = token_out.squeeze(1)  # (B, C)
                pred_body_pose = self.decpose(token_out) + pred_body_pose
                pred_betas = self.decshape(token_out) + pred_betas
                pred_cam = self.deccam(token_out) + pred_cam
                pred_body_pose_list.append(pred_body_pose)
                pred_betas_list.append(pred_betas)
                pred_cam_list.append(pred_cam)
                last_token = token_out
            joint_conversion_fn = {
                "6d": rot6d_to_rotmat,
                "aa": lambda x: aa_to_rotmat(x.view(-1, 3).contiguous()),
            }[self.joint_rep_type]
            pred_smpl_params_list = {}
            pred_smpl_params_list["body_pose"] = torch.cat(
                [joint_conversion_fn(pbp).view(batch_size, -1, 3, 3)[:, 1:, :, :]
                 for pbp in pred_body_pose_list], dim=0)
            pred_smpl_params_list["betas"] = torch.cat(pred_betas_list, dim=0)
            pred_smpl_params_list["cam"] = torch.cat(pred_cam_list, dim=0)
            pred_body_pose = joint_conversion_fn(pred_body_pose).view(
                batch_size, self.cfg.SMPL.NUM_BODY_JOINTS + 1, 3, 3)
            pred_smpl_params = {"global_orient": pred_body_pose[:, [0]],
                                "body_pose": pred_body_pose[:, 1:],
                                "betas": pred_betas}
            return pred_smpl_params, pred_cam, pred_smpl_params_list, last_token

    class HMR2WithToken(HMR2):
        """HMR2 子类：forward_step 额外透出 token_out。"""

        def forward_step(self, batch: dict) -> dict:
            x = batch["img"]
            batch_size = x.shape[0]
            conditioning_feats = self.backbone(x[:, :, :, 32:-32])
            pred_smpl_params, pred_cam, _, token_out = self.smpl_head(conditioning_feats)
            output = {}
            output["pred_cam"] = pred_cam
            output["pred_smpl_params"] = {k: v.clone()
                                          for k, v in pred_smpl_params.items()}
            output["token_out"] = token_out
            device = pred_smpl_params["body_pose"].device
            dtype = pred_smpl_params["body_pose"].dtype
            focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(
                batch_size, 2, device=device, dtype=dtype)
            pred_cam_t = torch.stack(
                [pred_cam[:, 1], pred_cam[:, 2],
                 2 * focal_length[:, 0] / (self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] + 1e-9)],
                dim=-1)
            output["pred_cam_t"] = pred_cam_t
            return output

    cfg = load_cfg(finetune_aug=finetune_aug)
    model = HMR2WithToken(cfg, init_renderer=False)
    model.smpl_head = SMPLHeadWithToken(cfg)
    ckpt_path = Path(checkpoint) if checkpoint else DEFAULT_CKPT
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"官方 ckpt 未下载: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state.get("state_dict", state)
    if all(k.startswith("model.") for k in sd):
        sd = {k[len("model."):]: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "discriminator" not in k]
    unexpected = [k for k in unexpected if "discriminator" not in k]
    if missing or unexpected:
        print(f"[load_hmr2_plain_with_token] missing={missing}")
        print(f"[load_hmr2_plain_with_token] unexpected={unexpected}")
    return model, cfg
