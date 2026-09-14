import math
import os
from collections import defaultdict

import hydra
import numpy as np
import smplx
import torch
from loguru import logger as log
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from app.train_frappe import expand_session_ids, flatten_batch, parse_cli_args
from lib.dataset.image_pressure import ImagePressureDataset, apply_cam_seq_root
from lib.eval.metrics import DEFAULT_FPS, compute_session_metrics, write_metrics_files
from lib.model.FRAPPE import FRAPPE
from lib.util.workspace import resolve_path as resolve_workspace_path


class SessionAccumulator:
    def __init__(self):
        self.n_frames = 0
        self.pred_joints = {}
        self.gt_joints = {}
        self.pred_verts = {}
        self.gt_verts = {}
        self.valid = {}

    def add(self, frame_idx, pred_joint, gt_joint, pred_vert, gt_vert, keep):
        frame_idx = int(frame_idx)
        if frame_idx < 0:
            return
        self.n_frames = max(self.n_frames, frame_idx + 1)
        self.pred_joints[frame_idx] = np.asarray(pred_joint, dtype=np.float32)
        self.gt_joints[frame_idx] = np.asarray(gt_joint, dtype=np.float32)
        self.pred_verts[frame_idx] = np.asarray(pred_vert, dtype=np.float32)
        self.gt_verts[frame_idx] = np.asarray(gt_vert, dtype=np.float32)
        self.valid[frame_idx] = bool(keep)

    def packed(self):
        n = self.n_frames
        if n == 0:
            empty_j = np.zeros((0, 23, 3), dtype=np.float32)
            empty_v = np.zeros((0, 0, 3), dtype=np.float32)
            return empty_j, empty_j, empty_v, empty_v, np.zeros((0,), dtype=bool)
        sample_v = next(iter(self.pred_verts.values()))
        n_verts = sample_v.shape[0]
        pred_j = np.zeros((n, 23, 3), dtype=np.float32)
        gt_j = np.zeros((n, 23, 3), dtype=np.float32)
        pred_v = np.zeros((n, n_verts, 3), dtype=np.float32)
        gt_v = np.zeros((n, n_verts, 3), dtype=np.float32)
        keep = np.zeros((n,), dtype=bool)
        for t, is_valid in self.valid.items():
            if not is_valid:
                continue
            pred_j[t] = self.pred_joints[t]
            gt_j[t] = self.gt_joints[t]
            pred_v[t] = self.pred_verts[t]
            gt_v[t] = self.gt_verts[t]
            keep[t] = True
        return pred_j, gt_j, pred_v, gt_v, keep


def resolve_path(path, orig_cwd):
    return resolve_workspace_path(path, orig_cwd)


def resolve_checkpoint(cfg, orig_cwd, task_info):
    explicit = str(cfg['task'].get('checkpoint_path', 'None'))
    if explicit not in ('', 'None', 'none', 'null'):
        path = resolve_path(explicit, orig_cwd)
        if not os.path.isfile(path):
            raise FileNotFoundError(f'checkpoint_path does not exist: {path}')
        return path
    checkpoint_dir = resolve_path(cfg['task']['checkpoint_dir'], orig_cwd)
    path = os.path.join(checkpoint_dir, task_info, 'imagepressure2smpl_best.pth')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Missing best checkpoint: {path}')
    return path


def fmt_metric(value):
    if value is None or not math.isfinite(value):
        return 'nan'
    return f'{value:.3f}'


def smpl_output(smpl, beta, theta, trans):
    pred_global_orient, pred_body_pose = torch.split(theta, [3, 69], dim=1)
    return smpl(
        betas=beta,
        body_pose=pred_body_pose,
        global_orient=pred_global_orient,
        transl=trans,
    )


def evaluate_checkpoint(cfg, checkpoint_path, result_dir, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fps = float(cfg['task'].get('eval_fps', DEFAULT_FPS))
    os.makedirs(result_dir, exist_ok=True)

    model = torch.nn.DataParallel(FRAPPE())
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    smpl = smplx.create(cfg['task']['smpl_model']).to(device)
    dataset = ImagePressureDataset(cfg, 'test')
    dataloader = DataLoader(
        dataset,
        batch_size=cfg['task']['batch_size'],
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )
    sessions = defaultdict(SessionAccumulator)

    with torch.no_grad():
        for item in dataloader:
            item = flatten_batch(item, device)
            smpl_params = model(item['feature'], item['pressure'])
            _, pred_theta, pred_trans = torch.split(smpl_params, [10, 72, 3], dim=1)
            pred_smpl = smpl_output(smpl, item['beta'], pred_theta, pred_trans)
            gt_smpl = smpl_output(smpl, item['beta'], item['theta'], item['trans'])
            pred_joint = pred_smpl.joints[:, :23]
            gt_joint = gt_smpl.joints[:, :23]
            pred_verts = pred_smpl.vertices
            gt_verts = gt_smpl.vertices
            n_frames = pred_joint.shape[0]
            valid = item['valid'].reshape(n_frames) > 0.5
            frame_index = item['frame_index'].reshape(n_frames).detach().cpu().numpy()
            session_ids = expand_session_ids(item['session_id'], n_frames)
            pred_j_np = pred_joint.detach().cpu().numpy()
            gt_j_np = gt_joint.detach().cpu().numpy()
            pred_v_np = pred_verts.detach().cpu().numpy()
            gt_v_np = gt_verts.detach().cpu().numpy()
            valid_np = valid.detach().cpu().numpy()
            for sid, t, pj, gj, pv, gv, keep in zip(
                session_ids, frame_index, pred_j_np, gt_j_np, pred_v_np, gt_v_np, valid_np
            ):
                sessions[sid].add(t, pj, gj, pv, gv, keep)

    rows = []
    for sid in sorted(sessions):
        pred_j, gt_j, pred_v, gt_v, keep = sessions[sid].packed()
        rows.append(compute_session_metrics(
            pred_j, gt_j, pred_v, gt_v, keep, fps=fps, session_id=sid
        ))
        row = rows[-1]
        log.info(
            f"{sid}: n={int(row['n_valid_frames'])} "
            f"MPJPE={fmt_metric(row['MPJPE'])} PMPJPE={fmt_metric(row['PMPJPE'])} "
            f"PVE={fmt_metric(row['PVE'])} Accel={fmt_metric(row['Accel'])} "
            f"WMPJPE={fmt_metric(row['WMPJPE'])} WAMPJPE={fmt_metric(row['WAMPJPE'])} "
            f"RTE={fmt_metric(row['RTE'])} Jitter={fmt_metric(row['Jitter'])} WBCE={fmt_metric(row['WBCE'])}"
        )

    csv_path, log_path = write_metrics_files(rows, result_dir, fps=fps)
    log.info(f'Wrote {csv_path}')
    log.info(f'Wrote {log_path}')
    return csv_path, log_path


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['task']['gpu'])
    orig_cwd = hydra.utils.get_original_cwd()
    os.chdir(orig_cwd)
    apply_cam_seq_root(cfg.task)
    for key in ('split_csv', 'seq_root', 'smpl_model', 'smpl_mean_params', 'checkpoint_dir', 'result_dir'):
        if cfg.task.get(key):
            cfg.task[key] = resolve_path(cfg.task[key], orig_cwd)

    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    hydra_name = hydra.core.hydra_config.HydraConfig.get().job.name
    log.add(os.path.join(hydra_path, hydra_name + '.log'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task_info = os.path.join(str(cfg['task']['name']), str(cfg['task']['loss_name']), str(cfg['task']['learning_rate']))
    checkpoint_path = resolve_checkpoint(cfg, orig_cwd, task_info)
    result_dir = os.path.join(cfg['task']['result_dir'], task_info)

    log.info(f'Device: {device}')
    log.info(f'Checkpoint: {checkpoint_path}')
    log.info(f'Sequence root: {cfg.task.seq_root}')
    log.info(f'Result dir: {result_dir}')
    log.info(f'Configuration: {OmegaConf.to_yaml(cfg)}')
    evaluate_checkpoint(cfg, checkpoint_path, result_dir, device=device)


if __name__ == '__main__':
    cli_args, remaining = parse_cli_args()
    import sys
    sys.argv = [sys.argv[0]] + remaining
    if cli_args.wandb_mode is not None:
        os.environ['WANDB_MODE'] = cli_args.wandb_mode
    if cli_args.wandb_project is not None:
        os.environ['WANDB_PROJECT'] = cli_args.wandb_project
    if cli_args.wandb_entity is not None:
        os.environ['WANDB_ENTITY'] = cli_args.wandb_entity
    main()
