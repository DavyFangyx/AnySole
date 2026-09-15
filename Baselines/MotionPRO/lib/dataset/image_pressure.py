import torch
import numpy as np
from pathlib import Path
import csv

from torch.utils.data import Dataset

from lib.util.io import load_smpl_npy
from lib.util.workspace import sequence_root


def load_split_ids(split_csv, column):
    ids = []
    with Path(split_csv).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"{split_csv} missing column {column}")
        for row in reader:
            value = (row.get(column) or "").strip()
            if value:
                ids.append(value)
    return ids


def default_seq_root(cam_id):
    return sequence_root(cam_id)


def is_default_cam_seq_root(seq_root):
    value = str(seq_root or '')
    if not value or value in ('None', 'none', 'null'):
        return True
    normalized = Path(value).as_posix().rstrip("/")
    prefixes = ('data/sequences/', 'workspace:/derived/MotionPRO/sequences/')
    prefix = next((item for item in prefixes if normalized.startswith(item)), None)
    if prefix is None:
        return False
    name = normalized[len(prefix):]
    return name.startswith('cam') and name[3:].isdigit() and '/' not in name


def apply_cam_seq_root(task):
    cam_id = int(task.get('cam_id', 3))
    task['cam_id'] = cam_id
    seq_root = str(task.get('seq_root', '') or '')
    if is_default_cam_seq_root(seq_root):
        task['seq_root'] = str(default_seq_root(cam_id))
    return cam_id, task['seq_root']


def resolve_seq_root(cfg):
    task = cfg['task']
    apply_cam_seq_root(task)
    return Path(task['seq_root'])


def feature_path_for_session(seq_root, session_id, cam_id=None):
    matches = sorted(seq_root.glob(f"*/*/{session_id}/feature_hrnet.pth"))
    if not matches:
        cam_text = f' cam_id={cam_id}' if cam_id is not None else ''
        raise FileNotFoundError(
            f"No feature_hrnet.pth for {session_id} under {seq_root}.{cam_text} "
            f"Missing training files for this camera."
        )
    return matches[0]

class ImagePressureDataset(Dataset):

    def __init__(self, cfg, mode):
        self.mode = mode
        self.window_length = cfg['task']['window_length']
        self.dtype = torch.float32
        self.pressure_hw = (96, 96)

        seq_root = resolve_seq_root(cfg)
        cam_id = cfg['task'].get('cam_id')
        split_csv = cfg['task'].get('split_csv')
        contact_method = str(cfg['task'].get('contact_method', 'tactile_abs'))
        if split_csv:
            if self.mode == 'train':
                column = 'train'
            elif self.mode == 'eval':
                column = 'val'
            elif self.mode == 'test':
                column = 'test'
            else:
                raise ValueError(f'Unknown dataset mode: {mode}')
            session_ids = load_split_ids(split_csv, column)
            self.feature_files = [
                str(feature_path_for_session(seq_root, session_id, cam_id))
                for session_id in session_ids
            ]
        elif self.mode == 'train':
            from lib.util.io import findAllFilesWithSpecifiedName
            self.feature_files = findAllFilesWithSpecifiedName(cfg['task']['train_data_dir'], 'feature_hrnet.pth')
        elif self.mode in ('eval', 'test'):
            from lib.util.io import findAllFilesWithSpecifiedName
            self.feature_files = findAllFilesWithSpecifiedName(cfg['task']['eval_data_dir'], 'feature_hrnet.pth')
        else:
            raise ValueError(f'Unknown dataset mode: {mode}')
        print(f"{self.mode} seq_root={seq_root} n_sessions={len(self.feature_files)}")

        self.gt_kps = []
        self.gt_contact = []
        self.gt_verts = []
        self.gt_smpl = []
        self.feature_all = []
        self.pressure_all = []
        self.fake_all = []
        self.valid_windows = []

        for file_index, feature_file in enumerate(self.feature_files):
            feature = torch.load(feature_file)
            self.feature_all.append(feature)

            pressure_file = feature_file.replace('feature_hrnet.pth', 'pressure.npz')
            gt_kps_file = feature_file.replace('feature_hrnet.pth', 'keypoints.npy')
            gt_smpl_file = feature_file.replace('feature_hrnet.pth', 'smpl.npy')
            if contact_method == 'tactile_abs':
                contact_file = feature_file.replace('feature_hrnet.pth', 'contact.npy')
            else:
                contact_file = feature_file.replace('feature_hrnet.pth', f'contact_{contact_method}.npy')
            fake_file = feature_file.replace('feature_hrnet.pth', 'fake_mask.npy')

            pressure = np.load(pressure_file)['pressure']
            pressure = self._resize_pressure(pressure)
            kps_gt = np.load(gt_kps_file)
            if not Path(contact_file).is_file():
                raise FileNotFoundError(
                    f"Missing contact labels {contact_file} (contact_method={contact_method}). Run "
                    f"`python results_display/script/contact_methods.py --methods {contact_method}` "
                    f"to generate them."
                )
            contact_gt = np.load(contact_file)
            smpl_gt = load_smpl_npy(gt_smpl_file)
            if Path(fake_file).is_file():
                fake_mask = np.load(fake_file).astype(np.uint8).reshape(-1)
            else:
                fake_mask = np.zeros((kps_gt.shape[0],), dtype=np.uint8)

            n_frames = kps_gt.shape[0]
            n_windows = (n_frames + self.window_length - 1) // self.window_length
            for window_idx in range(n_windows):
                left = window_idx * self.window_length
                right = min(n_frames, (window_idx + 1) * self.window_length)
                if fake_mask[left:right].any():
                    continue
                self.valid_windows.append((file_index, left, right, n_frames))

            self.pressure_all.append(pressure)
            self.gt_kps.append(kps_gt)
            self.gt_smpl.append(smpl_gt)
            self.gt_contact.append(contact_gt)
            self.fake_all.append(fake_mask)

        print('Valid windows: ', len(self.valid_windows))

    def __len__(self):
        return len(self.valid_windows)

    def _resize_pressure(self, pressure):
        height, width = self.pressure_hw
        pressure = torch.from_numpy(np.ascontiguousarray(pressure)).float()
        if pressure.ndim != 3:
            raise ValueError(f'pressure must be (T, H, W), got {tuple(pressure.shape)}')
        if tuple(pressure.shape[-2:]) == (height, width):
            return pressure
        return torch.nn.functional.interpolate(
            pressure.unsqueeze(1),
            size=(height, width),
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)

    def __getitem__(self, idx):
        item = {}
        file_index, window_left, window_right, _origin_max_index = self.valid_windows[idx]
        session_id = Path(self.feature_files[file_index]).parent.name

        pressure = self.pressure_all[file_index][window_left:window_right] / 255
        feature = self.feature_all[file_index][window_left:window_right, :]
        keypoint = self.gt_kps[file_index][window_left:window_right, :23]
        contact = self.gt_contact[file_index][window_left:window_right, :10]
        smpl_gt = self.gt_smpl[file_index]

        n_real = window_right - window_left
        beta = torch.from_numpy(smpl_gt['betas'][:10]).repeat(n_real, 1).float()
        body_pose = torch.from_numpy(smpl_gt['body_pose'][window_left:window_right]).float()
        global_orient = torch.from_numpy(smpl_gt['global_orient'][window_left:window_right]).float()
        transl = torch.from_numpy(smpl_gt['transl'][window_left:window_right]).float()
        theta = torch.cat([global_orient, body_pose], dim=1)

        keypoint = torch.from_numpy(keypoint).float()
        contact = torch.from_numpy(contact).float()
        valid = torch.ones(n_real, dtype=self.dtype)
        frame_index = torch.arange(window_left, window_right, dtype=torch.long)

        if feature.shape[0] < self.window_length:
            pad = self.window_length - feature.shape[0]
            pressure = torch.cat([pressure, torch.zeros(pad, pressure.shape[1], pressure.shape[2], dtype=self.dtype)], dim=0)
            feature = torch.cat([feature, torch.zeros(pad, feature.shape[1], dtype=self.dtype)], dim=0)
            keypoint = torch.cat([keypoint, torch.zeros(pad, keypoint.shape[1], keypoint.shape[2], dtype=self.dtype)], dim=0)
            contact = torch.cat([contact, torch.zeros(pad, contact.shape[1], dtype=self.dtype)], dim=0)
            beta = torch.cat([beta, torch.zeros(pad, beta.shape[1], dtype=self.dtype)], dim=0)
            theta = torch.cat([theta, torch.zeros(pad, theta.shape[1], dtype=self.dtype)], dim=0)
            transl = torch.cat([transl, torch.zeros(pad, transl.shape[1], dtype=self.dtype)], dim=0)
            valid = torch.cat([valid, torch.zeros(pad, dtype=self.dtype)], dim=0)
            frame_index = torch.cat([frame_index, torch.full((pad,), -1, dtype=torch.long)], dim=0)

        item['pressure'] = pressure
        item['feature'] = feature
        item['beta'] = beta
        item['theta'] = theta
        item['trans'] = transl
        item['joint'] = keypoint
        item['contact'] = contact
        item['valid'] = valid
        item['frame_index'] = frame_index
        item['session_id'] = session_id
        return item
