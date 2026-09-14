# PressureWasher -> MotionPRO

Convert reviewed PressureWasher sessions into MotionPRO sequence folders.

```bash
python data_prep/prepare_sequences.py --cam-id 3 --overwrite
python data_prep/prepare_sequences.py --cam-id 3 --session S13102 --overwrite
python data_prep/prepare_sequences.py --cam-id 3 --refresh-contact
```

Outputs land in `../../AnysoleWorkspace/derived/MotionPRO/sequences/cam<id>/<date>/<subject>/<session_id>/` with `color/`, `pressure.npz`, `smpl.npy`, `contact.npy`, `fake_mask.npy`.
Train/val/test splits are one CSV at `../../AnysoleWorkspace/splits/default/splits.csv`. `val` equals `test`.
Complete sequences are used unless `--exclude` drops their subjects. Default assignment is IID; `--ood` holds those subjects out of train.

```bash
python data_prep/make_splits.py --cam-id 3
python data_prep/make_splits.py --cam-id 3 --ood S11,S10
python data_prep/make_splits.py --cam-id 3 --exclude S11,S10
```

- Default IID: group by subject + action. One take stays in train; more than one take leaves the largest trial in val/test.
- `--ood S11,S10`: those subjects go entirely to val/test; remaining subjects stay IID.
- `--exclude S11,S10`: those subjects are omitted from the CSV.

Training reads `task.split_csv` (default `../../AnysoleWorkspace/splits/default/splits.csv`):

```bash
python -m app.train_frappe
```
Sessions marked `C` or `D` in `AnysoleWorkspace/sources/PressureWasher/outputs/stats/pressure_stats_20260814_231054/overall/missing_pressure_objects.csv` are skipped.
`contact.npy` is `(T, 10)` and only columns 6/7 (left/right foot) can be 1; hips, knees, ankles, and hands stay 0.

After conversion, generate the remaining MotionPRO files from the MotionPRO repo root. bbox must exist before image features:

```bash
cd /data/fangyuxuan/projects/gait/Baselines/MotionPRO
conda activate bbox_scan
python -m lib.util.gen_bbox --cam-id 3
conda activate touch_gait
python -m lib.util.gen_image_feature --cam-id 3
python -m lib.util.gen_kps --cam-id 3
```
