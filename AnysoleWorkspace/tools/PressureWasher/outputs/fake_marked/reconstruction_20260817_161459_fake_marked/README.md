# Fake-Marked Reconstructed Tactile Dataset

This directory is a post-processed version of `reconstruction_20260817_161459`.

## Layout

```text
reconstruction_20260817_161459_fake_marked/
├── <date>/
│   ├── <Sx>/
│   │   ├── <rec...>/
│   │   │   ├── pressure_left.csv
│   │   │   ├── pressure_right.csv
│   │   │   └── reconstruction_manifest.csv
│   │   └── ...
│   └── ...
```

Each session folder contains:

| File | Meaning |
|---|---|
| `pressure_left.csv` | Left-foot reconstructed tactile sequence |
| `pressure_right.csv` | Right-foot reconstructed tactile sequence |
| `reconstruction_manifest.csv` | Reconstruction blocks and bridge/resample metadata |

## Pressure CSV Columns

| Column | Meaning |
|---|---|
| `frame_idx` | Continuous row index within the side file |
| `t_us` | Reconstructed timestamp in microseconds |
| `valid_mask` | `1` = normal reconstructed frame, `0` = bridge frame |
| `fake` | `1` = frame listed in `PressureWasher/configs/fake_frames/*_fake_frames.csv`, `0` otherwise |
| `1..48` | 48 tactile pressure channels |

## Notes

- `source_frame_idx` and `source_t_us` are removed in this version.
- `fake` is applied per side using the matching `*_fake_frames.csv` file.
- `reconstruction_manifest.csv` is kept unchanged from the reconstruction step.
