# Offline experiment scheduler

`configs/` only manages data-producing model tasks. Analysis and
visualization scripts live in `results_display/script/` and are not queue
tasks.

## Data-producing experiments

```text
configs/gen/singlemodal_eval.py  -> results/singlemodal_eval/
configs/gen/rho_grid_eval.py     -> results/rho_grid_eval/
```

The generators create one `.conf` per model task directly in `queue/`.

```bash
python configs/gen/singlemodal_eval.py
python configs/gen/rho_grid_eval.py --models V3_3B
```

The generated task then follows the only scheduler state machine:

```text
queue/ -> running/ -> done/
                    └-> failed/
```

The four directories are intentionally retained. `logs/` and
`generated/tasks/` are not part of the new design. Task logs are written as
`configs/<task-name>.log`; worker logs are `configs/scheduler_GPU<N>.log`.

Start one worker per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash configs/bg.sh
CUDA_VISIBLE_DEVICES=1 bash configs/bg.sh
python configs/tools/status.py
```

For a foreground/debug run, generate tasks first and then use:

```bash
python configs/tools/runner.py task configs/queue/<task>.conf --dry-run
```

## Analysis and visualization

These scripts consume completed results and write to `results_display/`:

```bash
python results_display/script/singlemodal_analysis.py
python results_display/script/rho_grid_analysis.py
python results_display/script/complement.py
python results_display/script/dropout_ablation.py
python results_display/script/v2t_upper.py
python results_display/script/trust.py
```

The source data is declared by each display script:

| Display script | Input experiment(s) |
|---|---|
| `singlemodal_analysis.py` | `singlemodal_eval` |
| `rho_grid_analysis.py` | `rho_grid_eval` |
| `complement.py` | `singlemodal_eval` |
| `dropout_ablation.py` | `singlemodal_eval` |
| `v2t_upper.py` | `singlemodal_eval`, `rho_grid_eval` |
| `trust.py` | `singlemodal_eval`, `rho_grid_eval` |

The complete model/configuration matrix is documented in
`configs/gen/README.md`.
