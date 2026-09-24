# Experiment generation matrix

Only the following two groups produce model/evaluation data. The remaining
analysis is deliberately decoupled and lives under `results_display/`.

## `singlemodal_eval`

Generator: `singlemodal_eval.py`

| Role | Model | Configuration |
|---|---|---|
| Main | `V3_3B` | Full multimodal model configuration from the registry |
| V specialist | `V3_3B_vonly` | Main model with `vonly` sampling configuration |
| T specialist | `V3_3B_tonly` | Main model with `tonly` sampling configuration |

Each model task performs checkpoint materialization when necessary, then
formal validation and test evaluation.

Produces main, V-only and T-only val/test fseries, model manifests and
checkpoint provenance. It is consumed by the singlemodal, complement,
dropout, V-to-T and trust analyses.

## `rho_grid_eval`

Generator: `rho_grid_eval.py`

| Parameter | Values |
|---|---|
| Model | `V3_3B` |
| rho grid | `0, 20, 40, 60, 80, 100` |
| train representation rho | `0, 100` |
| validation seeds | `0, 1, 2` |
| test seeds | `0` |

It produces split-specific rho-grid metrics under
`results/rho_grid_eval/<model>/`. It is consumed by rho-grid, V-to-T and
trust analyses.

## Execution contract

Generators write task confs directly into `configs/queue/`. They use a
global numeric prefix across `queue/`, `running/`, `done/` and `failed/`, so
the two generators can be run independently without overwriting task files.

Analysis scripts do not enter the queue and do not participate in the four
state directories. Run them after their declared input results are complete.
