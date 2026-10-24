"""触觉数据管线（端到端一处）：原始 CSV → 四级清洗 → 行 npz → 帧对齐。

- 清洗：``rebuild.py``（fault_cell → quality_rules → noise_model →
  baseline_core + detector），raw CSV → dataset npz / processed_csv；
- 对齐：``align.py``（绝对墙钟最近行 → 视觉 40Hz 帧网格）；
- 数据根：``AnysoleWorkspace/work/data_pipeline/tactile_v3/``
  （inputs/raw_csv 原始、data/dataset 行 npz、data/processed_csv）。
"""
