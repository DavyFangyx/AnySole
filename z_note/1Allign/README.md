# 1Allign —— R2 对齐残差评估脚本归档

来源：`/tmp/r2-scratch/`（2026-09-28，R2 S5091 深挖审计）+ `/tmp/s1-scratch/`（2026-09-29，S1 全库 140 session 扫描）。原样复制，未改动，未适配路径。证据与结论全文见 `Baselines/决策/05_R1-R3_风险裁定.md` §三（R2 裁定 §1–§5、S1 扫描 §6）。

## 目录

- `r2_scratch/` —— S5091 单 session 深挖（09-28，裁定代理 R2）
  - `s1_chain.py`/`s1b_chain.py`/`s2_proj.py`：投影链（3D GT → cam3 标定 → 像素）与链核对
  - `s3_bvh.py`/`s3b_bvh.py`：BVH 侧核对；`s4_verdict.py`/`s5_bias.py`：五对照组（轴/单位/尺度/R-t/位精确）
  - `s6_perjoint.py`/`s7_fit.py`/`s8_final.py`：逐关节误差与几何链终验
  - `s9_timeshift.py`/`s9b.py`/`s10_lag_decisive.py`：时间平移扫描与 +9.5 帧判定
  - `s11_phase.py`/`s12_spectrum.py`/`s13_dir.py`/`s14_subframe.py`：相位/频谱/方向/亚帧（亚帧极小 +9.25~+9.75）
  - `s15_multi.py`/`s18_wide.py`：多 session 扫描（窄窗 −4..+20 与宽窗 −30..+30，0.5 帧步长）→ `multi_lag.json`/`wide_lag.json`
  - `s16_iou_lag.py`/`s17_confirm.py`/`s19_validate.py`/`s20_blast.py`/`s21*.py`/`s22_table.py`：YOLOX IoU 独立复核与汇总表
- `s1_scratch/` —— S1 全库扫描（09-29，执行代理 S1）
  - `sweep_lib.py`：共享库（sessions/load_session/sweep）
  - `s30_fleet.py`：全库三口径扫描（wide 0.5 步 / narrow / wideint）→ `full_lag.json` 等
  - `s31_verify.py`：复跑一致性（140/140、53/53 位精确）；`s32_classify.py`/`s33_table.py`/`s34_reclass.py`/`s35_joint.py`：A/B/C/D 分类与回归
  - `s40–s42`：YOLOX IoU 车队复核；`s50–s53`：RTMPose×YOLOX 对比与裁定 → `adjudication.json`
  - `s60/s61`：分段漂移；`s70/s71`：抽样；`s80_final.py`：终表 → `lag_fleet.csv` / `lag_fleet_final.json`
- 关键结果表（已随带）：`s1_scratch/lag_fleet.csv`（逐 session lag_eff 表，A/B/C/D 分类）、`s1_scratch/adjudication.json`、`s1_scratch/clock_probe.json`、`r2_scratch/multi_lag.json`、`r2_scratch/wide_lag.json`

## 使用

- 只读参考/移植用，路径硬编码（`/data/fangyuxuan/projects/gait/AnysoleWorkspace` 与 `/tmp/*-scratch` 输出）。
- 输入：`model_inputs/MotionPRO/adapter_v1/cam3/`（keypoints.npy/frame_id.npy/bbox.npy）、`shared/frontends/rtmpose_halpe26/`、`protocol/calibration/<date>.json`。
- 主结论复现：`python3 s1_scratch/s30_fleet.py`（全库扫描）→ `s32_classify.py` → `s33_table.py`。
- 大件中间产物（npy/jpg）未复制，需要时从 `/tmp/r2-scratch/`、`/tmp/s1-scratch/` 原目录取。
