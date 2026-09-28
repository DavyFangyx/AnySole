# AnySole 执行归档（Agent：Claude，2026-09-29）

## 1. 提交清单

本会话**零 commit**（主库与 Baselines 库均无），遵守任务书铁律"只写档案、不 commit/push，提交由主会话审核后统一执行"。全部改动留在工作区，§2 各条目的 commit 一栏待主会话统一提交后回填；裁定与证据均已齐备（见 §2/§4）。

| 短 hash | 仓库(主库/Baselines) | 主题 | 类型(D/T1/T2/M/审计/文档) |
|---|---|---|---|
| — | — | 无（零 commit，待主会话统一提交） | — |

## 2. 修改条目

编号规则：无既有编号，自拟 `AnySole-<序号>` 并在表后声明。类型取：裁定执行（用户 09-28 会话拍板）/ 审计 / T1（数据契约 display 收尾）。

| 修改编号 | 类型 | 位置(文件:行) | 原生行为 → 当前行为 | 对应裁定/条目号 | 证据(测试名+结果) | 状态 |
|---|---|---|---|---|---|---|
| AnySole-1 | 裁定执行 | anysole/utils/eval_protocol.py:132-224,761-784 | metrics 块 snake_case 键 + 字母序（明细 sort_keys=True）→ 经典大写展示键名（PA-MPJPE/MPJPE/MPJRE/W-MPJPE100/traj_ATE/RootOrientationDrift/FootSliding/Jitter_gt/Jitter_pred，明细另含 WA-MPJPE100/RootOrientation/AccelError）+ 固定顺序 PA-MPJPE 第一 + 两文件 sort_keys=False；映射仅发生在序列化层（METRIC_DISPLAY_NAMES/_display_block/display_metrics/order_payload），内部累积保持 snake_case | 用户 09-28 拍板："没问题，明细文件也是相同风格；新键名风格无误，可以执行" | tests/test_public_metrics.py 全套 101 passed 0 failed（含新增断言：PA-MPJPE 首位、映射表覆盖白名单） | 完成 |
| AnySole-2 | 裁定执行 | anysole/backfill_brief.py:35-45,96-120,143-165,196-215 | 回填只认内部键名、detail 用 sort_keys=True → 支持展示键名幂等回填（restore_display_names 先还原再派生）+ 两文件按新口径写出；阶段 1 fseries 同步 | 同 AnySole-1 | results/AnySole 9 对文件重写成功；连跑三次 md5 一致（字节级幂等）；V3_3B MPJPE 57.5194 与改写前逐位一致 | 完成 |
| AnySole-3 | 裁定执行 | results_display/script/d_test2_dataset_check.py:391 | `vt2m.get("mpjpe_mm")` → `vt2m.get("MPJPE")`（唯一读这份 JSON 键名的消费端） | 同 AnySole-1 | 全仓 grep 确认无其他消费端；py_compile 通过 | 完成 |
| AnySole-4 | 裁定执行 | anysole/eval.py:180-186,291-323,456-466 | 无 --sweep（手动 eval 漏 --variant 直接 FileNotFoundError）→ `--model-name X --sweep` 扫描 results/AnySole/X_*/ 全部 variant 逐一评估（_sweep_targets），--contact-method 失效（目录名自带）、--split 失效（强制 test 并打印提示），支持 CSV 与 --which best | 用户 09-28 口述裁定："加上一个参数 --sweep…contact-method split 全都失效，直接开扫" | _sweep_targets 实测命中 V3_3B/V3_4a/V3_4b/CSV 的 ckpt_last/best；monkeypatch 端到端实测：打印 "--sweep ignores --split; using test"、命中 variant ckpt、循环接线正常、rc=0 | 完成 |
| AnySole-5 | 裁定执行 | results_display/script/utils/cli_common.py:201-207,119-133；r_test1_visualize_anysole.py:186,246-250；r_test3_traj.py:464,616-625,758-770 | 两可视化工具无 --sweep → 共享 add_common_args(sweep=True) + sweep_anysole_model_dirs，--sweep 时遍历全部 variant 出图（Test3 的 --compare 视图同接） | 同 AnySole-4 | sweep_anysole_model_dirs 实测输出正确（含空模型返回 []）；Test1 冒烟 S13011 163 帧 GIF 落盘、Test3 冒烟 GIF+PNG 落盘 traj_ATE=69.7mm | 完成 |
| AnySole-6 | 审计 | Baselines/utils（符号链接 → ../Baselines_old/utils，主仓库工作树内、独立库 untracked） | `import Baselines.utils.*` ModuleNotFoundError → 全部可导入；零代码改动、不碰冻结的 Baselines_old 本体 | 两库拆分遗留断链（记忆 baselines-independent-repo-20260928："第 4 步被叫停，ImportError 属预期"；用户 09-28 贴报错要求修复） | test_baseline_utils 34 passed 0 failed、test_protocol_consumer 61 passed 0 failed（拆分后首度转绿）；两可视化 --help 走通 import 链 | 完成 |
| AnySole-7 | T1 | results_display/script/r_test1_visualize_anysole.py:57-81,90-104,187-193 | 读旧布局 pressure.npz 2-D 画布 + fake_mask.npy（全盘 0 个）→ shared facts pressure_48.npz（left48+right48 拼接 == 旧 pressure 口径）经 load_shared_session 装载（旧文件存在时回退）；触觉面板 48-token 直接 reshape(4,12)（与 cop_from_grid/crop_foot 同布局）+ PRESSURE_CLIP 归一 0-255（模型 normalize_raw 口径） | T1 数据契约（display 收尾，无独立条目号，自拟） | Test1 冒烟：pressure (325,96) float32 装载成功、163 帧 GIF 落盘 | 完成 |
| AnySole-8 | T1 | results_display/script/utils/motion_io.py:21-45 | load_session_gt 读已废弃 align_meta.json → 无该文件时走 frames.npz mocap_time_s + session.json source_files URI（shared_smpl_path/shared_bvh_path）；旧布局回退 _baselines_load_session_gt 原实现 | T1 数据契约（display 收尾，自拟） | Test1/Test3 冒烟均过；GT joints (325,24,3) 实测装载正确 | 完成 |

自拟编号声明：AnySole-1~AnySole-8 均无既有条目号；对应裁定分别为用户 09-28 会话两次拍板（展示键名+明细同风格；--sweep 三工具）与两库拆分/T1 既有档案（AnySole-6~8 挂靠）。

## 3. 数据产物

| 产物 | 路径 | 计数/大小(磁盘实测) | artifact.json(有/无) | 源 hash |
|---|---|---|---|---|
| 明细+简略 JSON 重写 | results/AnySole/*/tw*/metrics/test.json + test_brief.json | 18 文件（9 模型 × 2），合计 180K（du -ch 实测） | 无 | 无源数据输入（纯键名/顺序文本重写；V3_3B MPJPE 57.5194 改写前后逐位一致；backfill 三次运行 md5 一致） |
| Baselines.utils 符号链接 | Baselines/utils → ../Baselines_old/utils | 1 个符号链接（22 B，untracked） | 无 | — |
| Test1 冒烟产物 | results_display/ResultTest/R1Test_visualize/AnySole/V3_4b_joint_and/…/VT2M/bone/gif/S13011_VT2M_compare.gif | 1 个 / 4.0M（163 帧） | 无 | — |
| Test3 冒烟产物 | …/R3Test_traj/…/VT2M/gif/S13011_VT2M_traj.gif + png/S13011_VT2M_traj.png | 2 个 / 1.3M + 220K（traj_ATE=69.7mm） | 无 | — |
| 用户随后全量运行（非本会话产生，仅实测登记） | R1Test_visualize gif / R3Test_traj gif / R3Test_traj png | 139 个/571M、141 个/207M、141 个/30M（其中 S13011 三件为本会话冒烟） | 无 | — |

## 4. 挂账与偏差

1. **全部改动未提交**（主库 9 个文件 + 新库 1 个 untracked 符号链接 + results 18 个 gitignored JSON）：anysole/backfill_brief.py→AnySole-2、anysole/eval.py→AnySole-4、anysole/utils/eval_protocol.py→AnySole-1、results_display/script/d_test2_dataset_check.py→AnySole-3、r_test1_visualize_anysole.py→AnySole-5/7、r_test3_traj.py→AnySole-5、utils/cli_common.py→AnySole-5、utils/motion_io.py→AnySole-8、tests/test_public_metrics.py→AnySole-1、Baselines/utils 符号链接→AnySole-6；待主会话统一提交（注意工作区内另有他人条目：command_manual.md、z_note 删除、Baselines_Backup 删除、PhysPT submodule，非本会话）。
2. **test_fpp_posetransopt_adapter 4/9 失败**：PoseTransOpt 子进程（run_full_mmvp.py --gpu 0）exit 1，GPU 脚本调用失败，与本会话改动无关、未修；其余三套测试全绿（101/61/34）。
3. **d_test1/d_test2/d_test3_contact/d_test5 等仍读 align_meta.json**（同类过期未适配，挂账）——已写入记忆档案 display-layer-shared-facts-adapt-20260928。
4. **Baselines/utils 符号链接 untracked 待拍板**：新库 .gitignore 登记 `/utils` 或第 4 步迁移恢复时删链改接——同上记忆档案。
5. **触觉面板视觉口径变更**：48-token 归一化渲染 ≠ 旧 pressure_tookit 画布渲染（已知效果亏空项）——同上记忆档案。
6. **eval 手动跑必须带 --variant 或 --sweep**（缺 --variant 时推断路径停在 model_dir 层报 FileNotFoundError）——同上记忆档案。
7. 展示键名/顺序裁定（AnySole-1~3）已写入记忆档案 metrics-two-file-brief-detail-20260926.md（Claude memory，非 z_note 档案）。

## 自查

1. **commit 对账**：主库 `git log --oneline`（截至 bf26da7）与 Baselines `git -C Baselines log --oneline`（截至 cedd921）中无本会话 commit——§1 声明"零 commit"与实际一致，无漏档。
2. **未提交工作区改动**：主库 `git status` 中属本会话的 9 个文件 + 新库 `?? utils` 已全部单列于 §4 第 1 条并注明归属条目；results 18 个重写 JSON 在 gitignore 内，记入 §3。
3. **Baselines 树内改动**：本会话未改任何 Baselines 树内已跟踪代码（`git -C Baselines diff 111220c` 无输出），仅新增 untracked 符号链接 `utils`，已由 §2 AnySole-6 覆盖。
