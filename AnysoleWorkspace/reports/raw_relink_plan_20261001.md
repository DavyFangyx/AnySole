# raw 层重链/重组织方案（2026-10-01）

> 性质：**调研 + 设计，未做任何修改**。本文只描述方案，不含已执行动作。
> 前因：`AnysoleWorkspace/reports/workspace_organization_audit_20261001.md` §1 与「用户决策记录」第 5 条（raw 三别名指向同一棵树，要处理）。
> 用户已定约束（2026-10-01，本方案必须遵守）：
> ① `model_inputs/PoseTransOpt/adapter_v1/*/color/*.jpg` 的现状链接策略**不处理**，方案不得依赖改动它；
> ② `protocol/manifests/session_manifest.*` 与 `protocol/splits/default/splits.csv` 的**行内容不变**（可以评估 URI 字段是否需要改写，但方案不得要求改写）；
> ③ 上游 `/data/lizhe/projects/Tactile` 可写，但必须先证明安全（谁在消费）。

---

## 0. 结论速览

| 方案 | 一句话 | 结论 |
|---|---|---|
| **A 重排上游** | 把 `1_Data/<date>/mocap_ori_*` 等混装内容从采集树里搬走 | **判定不可行（本轮）**。上游是 lizhe 的活跃树（`.ai/TASK_LOG.md`、`2_Code_preprocess/analyze_smpl_reviews.py` 修改于 2026-09-27/28），有 12 处代码硬编码 `1_Data/<date>/mocap_ori_*` 结构、1 处按目录名 `parts.index("1_Data")` 反解路径、7,874 行 provenance 记录指向 `1_Data` 绝对路径；且**即使重排上游，workspace 侧仍要建同样的模态农场**（manifest URI 形状被冻结），收益≈0、风险与协调成本极高。证据见 §3.4。 |
| **B 只动 workspace** | raw 五个入口名字与 URI 语义不变，内容换成**按模态过滤、路径形状与 URI 逐段一致的链接农场** | **推荐**。零 manifest/splits 改动、零消费者代码改动、上游零改动；新增约 **2,800 个软链接 + 1,300 个实目录**，改 2 个文件（`tool/workspace.py`、`README.md`）+ 新增 1 个生成器脚本。见 §4。 |
| **C 混合（上游小改）** | 只做"安全的小改"（Mocap 日期别名、消 0808 双嵌套等） | **本轮不做**。逐条核验后发现每个"小改"都有现存消费方代价（`analyze_smpl_reviews.py` 的 `*/*smpl/fail.txt` glob、被冻结的 manifest URI、anysole `SMPL_ROOTS` 与 checkpoint 内嵌路径），净收益为负。见 §5。 |

**需要用户拍板的 5 个点**（详见 §8）：
1. 农场覆盖范围：只覆盖 4 个管线日期的 313 个 rec（推荐），还是连 0419/0422 一起共 445 个 rec（全量上游视图）；
2. `raw/rgb` 下相机目录用「目录级链接」（推荐，4 链接/rec）还是只做 cam3 的「文件级链接」；
3. 是否顺带把 3 处硬编码的 `3_Result/processed/rgb_human_masks` 登记为 `raw/` 的正式入口；
4. `raw/calibration` 是否保持原样（无任何运行时消费者，推荐保持）；
5. 是否同意"**新日期上线必须跑一次 `relink`**"成为流程纪律（上游会改 rec 目录名，见 §4.7）。

---

## 1. 现状核实（本文全部结论的证据基线）

仓库根 `ROOT=/data/fangyuxuan/projects/gait`，上游根 `UP=/data/lizhe/projects/Tactile`。

### 1.1 workspace 侧 raw 五入口

```
raw/rgb         -> UP/1_Data            (workspace.py:32)
raw/pressure    -> UP/1_Data            (workspace.py:33)  ← 与 rgb 同一目标
raw/bvh         -> UP/1_Data            (workspace.py:34)  ← 与 rgb 同一目标
raw/smpl        -> UP/Mocap             (workspace.py:35)
raw/calibration -> UP/0_Calibration     (workspace.py:36)
```
登记处：`AnysoleWorkspace/tool/workspace.py:31-37`（`RAW_LINKS`）、`AnysoleWorkspace/README.md:39-43`。
`doctor` 当前要求这 5 个名字必须是**指向固定目标的软链接**（`workspace.py:134-142`），实测 `python3 AnysoleWorkspace/tool/workspace.py doctor` 通过、全 workspace 断链 0。

### 1.2 上游 `1_Data` 逐日期结构（实测）

```
UP/1_Data/<YYYYMMDD>/
├── S#/recYYYYMMDD_HHMMSS_姓名_S<trial><action>_<take>/     # 采集目录
│   ├── 1/ 2/ 3/ 4/       四路相机 JPG 序列（每相机 ≈430–530 张）
│   ├── <n>_<HHMMSS>.mp4  同机位视频（rec 级，不在相机目录内）
│   ├── meta.json          （仅 08xx）
│   └── pressure_{left,right}.csv       （仅 08xx；04xx 为大写 Pressure_*.csv，命名口径不同）
├── mocap_ori_bvh/<Sdddd>/<Sdddd>_Skeleton<N>.bvh (+ .avi，仅 04xx/0804)
├── mocap_ori_c3d|trc|cmr/*.c3d|trc|cmr   （平铺，Sdddd 前缀，另有 calibration 例外文件）
└── mocap_csv/            （仅 0419/0422）
```

| 日期 | subject | rec 数 | meta.json | pressure L/R | bvh 会话目录 | bvh 文件 | avi | c3d/trc/cmr | mocap_csv |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 20260419 | S1 | 33 | 0 | 0（大写 33×2） | 33 | 34 | 34 | 34 | 有 |
| 20260422 | S2,S3,S4 | 99 | 0 | 0（大写 99×2） | 99 | 99 | 99 | 100 | 有 |
| 20260804 | S5,S6,S7 | 83 | 83 | 83×2 | 83 | 83 | 83 | 84 | 无 |
| 20260807 | S8 | 33 | 33 | 33×2 | 33 | 33 | 0 | 34 | 无 |
| 20260808 | S9,S10,S11,S12 | 131 | 131 | 131×2 | 131 | 131 | 0 | 132 | 无 |
| 20260810 | S13,S14 | 66 | 66 | 66×2 | 66 | 66 | 0 | 67 | 无 |

- 4 个管线日期的 313 个 rec **全部**是 `<rec>/{1,2,3,4}` 结构（实测 313/313 无变体）。
- 0807 另有**日期级**校准采集目录 `rec20260807_165200_calibration_1/`（只有 `1/2/3/4`）。
- bvh 根目录的例外文件：0807/0808/0810 有 `calibration_Skeleton0.bvh`，0422 有 `Calibration1_Skeleton0.bvh`，0419 用 `Take_calication.trc`（`2_Code_preprocess/mocap_camera_alignment/PIPELINE.md:35-36`）。
- c3d/trc/cmr 比 bvh 多 1（校准 session），**数量口径本身就不齐**。
- 上游近 30 天仍在改：`PROJECT_INDEX.md:31` 记录「20260814 共重命名 99 个 rec 目录」——**rec 目录名会变**，这是农场维护的直接约束。

### 1.3 上游 `Mocap/`（raw/smpl 的目标）

```
UP/Mocap/<MMDD>/                     # 0804/0807/0808/0810，无年份、无 0419/0422
├── mocap_ori_c3d/*.c3d              # 输入 c3d 的副本（85/34/133/67 个，含 calibration.c3d）
└── <MMDD>smpl/
    ├── batch_manifest.json, fail.txt
    └── mocap_ori_c3d/<Sddddd>/{motion_neutral_smpl.npz, conversion_manifest.json,
                                motion_quality.json, smpl_validation.json}
```
- npz 数：0804=83、0807=33、0808=119、0810=65，共 **300**。
- 该树 **owner = fangyuxuan**（我方产物），由外部批处理生成（`batch_manifest.json` schema=`chingmu53-smpl-batch-archive-v1`，created 2026-09-09），生成器**不在**本仓也不在上游树内。
- 上游有**一个**消费者：`UP/2_Code_preprocess/analyze_smpl_reviews.py:73` glob `Mocap/*/*smpl/fail.txt` 并把复核目录定为 `<MMDD>/<MMDD>smpl/mocap_ori_c3d/<sid>/`（行 78-80）；`analyze_smpl_reviews.py:788` 把 `Mocap` 写进输出 manifest。

### 1.4 协议层 URI 使用分布（144 行实测统计）

`protocol/manifests/session_manifest.jsonl` 144 行：

| 字段 | 形状（144/144 或注明） |
|---|---|
| `video_path` | `raw://rgb/<YYYYMMDD>/<S#>/<rec…>`（144/144，**是目录**） |
| `bvh_path` | `raw://bvh/<YYYYMMDD>/mocap_ori_bvh/<Sddddd>/<Sddddd>_Skeleton<N>.bvh`（144/144，N∈0..5，**是文件**） |
| `smpl_path` | `raw://smpl/<MMDD>/<MMDD>smpl/mocap_ori_c3d/<Sddddd>/motion_neutral_smpl.npz`（140/144；4 行为空 = `missing_smpl`，即 S12072/S12081/S14101/S6101） |
| `pressure_path` | `work://data_pipeline/pressure_washer/final_fake_marked/<YYYYMMDD>/<S#>/<rec…>`（144/144，**不是 raw://**） |
| 其它 | `camera=cam3`×144；`quality=ok`×140 + `missing_smpl`×4；`eligible_anysole=1`×140 |

- 日期行数：0804=53、0808=42、0810=36、0807=9；subject：S11=24、S5=22、S6=21、S14=19、S13=18、S12=14、S7=11、S8=9、S10=6（**无 S9**，为历史有意排除，已拍板维持，不是开放问题）。
- `raw_session_index.jsonl` 144 行，同样字段 + `pressure_uri` 也是 `work://`。
- `shared/facts/sessions/cam3/<date>/<S>/<sid>/session.json` 的 `source_files.{rgb,bvh,smpl}` 逐字复制上述 raw:// 字符串（例：`shared/facts/sessions/cam3/20260804/S5/S5011/session.json`）；`left_source/right_source` 是 work://。
- `model_inputs/*/artifact.json` 若干处同样记录 `raw://bvh|raw://smpl`（例：`model_inputs/MotionPRO/adapter_v1/cam3/20260810/S13/S13082/artifact.json:110,177-178`；`model_inputs/AnySole/adapter_v1/labels/{tactile_rel,f6_soft}/artifact.json:14` 记 `"raw://bvh"`）。
- **`raw://pressure` 的空转事实**：`build_manifest.py:117` 与 `:237` 只在 meta 缺 `pressure_uri` 时回退到 `raw://pressure/<date>/<subject>/<rec>`；当前 144 行全部是 work://，即 **raw/pressure 目前没有任何运行时消费者**。
- **`raw://calibration` 同理**：全仓（含 Baselines）没有任何代码解析 `raw://calibration`；校准实际走 `protocol/calibration/<date>.json` → 软链到 `UP/4_Dataset/<date>/<date>.json`（6 个日期；`AnysoleWorkspace/tool/adapters/mmvp_series/common/common.py:31`、`prepare_mmvp_observations.py:90`、`results_display/script/d_test6_floor.py:218`）。

### 1.5 断链敏感性最高的地方：52,612 个假事实层相对链接

`shared/facts/sessions/cam3/<date>/<S>/<sid>/rgb/000000.jpg` 的内容形如：

```
../../../../../../../../raw/rgb/20260804/S5/rec20260804_192218_方晟宇_S501_1/3/3_192218.435.jpg
```
共 **52,612** 个（`shared/facts` 下全部软链接，140 个 session × 平均 ≈376 帧 × cam3）。
要点：
- 这 8 级相对路径**必须**从 workspace 内穿透 `raw/rgb` 这个名字，且要求 `raw/rgb/<date>/<S#>/<rec>/3/<file>.jpg` 逐段存在；
- `model_inputs` 共 **157,209** 个软链接，其中 104,597 个相对链指向 `shared/frontends/*`，**52,612 个绝对链指向 `AnysoleWorkspace/shared/facts/.../rgb/*.jpg`（全部在 MotionPRO）**；
- 因此 **MotionPRO 与 PoseTransOpt 等对 raw 的依赖都是"间接"的**：raw/rgb 一断，52,612 个 shared/facts 链接全断，连锁上到 model_inputs。

### 1.6 对审计报告 §4 的一处更正（重要）

审计报告称「`PoseTransOpt/adapter_v1/color` 直连 `/data/lizhe/...` 绝对路径」。**本次实测不是**：

```
$ find model_inputs -type l -lname '/data/lizhe*' | wc -l
0
$ readlink model_inputs/PoseTransOpt/adapter_v1/20260810/S13/S13011/color/000004.jpg
../../../../../../../shared/facts/sessions/cam3/20260810/S13/S13011/rgb/000004.jpg
```
PoseTransOpt 四个日期共 50,406 个 color 链接全部是**相对链 → shared/facts**（生成器 `tool/adapters/mmvp_series/posetransopt/build_inputs.py:78` 用 `os.path.relpath`；`fpp/build_inputs.py:71` 同款实现）。
结论：约束①的"不改现状链接"在本方案下自然满足，但**不能据此认为 PoseTransOpt 与 raw 无关**——它经由 shared/facts 间接依赖 `raw/rgb`。任何"改名 raw/rgb"的方案都会连带打断它（见 §4.3）。

---

## 2. 消费方全清单（逐一 file:line）

### 2.1 gait 仓（workspace 内）——按"需要的路径形状"归类

| 消费点（file:line） | 需要什么形状 | 方案 B 是否保持 |
|---|---|---|
| `tool/workspace.py:31-37,52-80,111-164` | 5 个入口 + `resolve_uri` + `doctor` | **需改**（见 §4.5） |
| `tool/build_raw_index.py:37` | `raw://rgb` 下 `*/S*/rec*/meta.json`（单层 glob） | 保持（meta.json 须在农场内） |
| `tool/build_raw_index.py:52-54` | `recording.parts[-3]=date`；`raw://bvh/<date>/mocap_ori_bvh/<sid>/*.bvh` | 保持 |
| `tool/build_raw_index.py:57,68-70` | `raw://smpl` 下 `**/<sid>/motion_neutral_smpl.npz`（**递归 glob**） | 保持（农场内该层必须是实目录，§4.4-①） |
| `tool/build_manifest.py:117,237` | `raw://pressure` 下 `**/<rec>/pressure_left.csv` 回退 | 保持（同上，实目录） |
| `tool/build_shared.py:137-138,145,150,184` | `resolve_uri(video_path)` 后 `parts[-3],[-2]`、`recording/"3"`、`recording/"meta.json"` | 保持 |
| `tool/contact_labels.py:88` | `raw/bvh` 下 `*/mocap_ori_bvh/<sid>*`（单层） | 保持 |
| `tool/adapters/mmvp_series/fpp/build_metadata.py:139,148` | 同上 + 识别 `raw://` 前缀 | 保持 |
| `tool/adapters/mmvp_series/common/common.py:60-63` | `resolve_uri(video_path, must_exist=True).parts[-3],[-2]` | 保持（农场路径必须实际存在） |
| `…/fpp/build_inputs.py:58-75`、`…/fpp/export_v2t.py:60-75`、`…/keypoints/run_rtmpose.py:73-95`、`…/pressure_tookit/build_inputs.py:115` | 同上（4 处 session_parts 变体） | 保持 |
| `tool/adapters/AnySole/write_label_artifacts.py:51` | provenance 字符串 `"raw://bvh"` | 保持 |
| `tool/adapters/MotionPRO/adapter.py:481-489` | `session.meta.source_files.smpl`（raw://smpl URI）读原始 npz | 保持 |
| `tool/adapters/Step2Motion/build_gait.py:183-184,441` | `source_files.bvh`（raw://bvh URI）→ 解析真实 bvh | 保持 |
| `tool/check_baseline_readiness.py:38`、`tool/smoke_baseline_integration.py:48` | `resolve_uri(video_path, must_exist=True)` | 保持 |
| `tool/pressure_washer/pwlib/paths.py:24-26` | `PRESSUREWASHER_RAW_ROOT` 默认 **上游绝对路径**（不走 raw/） | 不受影响（但见 §4.6 可选整改） |
| `anysole/types.py:41-45` + `anysole/configs/v1.yaml:59-63` | `SMPL_ROOTS` = `UP/Mocap/{0804,0807,0808,0810}` 绝对路径，`**/<sid>/motion_neutral_smpl.npz` glob | 不受影响（直连上游，见 §5.2-③ 的冻结理由） |
| `anysole/data/workspace_adapter.py:117-129`、`smpl_io.py:160-173` | 先试 raw://smpl URI，失败再 glob SMPL_ROOTS | 保持（URI 优先命中农场） |
| `anysole/train.py:92-100` → `infer.py:350-356` | **smpl_roots 写进 checkpoint config 并优先读回** | 不受影响（方案 B 不动 Mocap 路径） |
| `Baselines/pressure_tookit/lib/utils/workspace.py:1-30` | 转发 `workspace.py:resolve_uri` | 只需 workspace.py 兼容 |
| `Baselines/VP-MoCap/PoseTransOpt/run_full_mmvp.py:56-60` | `resolve_uri(video_path, must_exist=True).parts[-3],[-2]` | 保持 |
| `Baselines/Step2Motion/src/process_underpressure.py:135` 等 | BVH/BVH 派生产物（来自 shared facts + URI 解析） | 保持 |
| `results_display/script/utils/cli_common.py:55`、`d_test6_floor.py:49` | `raw/bvh` 单层 glob；`3_Result/processed/rgb_human_masks` 绝对路径 | 保持 / 见 §4.6 可选项 |
| `Baselines/决策/01_D_数据协议.md:34,51`、`05_R1-R3_风险裁定.md:298` | 文档中引用的 raw/smpl、raw/bvh 路径示例 | 保持（农场同名同形） |

### 2.2 上游（lizhe）——"改上游"的爆炸半径

| 文件:行 | 依赖的上游形状 |
|---|---|
| `2_Code_preprocess/preprocessing_common.py:122` | `project_root/"1_Data"/*/mocap_ori_bvh/<session>` |
| `2_Code_preprocess/preprocessing_common.py:145,289,302` | `1_Data/<date>/mocap_ori_trc/*.trc`、逐日期遍历 `1_Data/[0-9]*`、`1_Data/*/<group>` |
| `2_Code_preprocess/mocap_camera_alignment/sync.py:93,175,185` | 在 `1_Data/<date>` 下递归找文件；**按目录名 `parts.index("1_Data")` 反解相对路径**（对"目录名/层级"最脆的一处） |
| `…/mocap_camera_alignment/marker_trc.py:60`、`spatial_transform.py:227` | `1_Data/<date>/mocap_ori_trc/<session>.trc` |
| `…/mocap_camera_alignment/mocap_csv.py:175` | `1_Data/<date>/mocap_csv/**/<session>/<session>_bvh.csv` |
| `2_Code_preprocess/1_single_camera_calibration.py:139`、`2_Multi_camera_calibration.py:318`、`build_camera_world_extrinsics.py:75,176` | `0_Calibration/<date>/{1_single,2_multi,3_alignment}` |
| `2_Code_preprocess/analyze_smpl_reviews.py:73,78-80,788` | `Mocap/*/*smpl/fail.txt` + 复核目录形状 |
| `2_Code_preprocess/human_segmentation/constants.py:4`、`marker_removal/cli.py:23`、`debug_marker_removal_overlay.py:20` | `DEFAULT_PROJECT_ROOT=/data/lizhe/projects/Tactile` |
| `2_Code_preprocess/tests/test_bvh_motion.py:11,17-20,31`、`tests/test_preprocessing_common.py:17,59-60` | 测试里写死的 `1_Data/<date>/mocap_ori_bvh/...` |
| `2_Code_preprocess/dataset_publish/ssh_transfer/*.sh`、`http_transfer/*.sh` | `SOURCE_ROOT=UP/4_Dataset`（发布下游，间接） |
| `MocapVideoAligner_0811/config.py:8-9`、`REMOTE_VNC_MANUAL.zh-CN.md:150` | 默认 Windows 路径；服务器侧由用户手选目录（低风险，但升级说明里有示例） |
| `3_Result/**`（7,874 行命中 `1_Data`，json/csv/md 等文本） | 派生结果里的 **provenance 绝对路径**（3_Result 下软链接 0 个，不是活链） |
| `.ai/DECISIONS.md:35-42` | 「Raw data under `0_Calibration/` and `1_Data/` should not be overwritten… Do not edit raw JPG/MP4/AVI/BVH/TAK/TRC/C3D/CMR/CSV unless explicitly requested」 |
| `.ai/PROJECT_INDEX.md:15,31-37` | `1_Data` 各层级的官方定义 |
| `2_Code_preprocess/.git` | 上游自带 git；改结构会与其历史/provenance 冲突 |

上游活跃度实测：`.ai/TASK_LOG.md`、`.ai/TODO.md`、`2_Code_preprocess/analyze_smpl_reviews.py`、`tmp/annotations_bbox_transfer_20260927*` 的 mtime 均在 2026-09-27/28。

---

## 3. 方案 A：重排上游结构

### 3.1 目标结构（设计草案）

```
/data/lizhe/projects/Tactile/
├── 0_Calibration/<YYYYMMDD>/{1_single,2_multi,3_alignment}     # 不动
├── 1_Data/<YYYYMMDD>/                                          # 只留采集
│   ├── <S#>/rec.../{1,2,3,4, *.mp4, meta.json, pressure_*.csv}
│   └── rec..._calibration_1/（0807）
├── 2_Mocap/<YYYYMMDD>/{bvh,trc,c3d,cmr}/…                      # 新增：从 1_Data 移出
├── 3_Smpl/<YYYYMMDD>/mocap_ori_c3d/<Sddddd>/…                  # 新增：Mocap 改名+补年份
├── 4_Dataset/…                                                 # 不动
└── AlignResult / AlignReviews_csv / 2_Code_preprocess / …      # 不动
```

### 3.2 迁移步骤（若执行）

1. 冻结窗口：与 lizhe 约定停采/停跑时段（采集端是 Windows 的 `Gait-Data-Collector_0811`，`settings.json:2` 写死 `E:/Data`，与本机树无耦合）。
2. 上游 `2_Code_preprocess/.git` 打 tag，记录 `git status`。
3. 在 `1_Data/<date>/mocap_ori_*` 原位建**兼容软链接**（老路径 → 新位置），再逐个 `mv`（同文件系统 rename，秒级；314 GB 不搬字节）。
4. 打补丁：§2.2 表中 12 处代码 + 4 处测试 + `sync.py:175` 的 `parts.index("1_Data")` 反解逻辑改为不依赖目录名。
5. 重跑上游自检：`2_Code_preprocess` 的测试、`mocap_camera_alignment` 抽 1 个日期 dry-run。
6. 更新 `.ai/PROJECT_INDEX.md` / `DECISIONS.md`，重生成受影响的 `3_Result` diagnostics（7874 行 provenance）。
7. 兼容层保留 N 周后清理。

### 3.3 影响与兼容性

- 我方（workspace）：**不受直接影响**（一切经 raw 软链），但仍需在 workspace 侧建一模一样的模态农场（因为 §1.4 的 URI 形状与 §1.5 的 52,612 个相对链接被冻结）。
- 上游：12 处代码 + 4 个测试 + provenance；`analyze_smpl_reviews.py` 若同时碰 `Mocap`（我方产物）也要同步。
- lizhe 侧治理：直接违反 `.ai/DECISIONS.md:35-42` 的字面（"raw data … should not be overwritten/edited"）与 `AGENTS.md` 的"绝不覆盖 0_Calibration/1_Data 原始数据，除非用户明确指定"——需要 lizhe 明确授权。

### 3.4 风险与结论（**判定不可行，本轮**）

1. **收益为 0**：workspace 侧无论如何都要建农场（§1.4/§1.5 冻结）。方案 A = 方案 B + 上游大改。A 唯一多出来的"整齐"发生在一个**不是我们的、且正在被改的**树上。
2. **上游在活跃使用中**（2026-09-27/28 仍有提交与脚本变更），且 `analyze_smpl_reviews.py` 这类脚本会**持续再跑**。
3. **有按目录名反解的代码**（`sync.py:175`），不是简单改常量能覆盖。
4. **provenance 被动失效**：3_Result 7,874 行、4_Dataset 构建记录里的绝对路径全部过期；上游 DECISIONS 明确要求"可追溯、不移动 canonical 源"。
5. **不可逆面大**：一次 `mv` 出错（例如把 rec 目录当 mocap 目录）会污染 314 GB 原始数据的可获得性；raw 是**没有第二份备份**的采集数据。
6. **跨用户协调成本**：需要 lizhe 明确授权 + 冻结窗口 + 双方验证。

**结论：本轮不执行方案 A。** 若未来要做，也应走 §3.2 的兼容层 + 上游方主导。

---

## 4. 方案 B：只动 workspace 侧的 raw 链接（推荐）

### 4.1 设计原则

1. **名字不变、URI 语义不变**：`raw/rgb`、`raw/pressure`、`raw/bvh`、`raw/smpl`、`raw/calibration` 五个入口保留原名（`raw://<entry>/<path>` 的解析规则完全不变）。
2. **内容按模态过滤**：每个入口不再是"整棵上游树"，而是**只含该模态**的链接农场。
3. **形状保持（shape-preserving）**：农场内部的相对路径与 manifest/session.json 里已冻结的 URI **逐段一致** → 零 manifest/splits 改动、零消费者代码改动、52,612 个 shared/facts 相对链零改动。
4. **版本安全的 glob 语义**：Python 3.13 起 `**` 不再穿越"中间层软链接目录"（本机 `python3 -V` = 3.13.12）。因此凡是会被 `**` 递归命中的层级，一律用**实目录**，软链接只做在**叶子**（文件级）或**不会被 `**` 穿越的层**。
5. **可重建**：农场由脚本幂等生成；`doctor` 能发现陈旧链接。

### 4.2 目标结构树

```
AnysoleWorkspace/raw/                      # 全部由工具生成（raw/ 在 .gitignore 内）
├── rgb/                                   # 模态视图：四路相机 JPG + meta.json
│   └── <YYYYMMDD>/<S#>/<rec…>/            # 实目录（保 shape）
│       ├── 1/ 2/ 3/ 4/                    # 软链 → UP/1_Data/<date>/<S#>/<rec>/<n>
│       └── meta.json                      # 软链 → 同名文件
├── pressure/                              # 模态视图：双脚压力 CSV
│   └── <YYYYMMDD>/<S#>/<rec…>/            # 实目录
│       ├── pressure_left.csv              # 软链（文件级）
│       └── pressure_right.csv             # 软链（文件级）
├── bvh/                                   # 模态视图：BVH（+ 可选 avi）
│   └── <YYYYMMDD>/mocap_ori_bvh/<Sddddd>/ # 实目录
│       ├── <Sddddd>_Skeleton<N>.bvh       # 软链（文件级）
│       └── (可选) <Sddddd>_Video_*.avi    # 软链（仅 0804 有）
├── smpl/                                  # 模态视图：SMPL npz（保 shape，含 0808 双嵌套）
│   └── <MMDD>/<MMDD>smpl/mocap_ori_c3d/<Sddddd>/   # 实目录
│       └── motion_neutral_smpl.npz        # 软链（文件级）
└── calibration/                           # 保持现状（无运行时消费者）
    └── -> UP/0_Calibration                #   或 6 个日期目录软链（可选）
```

> 注：`raw/rgb/<rec>/` 里**不再**有 `*.mp4` 和 `pressure_*.csv`，`raw/bvh/` 里**不再**有视频/压力/图像 —— 即"名字按模态、内容也是该模态"。
> 仍不"绝对纯"的地方只剩两处，方案里如实登记：① `raw/smpl/.../<sid>/` 旁还有 `conversion_manifest.json / motion_quality.json / smpl_validation.json`（如果只链 npz 则它们不出现，**推荐只链 npz**）；② `raw/bvh/.../<sid>/` 的 avi 需要显式决定是否链。

### 4.3 链接规则与数量估算（覆盖 4 个管线日期 = 313 rec）

| 入口 | 实目录 | 软链接 | 数量 |
|---|---|---|---|
| `rgb` | 4 日期 + 10 subject + 313 rec = **327** | 每 rec：4 个相机目录 + 1 个 meta.json = 5 | **1,565** |
| `pressure` | 4 + 10 + 313 = **327** | 每 rec：left/right 各 1 = 2 | **626** |
| `bvh` | 4 日期 + 1（`mocap_ori_bvh`）+ 313 sid = **318** | 每 sid：1 个 `.bvh`（+ 可选 83 个 avi）；+ 3 个 `calibration_Skeleton0.bvh` | **316**（含 avi 399） |
| `smpl` | 4 + 4 + 4 + 300 sid = **312** | 每 sid：`motion_neutral_smpl.npz` | **300** |
| `calibration` | — | 1（保持现状） | **1** |
| **合计** | **≈1,284 个实目录** | **≈2,808 个软链接**（不含 avi） | — |

若把 0419/0422 也纳入（132 个 rec、134 个 bvh、0 个 smpl、02 的 `Pressure_*.csv` 大写命名需单独规则），再 +约 1,000 链接。**推荐默认只覆盖 4 个管线日期**（与"管线只导入 S5–S14 4 个日期"的已定事实一致），另留 `--dates` 参数。

### 4.4 兼容性矩阵（为什么几乎零成本）

| 现有引用 | 方案 B 后状态 |
|---|---|
| manifest/splits 144 行 URI | **零改动**（形状保持，§4.1-③） |
| `raw_session_index.jsonl` | 零改动 |
| `shared/facts/.../session.json` 的 `source_files` | 零改动 |
| **52,612 个 shared/facts rgb 相对链** | **零改动、零重生成**（穿越 `raw/rgb` 到农场，再落到上游实体文件） |
| **52,612 个 MotionPRO color 绝对链** | 零改动 |
| **50,406 个 PoseTransOpt color 相对链** | 零改动（约束①满足；同时更正 §1.6 的现状） |
| `model_inputs/*/artifact.json` 里的 raw:// provenance 字符串 | 零改动（仍可解析到真实文件） |
| `build_raw_index.py` / `build_manifest.py` / `build_shared.py` / 各 adapter | 零改动（glob 形状与 `parts[-3],[-2]` 均保持） |
| `anysole/*`（SMPL_ROOTS 直连上游） | 零改动 |
| `pressure_washer`（默认直连上游） | 零改动 |
| `Baselines/*`（转发 resolve_uri） | 零改动 |
| **需要改的**：`tool/workspace.py` | `RAW_LINKS` → 农场规格 + `init/doctor` 语义（§4.5） |
| **需要改的**：`AnysoleWorkspace/README.md:39-43` 与 raw 说明段 | 文档（§4.5） |
| **新增**：农场生成器脚本 | 约 150–250 行（§4.5） |

**逐条核验过的"会被打破"的假设（均不成立）**：
- ① 没有任何消费者需要 `raw/rgb/<rec>` 里的 `.mp4`（全仓 `mp4` 命中都在 results_display 生成自产动画，不读上游源视频）；
- ② 没有任何消费者需要 `raw/bvh/<date>/` 下的压力/图像（现有 glob 全部带 `mocap_ori_bvh` 前缀）；
- ③ 没有任何消费者对 `raw/rgb` 做 `**` 递归（只有单层 `*/S*/rec*/meta.json`）；`**` 递归只出现在 `raw/smpl`（`build_raw_index.py:57`）与 `raw/pressure`（`build_manifest.py:119-120`），这两处按 §4.1-④ 做成实目录 + 叶子链接即无版本风险；
- ④ 没有任何消费者解析 `raw://calibration`（§1.4）；
- ⑤ `resolve_uri` 的 `must_exist=True` 调用（`common.py:61`、`run_full_mmvp.py:60`、`check_baseline_readiness.py:38`、`smoke_baseline_integration.py:48`、`build_shared.py:137-138`）要求农场路径真实存在——生成器保证。

### 4.5 改动清单（文件级）

| 文件 | 改动 | 说明 |
|---|---|---|
| `AnysoleWorkspace/tool/workspace.py` | `RAW_LINKS` 常量改为"农场规格"（入口名 + 上游根 + 过滤规则）；`init` 调生成器（幂等）；`doctor` 改为校验农场（对每个入口抽查形状 + 全量断链检查，保留现有的 rglob 断链扫描） | 约 30–50 行 |
| **新增** `AnysoleWorkspace/tool/relink_raw.py`（名字待定） | 生成器：`build / verify / --refresh / --dates`（幂等，先写临时清单再落链；输出 `reports/raw_farm_<timestamp>.json` 供审计） | 约 150–250 行 |
| `AnysoleWorkspace/README.md:29-43` 及"数据基座构建"段 | 更新 raw 说明：五入口 = 模态视图；附"上游路径 ↔ 农场路径"对照表；新增日期后的维护命令 | 约 30 行 |
| `AnysoleWorkspace/reports/` | 新增本次执行的 readlink 快照（回滚依据） | 生成 |

**不改**：manifest/splits、shared/facts 内任何文件、model_inputs、上游、anysole、Baselines、pressure_washer。

### 4.6 可选附加项（与本方案解耦，用户可单独拍板）

1. **登记 human-mask 外部依赖**：`tool/adapters/mmvp_series/cliff/run_cliff.py:64`、`tool/build_human_mask_frontend.py:27`、`results_display/script/d_test6_floor.py:49` 三处硬编码 `UP/3_Result/processed/rgb_human_masks`（实测 660,683 个文件）。可新增 `raw/human_masks -> UP/3_Result/processed/rgb_human_masks`（一个链接）并在 README 登记，让"workаspace 之外的数据依赖"都有名有姓。
2. **pressure_washer 改走 raw**：其默认根是上游绝对路径（`pwlib/paths.py:24-26`）。改默认值指向 `raw/pressure` 会让它与其他阶段口径一致——但注意 `analyze_pressure_csv_stats.py:105-111` 需要 `<root>/<date>/<S#>/<rec>/` 结构（农场满足），而 04xx 早期命名（大写 CSV）会不匹配（农场默认不含 04xx，等价于现状的"不处理"）。**建议暂不改**，只登记。
3. **`raw/calibration` 保持现状**（无消费者）；若为整齐可改为 6 个日期目录软链。

### 4.7 增量维护与风险

| 风险 | 等级 | 缓解 |
|---|---|---|
| 上游**改 rec 目录名**（已有先例：20260814 重命名 99 个目录）→ 农场 `rgb`/`pressure` 链接陈旧 | 中 | `doctor` 的断链扫描会直接报错；`relink --refresh` 幂等重建；纪律：**上游改名后跑一次** |
| 上游新增日期/subject | 低 | 新日期上线时跑 `relink --dates <新日期>`；manifest 侧本来也要走 build_raw_index |
| Python `**` 语义变化（3.13 起不穿越中间软链） | 已消除 | 设计上 `**` 命中层全部是实目录（§4.1-④） |
| 农场与上游短暂不一致（重建窗口内 52,612 链接断开） | 低 | 采用"先在 `raw/.farm.new` 全建好 → `os.rename` 原子替换目录"的原子交换；替换瞬间无断链窗口 |
| 有人手工往 `raw/` 里放东西 | 低 | `doctor` + README 声明 raw/ 为生成物 |
| 农场把 `raw/` 变成"派生目录"，被误认为可以写数据 | 低 | `resolve_uri(for_write=True)` 已硬拒 `raw://`（`workspace.py:78-79`），不变 |

### 4.8 工作量估算

- 生成器 + workspace.py + README：**0.5–1 人日**（含 dry-run 报告）；
- 执行与验收：**0.5 人日**（建链秒级；验收为 5 项冒烟，见 §7）；
- 后续每次新日期：**分钟级**。
- **无数据搬移、无 manifest 重建、无 shared/facts 重建。**

---

## 5. 方案 C：混合（上游小改 + workspace 整理）

逐条核验"看起来安全的小改"：

| 候选小改 | 实测代价 | 判定 |
|---|---|---|
| ① `Mocap/0804 → Mocap/20260804` 加年份别名（软链） | `analyze_smpl_reviews.py:73` 的 `Mocap/*/*smpl/fail.txt` 会**双份命中**（同一 fail.txt 经别名再发现一次），该脚本"非空输出目录 fail-closed"会直接失败；且 `raw://smpl/0804/...` URI 被冻结，别名不解决任何问题 | 不做 |
| ② 消掉 `0808/0808smpl/mocap_ori_c3d` 双嵌套 | manifest `smpl_path`（冻结）+ shared/facts `session.json`（冻结）都写着旧形状；消嵌套＝要么改这些文件，要么再建一层兼容链（净零收益） | 不做 |
| ③ `Mocap` 日期目录改名去掉无年份 | anysole `SMPL_ROOTS`（`types.py:41-45`、`v1.yaml:59-63`）**逐字**引用这 4 个路径；`train.py:100` 把 smpl_roots 写进 checkpoint，`infer.py:350-356` 优先读回 → **50 个已有 checkpoint 全部指向死路径**（仅当 raw://smpl URI 也失效时才触发，但这是不必要的风险） | 不做 |
| ④ 0807 日期级校准采集目录挪进 S8 | 上游 calibration pipeline 目前只按 `0_Calibration/<date>/{1_single,2_multi,3_alignment}` 取值（§2.2），该 rec 目录**没有**已知消费者；挪动是"无收益的改动" | 不做 |
| ⑤ 在上游 `Mocap/` 根加 README/索引 | 不影响 `*/*smpl/fail.txt` glob；纯增量文档 | 可做（低优先） |

**结论**：C 的每一项要么破坏冻结物、要么净零收益。**本轮不做上游改动**；未来若要与 lizhe 推进上游治理，以 §3.2 的兼容层方案为准。

---

## 6. 三方案对比

| 维度 | A 重排上游 | **B workspace 农场（推荐）** | C 上游小改 |
|---|---|---|---|
| 上游改动 | 大（24 个目录 mv + 16 处代码/测试） | **无** | 小（但每项都有代价） |
| manifest/splits | 不变 | **不变** | 不变（但依赖项被冻结，小改失去意义） |
| shared/facts（52,612 链） | 需重建 | **不动** | 不动 |
| 消费者代码改动 | 上游 12+4 处 | **0** | 0（上游 1 处会崩） |
| 新增/改动文件 | 上游十余个 + 我方 0 | 我方 3 个（1 新增 2 修改） | 上游 1–2 个 |
| 主要风险 | 污染 314 GB 无备份原始数据、破坏 lizhe 在跑管线、provenance 失效 | 上游改名导致农场陈旧（doctor 可检、relink 可修） | 破坏 `analyze_smpl_reviews` / checkpoint 路径 |
| 收益 | ≈0（农场无论如何要建） | 名字与内容一致；新读者不再被"三别名"误导 | 低 |
| 回滚 | 难（24 目录 + provenance） | **易**（删农场、按快照重建 5 个旧链接） | 中 |
| 工作量 | 数人日 + 跨用户协调 | **1–1.5 人日** | 0.5 人日（但收益负） |

---

## 7. 推荐与执行顺序（拍板后）

**推荐方案 B。核心理由**：raw 层的"乱"有**两个来源**——上游树的混装与我方三别名。方案 B 只解决我方那半，但因为 manifest URI 与 `shared/facts` 相对链**已经被冻结**，我方那半正是**唯一能在零外部代价下变整齐的部分**；上游那半无论做与不做，workspace 侧都得有一个形状保持的模态视图。B 用"实目录 + 叶子链接"的农场把 glob/`**` 语义风险一并消除。

**分步执行顺序**（每步都可独立验收/回滚）：

1. **生成器 dry-run**：`relink_raw.py --dry-run --dates 20260804,20260807,20260808,20260810` 输出：链接清单、目标存在性预检、与 manifest 144 行 URI 的逐条对表（必须 144/144 命中真实文件）→ 报告落 `reports/`。
2. **改 `tool/workspace.py`**（农场规格 + `init/doctor`），保留对 5 个入口名的兼容解析；`doctor` 增加"农场形状抽样 + 全量断链"检查。
3. **快照回滚依据**：把当前 5 个链接的 `readlink` 与 `ls -la` 结果写入 `reports/raw_relink_rollback_<ts>.txt`。
4. **构建 `raw/.farm.new/`**（全套链接），校验：
   - (a) 5 项冒烟：`resolve_uri` 对 manifest 每类 URI 抽样 `must_exist=True` 通过；
   - (b) 52,612 个 shared/facts 链接 `-xtype l` 计数 = 0；
   - (c) `build_raw_index.py --dry-run`（或输出到 `work/` 临时路径）与现有 `raw_session_index.jsonl` 逐字段 diff = 0。
5. **原子替换**：`os.rename(raw/.farm.new/<name>, raw/<name>)`（先删旧别名链接，rename 原子完成）；替换后立刻复跑 (a)(b)(c)。
6. **收尾**：改 README；跑 `workspace.py doctor`；`build_shared.py --facts-only` **不重跑**（不需要），改为抽样加载 3 个 session 验证 `frames.npz/pressure_48.npz` 链路与 rgb 帧可读。
7. **回滚预案**：删农场 → 按步骤 3 快照重建 5 个旧链接 → `doctor` 通过。全程不触碰 manifest/shared/model_inputs，回滚零数据损失。

---

## 8. 需要用户拍板的开放点

1. **农场覆盖范围**：默认 4 个管线日期（313 rec）；是否纳入 0419/0422（+132 rec，且 04xx 压力命名是大写口径，需专门规则）。
2. **`raw/rgb` 的粒度**：推荐每 rec 4 个相机目录软链 + meta.json 软链；若希望"零目录链接"，可退化为只链 cam3 的 52,612 个文件（数量大 30 倍，收益仅"更纯"）。
3. **`raw/bvh` 是否连带 avi**（仅 0804 有 83 个 `*_Video_*.avi`）——推荐**不带**（管线不读，且 avi 属视频模态）。
4. **是否顺带登记 `raw/human_masks`**（§4.6-①）——三处硬编码绝对路径，登记后可由 doctor 监控。
5. **维护纪律确认**：上游 rec 改名/新增日期后，"跑一次 relink + doctor"是否被接受为流程的一部分（这是方案 B 唯一的持续成本）。

另：`raw/smpl` 农场只链 `motion_neutral_smpl.npz`（不链 `conversion_manifest.json / motion_quality.json / smpl_validation.json`）——如需保留这些 QC 文件可一并链（+900 链接），默认不链。

---

## 附录 A：关键证据文件:行号索引

- 现状登记：`AnysoleWorkspace/tool/workspace.py:31-37`、`AnysoleWorkspace/README.md:39-43`
- URI 形状：`protocol/manifests/session_manifest.jsonl`（144 行）、`protocol/manifests/raw_session_index.jsonl`（144 行）、`shared/facts/sessions/cam3/20260804/S5/S5011/session.json`
- 相对链拓扑：`shared/facts/sessions/cam3/*/*/*/rgb/*.jpg`（52,612 个，8 级相对路径）、`model_inputs/MotionPRO/adapter_v1/*/*/*/*/color/*.jpg`（52,612 个绝对链）、`model_inputs/PoseTransOpt/adapter_v1/*/*/*/*/color/*.jpg`（50,406 个相对链）
- workspace 消费点：`tool/build_raw_index.py:37,52-57,68-70`、`tool/build_manifest.py:117,237`、`tool/build_shared.py:137-138,145,150,184`、`tool/contact_labels.py:88`、`tool/adapters/mmvp_series/fpp/build_metadata.py:139,148`、`tool/adapters/mmvp_series/common/common.py:60-63`、`tool/adapters/MotionPRO/adapter.py:481-489`、`tool/adapters/Step2Motion/build_gait.py:183-184,441`、`tool/pressure_washer/pwlib/paths.py:24-26`、`tool/check_baseline_readiness.py:38`、`tool/smoke_baseline_integration.py:48`
- 上游消费点：`UP/2_Code_preprocess/preprocessing_common.py:122,145,289,302`、`UP/2_Code_preprocess/mocap_camera_alignment/{sync.py:93,175,185, marker_trc.py:60, spatial_transform.py:227, mocap_csv.py:175}`、`UP/2_Code_preprocess/analyze_smpl_reviews.py:73,78-80,788`、`UP/2_Code_preprocess/1_single_camera_calibration.py:139`、`UP/2_Code_preprocess/build_camera_world_extrinsics.py:75,176`、`UP/2_Code_preprocess/tests/test_bvh_motion.py:11,17-20,31`
- 上游治理：`UP/.ai/DECISIONS.md:35-42`、`UP/.ai/PROJECT_INDEX.md:15,31-37`、`UP/AGENTS.md`（原始数据不覆盖）
- 冻结物（不可改）：`protocol/manifests/*`、`protocol/splits/default/splits.csv`、`shared/facts/**`、`model_inputs/**`

## 附录 B：上游结构变体清单（"改上游"要同时处理 6 类变体）

1. 04xx 期采集：无 meta.json、压力 CSV 大写 `Pressure_{left,right}_*.csv`、bvh 自带 avi、有 `mocap_csv`；
2. 08xx 期采集：meta.json + 小写 pressure CSV、0807 起无 avi、无 mocap_csv；
3. 0807 日期级 calibration 采集目录（挂在日期级而非 S8 级）；
4. bvh 根例外文件：0419 `Take_calication.trc`、0422 `Calibration1_Skeleton0.bvh`、0807/0808/0810 `calibration_Skeleton0.bvh`、0804 无；
5. c3d/trc/cmr 平铺 vs bvh 目录化，且数量比 bvh 多 1（校准 session）；
6. `Mocap/`：无年份日期名（0804/0807/0808/0810）、缺 0419/0422、0808 双重嵌套、每日期 c3d 副本与 `<MMDD>smpl/` 并列。

---

## 用户裁定记录（2026-10-02）

方案 B 获裁定，五个决策点裁定如下：

1. **农场覆盖范围**：只覆盖管线用的 4 个日期（20260804/07/08/10）的 rec，不含 0419/0422。
2. **raw/rgb 粒度**：生成器增加 `--cam` 导入开关，指定相机号则只导入对应相机的目录/文件（默认 cam3）。
3. **raw/bvh**：带上 0804 目录里的 avi 文件。
4. **raw/human_masks**：待裁定（来源与消费端已查清：上游 3_Result/processed/rgb_human_masks 为唯一来源；
   消费端 = AnySole build_hrnet_bbox / pressure_toolkit depth mask / CLIFF bbox 重算；代码硬编码仅
   tool/build_human_mask_frontend.py:27 与 tool/adapters/mmvp_series/cliff/run_cliff.py:64 两处）。
5. **维护纪律**：接受。上游改名/新增日期后跑一次 relink（幂等重建），workspace.py doctor 断链扫描作为检出手段。

---

## 执行记录（2026-10-02，方案 B + 全部裁定已落地）

- 生成器：`tool/relink_raw.py`（build/verify，--dates/--cam/--dry-run，原子替换，报告落 reports/raw_farm_*.json）。
- `tool/workspace.py`：`RAW_LINKS` 改为 `RAW_SYMLINKS`（calibration、human_masks）+ `RAW_FARM_ENTRIES`；`init` 建软链接后调生成器；`doctor` 增加农场条目检查 + 144 行 manifest URI 全量解析检查。
- 农场实测规模：rgb=626（313 rec × meta.json + cam3 目录）、pressure=626、bvh=399（313 bvh + 83 avi + 3 calibration）、smpl=300；raw 下断链 0。
- raw/human_masks 已登记（软链接 + workspace.py + 三处硬编码改 `resolve_uri("raw://human_masks")`：`tool/build_human_mask_frontend.py`、`tool/adapters/mmvp_series/cliff/run_cliff.py`、`results_display/script/d_test6_floor.py`）。
- 验收全过：`workspace.py doctor`（428/428 manifest URI）；shared/facts 52,612 链断链 0；model_inputs 157,209 链断链 0；PoseTransOpt color → shared/facts → raw/rgb → 上游全链路可读；`build_raw_index.py` 重生成 144 行与冻结 `raw_session_index.jsonl` 逐字段一致。
- 执行中发现并修复的生成器缺陷：① 相对链目标误按 staging 路径计算（rename 后多一层 `..` 导致断链），改为按最终位置 `raw/<entry>/<rel_link>` 计算；② `.farm.old` 软链接清理误用 rmtree，改为 unlink；③ manifest 校验时机改为 swap 后（swap 前仅作 preflight）。
- 回滚依据：`reports/raw_relink_rollback_20261002.txt`（旧 5 链接快照）。
- 裁定第 4 点闭合：raw/human_masks **已登记**。
