# AnysoleWorkspace 组织结构审计（2026-10-01）

范围：`AnysoleWorkspace/` 全部目录（raw / protocol / shared / model_inputs / work / assets / reports / tool）。
性质：只检查、只归档，未做任何修改。所有结论均有实测证据（路径、链接目标、行数）。

## 总览

AnysoleWorkspace 自身的 canonical 分层（raw → protocol/manifests → shared/facts → model_inputs/<model> → work）是规整的。
用户看到的"混乱"来自两层：

1. **上游采集树本身的乱**（`/data/lizhe/projects/Tactile/1_Data`，只读透传，workspace 无权也不应改）；
2. **raw 三个模态别名指向同一棵树**（workspace 自己的设计，见 §1）。

## 1. raw/rgb、raw/pressure、raw/bvh 是同一目录的三个别名（"完整数据软链接"的真相）

`tool/workspace.py` `RAW_LINKS` 与 README 明文登记：

```
raw/rgb       -> /data/lizhe/projects/Tactile/1_Data
raw/pressure  -> /data/lizhe/projects/Tactile/1_Data   ← 同一个
raw/bvh       -> /data/lizhe/projects/Tactile/1_Data   ← 同一个
raw/smpl      -> /data/lizhe/projects/Tactile/Mocap
raw/calibration -> /data/lizhe/projects/Tactile/0_Calibration
```

后果：`raw/bvh/20260807/` 下能看到 pressure_left.csv、4 路 mp4/jpg、meta.json；`raw/pressure/<date>/S8/` 下能看到全部模态。
manifest 里的 `raw://rgb`、`raw://bvh` 只是同一棵树的三个入口。**名字按模态、内容全量** —— 这是设计，但任何"按文件夹拷贝 bvh"的操作都会搬走全部数据（含视频）。

## 2. 上游 1_Data 层级混乱（透传可见，不可改，建议文档化）

- 日期目录混装三类：`S#/rec...` 采集目录、`mocap_ori_{bvh,c3d,trc,cmr}` 四个导出目录、calibration 采集目录（仅 0807 有 `rec20260807_165200_calibration_1`，且挂在日期级而非 S8 级）。
- 四种 mocap 导出口径不一：`mocap_ori_bvh` 每 capture 一个子目录（`S8011/S8011_Skeleton1.bvh`），c3d/trc/cmr 平铺文件。
- `mocap_csv` 仅 0419/0422 两个早期日期有，0804 起消失。
- `Mocap/`（raw/smpl）日期目录命名 `0804/0807/0808/0810`（无年份、缺 0419/0422），且内部 `0808/0808smpl/mocap_ori_c3d/` 双重嵌套（manifest 的 smpl_path 原样记录了该嵌套）。
- 校准数据有两个上游位置：`0_Calibration`（raw/calibration 软链接，6 日期）与 `4_Dataset/<date>/<date>.json`（protocol/calibration/*.json 软链接目标，6 日期）。

## 3. 管线只覆盖 6 个日期中的 4 个（S5–S14）

- 上游 1_Data 有 6 日期、S1–S14；`session_manifest.jsonl` 144 行只含 S5–S14、4 日期（0804/07/08/10）。
- `shared/facts/sessions/cam3` 同样只有 4 日期。
- S1–S4（0419/0422）整体未进入管线。README 无任何说明 —— 需确认是有意排除（早期采集协议不同？）还是遗漏，结论应写进 README。

## 4. 适配器链接策略不一致

| 适配器 | rgb/color 链接目标 | 状态 |
|---|---|---|
| pressure_toolkit/v1/images | `shared/facts/sessions/cam3/...`（相对链） | canonical |
| FPP-Net/adapter_v1 | `shared/facts/sessions/cam3/...`（相对链） | canonical |
| PoseTransOpt/adapter_v1/color | `shared/facts/sessions/cam3/...`（相对链） | canonical（**2026-10-01 更正**：早前误记为直连 /data/lizhe；实测 101,736 个链接中直连 /data/lizhe 的为 0） |

- **更正记录（2026-10-01）**：本报告初版称 PoseTransOpt color 链接"绝对路径直连上游 raw 绕过 canonical 层"不成立。实测其字面目标为相对链 `../../../../../../../shared/facts/sessions/cam3/.../rgb/*.jpg`（经 shared/facts → raw/rgb 间接依赖上游），与 FPP-Net/pressure_toolkit 口径一致。初版结论源于 `readlink -f` 解析链路末端而非字面目标。
- PoseTransOpt 的 `pred_contact_smpl/*.npy` → `work/VP-MoCap/v1/fpp_predictions/...`（相对链）：一个模型的 work 输出直接充当另一个模型的 model_input（数据流方向本身是 VP-MoCap→PoseTransOpt 的管线设计，但链接横跨 work/ 与 model_inputs/，上游产物若清理即断）。
- FPP-Net 覆盖疑似未完成：仅 1 个 adapter_manifest（S5011）、color 链接 652 个，对比 PoseTransOpt 140 个 join_manifest、约 10.2 万个链接。是"有意子集"还是"构建中断"待确认。
- `shared/frontends/` 版本目录命名不一：cliff_hr48/v1、depthpro/v1、rtmpose_halpe26/v1，唯 human_masks 用 `sam31`。

## 5. 协议层数字与文档漂移

- `session_manifest.jsonl` 144 行 = `raw_session_index.jsonl` 144 行；
- `splits/default/splits.csv`：train 104 + val 36 + test 36（val 与 test 同批，README 声明"默认 val=test"为设计）；
- AnySole labels 每类 140 npz：manifest 中有 4 个 capture 无标签（S12072、S12081、S14101、S6101）；
- README split 示例命令 `--exclude S4,S5 --test S13,S14` 与实际不符：实际 manifest **含 S5**，被排除的是 S1–S4。文档漂移。

## 6. git 与可移植性

- 根 `.gitignore` 忽略 `raw/ shared/ model_inputs/ work/`（数据树不入库，合理）；AnysoleWorkspace 内无独立 .gitignore，规则挂在仓库根。
- 但被跟踪的 `protocol/calibration/*.json` 是 6 个**指向仓外绝对路径**的软链接（/data/lizhe/...）——换机或他人克隆必断。建议改相对路径或入库实体 json。
- 当前工作区有未提交修改：README.md、tool/adapters/mmvp_series/posetransopt/export_v2m.py、tool/export_baseline_motion.py（本次审计未动它们）。

## 7. 存储观察（不构成问题，仅记录）

- `shared/` 16G：facts 的 rgb 是软链接（不占空间），实文件为 pressure_48.npz、frames.npz、frontends 产物、mmvp_31x11 insole；
- `model_inputs/PoseTransOpt` 2.3G，其中约 1.9G 是 140 个 `template_scene_rgbd.npy`（每个 13.4MB；抽查 md5 互不相同，是逐 capture 的场景模板，内容合法）；
- `work/data_pipeline/pressure_washer/` 保留 5 个中间阶段（encoded / reconstructed / stats / AlignReviews_csv / final_fake_marked）约 800M；
- 全 workspace 断链软链接：0 个。

## 建议（仅记录，未执行）

1. README 增加"上游树导航"一节，把 §2 的混乱用一张示意图说明（哪些在日期级、哪些在 S# 级、bvh 子目录 vs c3d 平铺、calibration 采集目录位置）。
2. PoseTransOpt 的 color 链接迁到 shared/facts（FPP-Net 已完成同款迁移）。
3. 确认 S1–S4 排除原因并写入 README；README split 示例命令与真实命令对齐。
4. human_masks/sam31 → v1 命名统一。
5. protocol/calibration 软链接改相对路径或入库实体文件。
6. 确认 FPP-Net 适配器是子集还是未完成。

## 用户决策记录（2026-10-01）

1. **§4 适配器链接策略不一致：不处理**（含 pred_contact_smpl 跨模型链接、FPP-Net 覆盖子集、human_masks/sam31 命名）。注：§4 中"PoseTransOpt 直连上游"一项经核实不成立（见 §4 更正记录），此决策点的实际剩余项仅上述三项。
2. **S1–S4 排除：确认为有意**。S1–S4 是不完善数据集，导入是历史遗留问题；今后只导入 S5–S14 的 4 个日期记录。
   - ⚠️ S9 缺失已查证（2026-10-01 补充）：用户记忆"S9 全 D 级故排除"**定性成立、字母有出入**。触觉数据筛选表 = `work/data_pipeline/pressure_washer/stats/pressure_stats_20260814_231054/overall/missing_pressure_objects.csv`（quality_class 列 A/B/C/D，331 行，覆盖 S5–S14）。实测 S9 为 **33/33 条全部 C 级**（非 D），原因统一为 `single-foot missing: L`——左脚压力数据每个 session 全部缺失。S9 是唯一 100% 单脚缺失的 subject（对照：S8 有 12 条 C、S10 有 19 条 C，但均与 A/B 混杂且已在 manifest 内；S7 有 16 条 D——多为上游 rec 目录缺失——仍在 manifest 内）。因此排除 S9 是"整组数据不完整"级决定，非逐条等级过滤。附带事实：S9 的 pressure final_fake_marked 产物其实完整存在（33 条），只是未进 manifest；0808 mocap fail.txt 中 S9 另有 6 条解算失败（S9043/S9052/S9053/S9093/S9102/S9111）。**结论：S9 排除为历史有意决定，管线维持 9 个 subject 现状，不补 S9。**
3. **README split 示例 `--exclude S4,S5` 漂移：不改文档**。已实测验证 `--exclude` 参数真实生效（`tool/build_manifest.py` `write_splits` 第 176 行先按 excluded 过滤 eligible 再分配 train/val/test，第 180-181 行对请求的 test/val subject 不存在时给出 warning）。当前 splits.csv 保持不动。
4. **git 可移植性：不管**。raw/ 下 5 个入口以软链接导入上游只读源是登记在 workspace.py/README 的预期设计；git 相关问题（protocol/calibration 仓外绝对路径软链接）不处理。
5. **raw 三别名指向同一棵树（§1）：要处理**。用户有 `/data/lizhe/projects/Tactile` 读写权限。两种方向：A 重排上游结构后链过来；B 不动上游、只在 workspace 侧把链接理清楚。**已执行（2026-10-02）**：方案 B 模态链接农场 + raw/human_masks 登记，详见 `reports/raw_relink_plan_20261001.md`（方案 + 裁定 + 执行记录）。执行结果：raw/{rgb,pressure,bvh,smpl} 为形状保持型模态农场（626/626/399/300 链接），raw/{calibration,human_masks} 为外部只读软链接；`workspace.py doctor` 428/428 manifest URI、shared/facts 与 model_inputs 断链 0、`build_raw_index` 重生成与冻结索引逐字段一致。
6. **触觉质量标注入档（2026-10-01 已执行）**：按用户指示，将等级标注写入 workspace 协议层并作为排除依据说明：
   - `session_manifest.{csv,jsonl}` 追加两列 `pressure_quality_class`（A/B/C/D）与 `pressure_quality_reason`（源自 `pressure_stats_20260814_231054/overall/missing_pressure_objects.csv`，144/144 全部 join 成功）；
   - `protocol/schemas/session_manifest.schema.json` 同步登记两字段；
   - 新增 `protocol/splits/default/README.md`：等级定义（A=双脚完整+零 zero-frame 秒+最大间隔<0.3s；B=有 zero-frame 秒或间隔≥0.3s；C=单脚完全缺失；D=上游目录缺失）、10 个 subject 分布表、排除依据（manifest 144=127A+17B 零 C/D；S9 33/33 全 C 整组排除；S1–S4 不完善数据集排除；A/B 内部还有 mocap 解算失败等门槛）；
   - `README.md` split 段落指向新说明；
   - splits.csv 未动；干跑验证 `build_manifest.py split --test S13,S14` 输出与冻结文件逐字节一致（train=104/val=36/test=36）。
