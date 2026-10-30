# 视觉基线单元清单（unit manifest）——受跟踪指针

- 路径：`AnysoleWorkspace/work/visual_baselines/reference_results/3_Result/SAM3DB_baseline/.run/manifest.jsonl`（432MB，gitignore 的 work/ 区，本指针文件为受跟踪的完整性凭据）
- SHA256：见同目录 `visual_baselines_unit_manifest.sha256`
- 来源：传输包分卷 `Vision_Baseline/_transfer/sam3db_records-0001.tar.zst` 成员 `reference_results/3_Result/SAM3DB_baseline/.run/manifest.jsonl`（三基线共用主清单）
- 保留约定（决策 11）：清理 Vision_Baseline 时**保留该分卷**——副本已存 `AnysoleWorkspace/work/visual_baselines/_archive/sam3db_records-0001.tar.zst`（SHA256 `102d10002df949957ceb871a5bef159fe9fab8e2fe435be6becc28cc1f4f4933`，源在 lizhe 属主目录无法移动故复制）；作为单元定义的唯一恢复源；预测（result.pkl/npz）可由本地生成管线重建，清单不可由本地数据完全重建
- 消费方：`adapters/visual_baselines/generate.py` 与 `export_*`（经 `common.load_unit`）
