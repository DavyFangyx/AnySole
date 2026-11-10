"""E9 主评估切到四折 woS13：把旧 S13/S14 表降到 detail/eval_s13s14。

四折文件由 e9_crossfit_split.py 写进 V7_Cv5 / V8_Cv5；本脚本只搬旧的
7 会话 S13/S14 文件，避免和四折主表混在一起。
"""
import os
import shutil

ROOT = "/data/fangyuxuan/projects/gait"
ROOTS = [
    (ROOT + "/results_display/Etest_crossfit/E9_TrajectorySceneOcc", "V7_Cv5"),
    (ROOT + "/results_display/Etest_crossfit_v8/E9_TrajectorySceneOcc", "V8_Cv5"),
]
SEVEN = [
    "Main_Table1_E9_SceneOcc_OccAte.csv",
    "Main_Table2_E9_SceneOcc_FullAte.csv",
    "Main_Fig1_E9_SceneOcc_Teaser_action03.png",
    "Main_Fig2_E9_SceneOcc_Teaser_action08.png",
    "Main_Fig3_E9_SceneOcc_Teaser_action09.png",
    "Main_Fig4_E9_SceneOcc_OccAte.png",
    "Full_Table1_E9_SceneOcc_PerSession.csv",
    "Full_Table2_E9_SceneOcc_CalibAblation.csv",
    "Full_Table3_E9_SceneOcc_ModalityDecomp.csv",
]
ALLBASES = [
    "Main_Table1_E9_SceneOcc_OccAte_AllBases.csv",
    "Main_Table2_E9_SceneOcc_FullAte_AllBases.csv",
    "Main_Fig4_E9_SceneOcc_OccAte_AllBases.png",
]


def move(src, dst):
    if not os.path.exists(src):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)


def main():
    for root, base in ROOTS:
        leg = os.path.join(root, "detail", "eval_s13s14")
        os.makedirs(leg, exist_ok=True)
        b = os.path.join(root, base)
        for name in SEVEN:
            move(os.path.join(b, name), os.path.join(leg, name))
        for d in ("supplementary", "detail"):
            move(os.path.join(b, d), os.path.join(leg, d))
        ab = os.path.join(leg, "AllBases")
        for name in ALLBASES:
            move(os.path.join(root, name), os.path.join(ab, name))
        move(os.path.join(root, "detail", "E9_双基座说明.md"),
             os.path.join(leg, "E9_双基座说明.md"))
        # 旧的 per-base crossfit 副本：挪到 detail/legacy_crossfit
        oldcf = os.path.join(root, "crossfit")
        if os.path.isdir(oldcf):
            move(oldcf, os.path.join(root, "detail", "legacy_crossfit"))
        print("organized", root)
    # V7 crossbase 里的旧 7 会话 AllBases/说明 -> legacy_s13s14
    cb = ROOT + "/results_display/Etest_crossfit/E9_TrajectorySceneOcc/crossbase"
    legacy = os.path.join(cb, "legacy_s13s14")
    for name in ALLBASES + ["E9_双基座说明.md"]:
        move(os.path.join(cb, name), os.path.join(legacy, name))
    print("organized crossbase")


if __name__ == "__main__":
    main()
