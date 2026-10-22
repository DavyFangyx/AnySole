"""CoP/相位探针 raw108 → raw108+offset 对比助手（2026-10-20，offset 修正轴审计）。

读取三个 JSON：
  ref     = audit_20261019/cop_phase/probe_v5_phase_cop_readout_raw108.json
            （未修正；载体 V3_Ev1_f6_soft ckpt，cuda:3，10-19）
  ctrl    = 本审计同载体同设备（V3_Bv1_denoised ckpt，cpu）未修正对照
  offset  = 本审计同载体同设备（V3_Bv1_denoised ckpt，cpu）修正轴
输出：
  - 表1：results mse_test（4 通道 × 3 读出 × 3 版本）+ const/lam/n_valid；
  - 差表：offset−ref、offset−ctrl（results mse + verdict 三元组）；
  - 设备漂移参考：ctrl−ref 的 raw108 读出（特征与 GT 均不依赖 ckpt/设备，
    理论 = 0/算术漂移级）与 v_tok 读出（ckpt 不同 → 权重效应，非漂移）；
  - 显著性判定对照（CI 是否含 0；T 可读 = const−raw108，V 更强 =
    raw108−v_tok）——raw108 读出与 T 可读判据不含编码器，ref↔offset
    直接可比；v_tok/vt 判据跨 ckpt 不可比，用 ctrl↔offset 比。
只读，不修改任何输入 JSON。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CHANNELS = ("phase_L", "phase_R", "cop_L", "cop_R")
READOUTS = ("raw108", "v_tok", "vt")


def load(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def fmt(x: float, nd: int = 6) -> str:
    return ("%." + str(nd) + "g") % x


def sig(ci) -> str:
    """CI 是否不含 0。"""
    return "SIG" if (ci[1] > 0 or ci[2] < 0) else "ns"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True,
                    help="10-19 raw108 JSON（未修正，V3_Ev1，GPU）")
    ap.add_argument("--ctrl", type=Path, required=True,
                    help="本审计未修正对照（V3_Bv1，CPU）")
    ap.add_argument("--offset", type=Path, required=True,
                    help="本审计修正轴（V3_Bv1，CPU）")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "compare_offset_summary.json")
    args = ap.parse_args()

    d = {k: load(p) for k, p in (("ref", args.ref), ("ctrl", args.ctrl),
                                 ("offset", args.offset))}
    for k, v in d.items():
        print("%-7s provenance=%s" % (k, v.get("provenance")))

    print()
    print("== 表1：results mse_test（通道×读出×版本）")
    print("%-9s %-7s | %12s %12s %12s | n_valid(ref/offset)  const" % (
        "channel", "readout", "ref", "ctrl", "offset"))
    res_diff_or = {}
    res_diff_oc = {}
    res_diff_cr = {}
    n_valid_all = {}
    for c in CHANNELS:
        for r in READOUTS:
            rows = [d[v]["results"][c + "|" + r] for v in ("ref", "ctrl", "offset")]
            assert all(x["status"] == "ok" for x in rows), (c, r)
            n_valid_all[c + "|" + r] = (rows[0]["n_valid_train"],
                                        rows[0]["n_valid_test"],
                                        rows[2]["n_valid_train"],
                                        rows[2]["n_valid_test"])
            res_diff_or[c + "|" + r] = rows[2]["mse_test"] - rows[0]["mse_test"]
            res_diff_oc[c + "|" + r] = rows[2]["mse_test"] - rows[1]["mse_test"]
            res_diff_cr[c + "|" + r] = rows[1]["mse_test"] - rows[0]["mse_test"]
            print("%-9s %-7s | %12s %12s %12s | %5d/%-5d→%5d/%-5d  %s" % (
                c, r, fmt(rows[0]["mse_test"]), fmt(rows[1]["mse_test"]),
                fmt(rows[2]["mse_test"]),
                rows[0]["n_valid_train"], rows[0]["n_valid_test"],
                rows[2]["n_valid_train"], rows[2]["n_valid_test"],
                fmt(rows[0]["mse_const"])))

    print()
    print("== 差表（results mse_test）")
    print("%-9s %-7s | %14s | %14s | %14s" % (
        "channel", "readout", "offset−ref", "offset−ctrl", "ctrl−ref"))
    for c in CHANNELS:
        for r in READOUTS:
            print("%-9s %-7s | %+14.4g | %+14.4g | %+14.4g" % (
                c, r, res_diff_or[c + "|" + r], res_diff_oc[c + "|" + r],
                res_diff_cr[c + "|" + r]))

    print()
    print("== verdict 差（mean [lo, hi]；offset−ctrl 同 ckpt 同设备 = GT 轴效应）")
    ver_diff_oc = {}
    ver_diff_or = {}
    for k in sorted(d["ref"]["verdict"].keys()):
        a = d["ref"]["verdict"][k]
        b = d["ctrl"]["verdict"][k]
        c2 = d["offset"]["verdict"][k]
        if a is None or b is None or c2 is None:
            print("%-32s ref=%s ctrl=%s offset=%s" % (k, a, b, c2))
            ver_diff_oc[k] = ver_diff_or[k] = None
            continue
        d_oc = [c2[i] - b[i] for i in range(3)]
        d_or = [c2[i] - a[i] for i in range(3)]
        ver_diff_oc[k] = d_oc
        ver_diff_or[k] = d_or
        print("%-32s oc %+.3g [%+.3g,%+.3g] | or %+.3g [%+.3g,%+.3g]" % (
            k, d_oc[0], d_oc[1], d_oc[2], d_or[0], d_or[1], d_or[2]))

    def verdict_table(name, key):
        print()
        print("== 显著性判定对照（%s；粗体无，SIG = CI 不含 0）" % name)
        print("%-9s | %-22s | %-22s" % ("channel", key[0], key[1]))
        for c in CHANNELS:
            t1 = d[key[0]]["verdict"][c + "|" + key[2]]
            t2 = d[key[1]]["verdict"][c + "|" + key[2]]
            s1 = "%s %+8.4g" % (sig(t1), t1[0]) if t1 else "n/a"
            s2 = "%s %+8.4g" % (sig(t2), t2[0]) if t2 else "n/a"
            flip = "  <== FLIP" if (t1 and t2 and sig(t1) != sig(t2)) else ""
            print("%-9s | %-22s | %-22s%s" % (c, s1, s2, flip))

    verdict_table("T 可读（const−raw108，无编码器）", ("ref", "offset", "raw108|vs_const"))
    verdict_table("T 可读（const−raw108，同 ckpt）", ("ctrl", "offset", "raw108|vs_const"))
    verdict_table("V 更强（raw108−v_tok，同 ckpt）", ("ctrl", "offset", "v_tok|vs_raw108"))
    verdict_table("vt>const（同 ckpt）", ("ctrl", "offset", "vt|vs_const"))

    payload = {
        "provenance": {k: v.get("provenance") for k, v in d.items()},
        "results": {v: {f"{c}|{r}": d[v]["results"][c + "|" + r]
                        for c in CHANNELS for r in READOUTS}
                    for v in d},
        "diff": {"offset_minus_ref": res_diff_or,
                 "offset_minus_ctrl": res_diff_oc,
                 "ctrl_minus_ref": res_diff_cr},
        "verdict_diff": {"offset_minus_ctrl": ver_diff_oc,
                         "offset_minus_ref": ver_diff_or},
        "n_valid_train_test": {"ref": {k: v[:2] for k, v in n_valid_all.items()},
                               "offset": {k: v[2:] for k, v in n_valid_all.items()}},
        "verdict": {v: d[v]["verdict"] for v in d},
    }
    Path(args.out).write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
