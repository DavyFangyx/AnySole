"""CoP/相位探针三版对比助手（2026-10-20，v3 收口审计）。

读取 10-19 既有 JSON（raw108 / denoised108_v2，GPU 跑）与本审计 v3 JSON
（CPU 跑），输出：
  - 表1 数字：4 通道 × 3 读出 results mse_test（+ const/lam/n_valid 一致性）；
  - v3−v2 差表：results mse_test 与 verdict (mean/lo/hi) 两个精度级；
  - 可选：v2 同设备（CPU）参考 JSON，用于把 v3−v2 差分解为
    「数据版本差」与「设备算术漂移」（--v2-cpu 给出时输出漂移表）；
  - verdict 显著性判定（CI 是否含 0）。

用法：
  python compare_probe.py --ref-dir <audit_20261019/cop_phase> \
      --v3 <probe_v5_phase_cop_readout_denoised108_v3.json> \
      [--v2-cpu <cpu 参考 json>] [--out compare_summary.json]
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
    ap.add_argument("--ref-dir", type=Path, required=True)
    ap.add_argument("--v3", type=Path, required=True)
    ap.add_argument("--v2-cpu", type=Path, default=None)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "compare_summary.json")
    args = ap.parse_args()

    ref_dir: Path = args.ref_dir
    d = {
        "raw108": load(ref_dir / "probe_v5_phase_cop_readout_raw108.json"),
        "v2": load(ref_dir / "probe_v5_phase_cop_readout_denoised108_v2.json"),
        "v3": load(args.v3),
    }
    v2cpu = load(args.v2_cpu) if args.v2_cpu else None

    print("== 表1：results mse_test（通道×读出×版本）")
    print("%-9s %-7s | %12s %12s %12s | const/lam/n_valid 一致性" % (
        "channel", "readout", "raw108", "v2", "v3"))
    for c in CHANNELS:
        for r in READOUTS:
            rows = [d[v]["results"][c + "|" + r] for v in ("raw108", "v2", "v3")]
            assert all(x["status"] == "ok" for x in rows), (c, r, [x.get("status") for x in rows])
            const_same = len({fmt(x["mse_const"], 12) for x in rows}) == 1
            lam_same = len({fmt(x["lam"], 6) for x in rows}) == 1
            n_same = len({(x["n_valid_train"], x["n_valid_test"]) for x in rows}) == 1
            print("%-9s %-7s | %12s %12s %12s | const=%s lam=%s n=%s" % (
                c, r, fmt(rows[0]["mse_test"]), fmt(rows[1]["mse_test"]),
                fmt(rows[2]["mse_test"]), const_same, lam_same, n_same))

    print()
    print("== v3−v2 差（results mse_test）")
    print("%-9s %-7s | %12s" % ("channel", "readout", "d_mse_test"))
    res_diff = {}
    for c in CHANNELS:
        for r in READOUTS:
            dv = (d["v3"]["results"][c + "|" + r]["mse_test"]
                  - d["v2"]["results"][c + "|" + r]["mse_test"])
            res_diff[c + "|" + r] = dv
            print("%-9s %-7s | %+12.3g" % (c, r, dv))

    print()
    print("== v3−v2 差（verdict：mean/lo/hi 三元组，逐键）")
    keys_v2 = sorted(d["v2"]["verdict"].keys())
    keys_v3 = sorted(d["v3"]["verdict"].keys())
    assert keys_v2 == keys_v3, (set(keys_v2) ^ set(keys_v3))
    ver_diff = {}
    for k in keys_v2:
        a = d["v2"]["verdict"][k]
        b = d["v3"]["verdict"][k]
        if a is None or b is None:
            print("%-32s v2=%s v3=%s" % (k, a, b))
            ver_diff[k] = None
            continue
        dd = [b[i] - a[i] for i in range(3)]
        ver_diff[k] = dd
        print("%-32s %+.3g [%+.3g, %+.3g]" % (k, dd[0], dd[1], dd[2]))

    if v2cpu is not None:
        print()
        print("== 设备漂移：v2(CPU 参考) − v2(10-19 GPU JSON)")
        print("%-9s %-7s | %12s" % ("channel", "readout", "d_mse_test"))
        dev_drift = {}
        for c in CHANNELS:
            for r in READOUTS:
                dv = (v2cpu["results"][c + "|" + r]["mse_test"]
                      - d["v2"]["results"][c + "|" + r]["mse_test"])
                dev_drift[c + "|" + r] = dv
                print("%-9s %-7s | %+12.3g" % (c, r, dv))
        print("-- verdict 级最大 |Δ|：")
        for k in sorted(d["v2"]["verdict"].keys()):
            a = v2cpu["verdict"][k]
            b = d["v2"]["verdict"][k]
            if a is None or b is None:
                continue
            print("   %-32s %+.3g" % (k, max(abs(a[i] - b[i]) for i in range(3))))
        # v3−v2 同设备差（v3 CPU − v2 CPU）
        print()
        print("== 同设备 v3−v2 差（CPU：v3 − v2cpu）")
        same_dev = {}
        for c in CHANNELS:
            for r in READOUTS:
                dv = (d["v3"]["results"][c + "|" + r]["mse_test"]
                      - v2cpu["results"][c + "|" + r]["mse_test"])
                same_dev[c + "|" + r] = dv
                print("%-9s %-7s | %+12.3g" % (c, r, dv))
        print("-- verdict 级同设备差：")
        for k in sorted(keys_v2):
            a = v2cpu["verdict"][k]
            b = d["v3"]["verdict"][k]
            if a is None or b is None:
                continue
            dd = [b[i] - a[i] for i in range(3)]
            print("   %-32s %+.3g [%+.3g, %+.3g]" % (k, dd[0], dd[1], dd[2]))
    else:
        dev_drift, same_dev = None, None

    print()
    print("== verdict 显著性（v3；粗体无）")
    print("%-9s | %-8s | %-8s | %-8s" % ("channel", "T可读", "vt>const", "V更强"))
    for c in CHANNELS:
        t1 = d["v3"]["verdict"][c + "|raw108|vs_const"]
        t2 = d["v3"]["verdict"][c + "|vt|vs_const"]
        t3 = d["v3"]["verdict"][c + "|v_tok|vs_raw108"]
        print("%-9s | %s %+.4g | %s %+.4g | %s %+.4g" % (
            c, sig(t1), t1[0], sig(t2), t2[0], sig(t3), t3[0]))

    payload = {
        "versions": {"raw108": str(ref_dir / "probe_v5_phase_cop_readout_raw108.json"),
                     "v2": str(ref_dir / "probe_v5_phase_cop_readout_denoised108_v2.json"),
                     "v3": str(args.v3),
                     "v2_cpu": str(args.v2_cpu) if args.v2_cpu else None},
        "results": {v: {f"{c}|{r}": d[v]["results"][c + "|" + r]
                        for c in CHANNELS for r in READOUTS}
                    for v in d},
        "diff_v3_minus_v2": {"results_mse": res_diff, "verdict": ver_diff},
        "device_drift_v2cpu_minus_v2gpu": (
            {"results_mse": dev_drift} if dev_drift is not None else None),
        "same_device_v3_minus_v2cpu": (
            {"results_mse": same_dev} if same_dev is not None else None),
    }
    Path(args.out).write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
