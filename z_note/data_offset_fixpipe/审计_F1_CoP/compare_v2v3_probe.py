"""CoP/相位探针三版（修正轴）对比助手（2026-10-20，v2v3+offset 补审）。

读取三个 JSON（同修正轴）：
  anchor = probe_v5_phase_cop_readout_raw108_offset.json（raw108+offset）
  v2     = probe_v5_phase_cop_readout_denoised108_v2_offset.json
  v3     = probe_v5_phase_cop_readout_denoised108_v3_offset.json
输出：
  - 表1：results mse_test（4 通道 × 3 读出 × 3 版本）+ const/lam/n_valid；
  - 差表：v2−anchor、v3−anchor、v3−v2（results mse + verdict 三元组）；
  - 显著性判定对照（CI 是否含 0；T 可读 = const−raw108，V 更强 =
    raw108−v_tok，vt>const）。

口径事实（如实登记）：anchor 的 provenance.ckpt =
V3_3B_denoised_joint_and；v2/v3 本跑按任务指定 ckpt =
V3_3B_joint_and。raw108 读出（特征 = T 拼接）与 T 可读判据（const−raw108）
不含编码器 → 三版直接可比；v_tok/vt 读出含编码器 → anchor↔v2/v3 跨 ckpt
不可比（v2↔v3 同 ckpt 可比）。
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
    ap.add_argument("--anchor", type=Path, required=True,
                    help="raw108+offset JSON")
    ap.add_argument("--v2", type=Path, required=True,
                    help="denoised108_v2+offset JSON")
    ap.add_argument("--v3", type=Path, required=True,
                    help="denoised108_v3+offset JSON")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "compare_v2v3_summary.json")
    args = ap.parse_args()

    d = {k: load(p) for k, p in (("anchor", args.anchor), ("v2", args.v2),
                                 ("v3", args.v3))}
    for k, v in d.items():
        print("%-7s provenance=%s" % (k, v.get("provenance")))

    print()
    print("== 表1：results mse_test（通道×读出×版本）")
    print("%-9s %-7s | %12s %12s %12s | n_valid(anchor/v2/v3)  const" % (
        "channel", "readout", "anchor", "v2", "v3"))
    res_diff = {"v2_anchor": {}, "v3_anchor": {}, "v3_v2": {}}
    n_valid_all = {}
    for c in CHANNELS:
        for r in READOUTS:
            rows = [d[v]["results"][c + "|" + r] for v in ("anchor", "v2", "v3")]
            assert all(x["status"] == "ok" for x in rows), (c, r)
            k = c + "|" + r
            n_valid_all[k] = {v: (rows[i]["n_valid_train"], rows[i]["n_valid_test"])
                              for i, v in enumerate(("anchor", "v2", "v3"))}
            res_diff["v2_anchor"][k] = rows[1]["mse_test"] - rows[0]["mse_test"]
            res_diff["v3_anchor"][k] = rows[2]["mse_test"] - rows[0]["mse_test"]
            res_diff["v3_v2"][k] = rows[2]["mse_test"] - rows[1]["mse_test"]
            print("%-9s %-7s | %12s %12s %12s | %5d/%-5d %5d/%-5d %5d/%-5d  %s" % (
                c, r, fmt(rows[0]["mse_test"]), fmt(rows[1]["mse_test"]),
                fmt(rows[2]["mse_test"]),
                rows[0]["n_valid_train"], rows[0]["n_valid_test"],
                rows[1]["n_valid_train"], rows[1]["n_valid_test"],
                rows[2]["n_valid_train"], rows[2]["n_valid_test"],
                fmt(rows[0]["mse_const"])))

    print()
    print("== 差表（results mse_test）")
    print("%-9s %-7s | %14s | %14s | %14s" % (
        "channel", "readout", "v2−anchor", "v3−anchor", "v3−v2"))
    for c in CHANNELS:
        for r in READOUTS:
            k = c + "|" + r
            print("%-9s %-7s | %+14.4g | %+14.4g | %+14.4g" % (
                c, r, res_diff["v2_anchor"][k], res_diff["v3_anchor"][k],
                res_diff["v3_v2"][k]))

    print()
    print("== verdict 三元组差（mean [lo, hi]；同 GT 轴）")
    ver_diff = {"v2_anchor": {}, "v3_anchor": {}, "v3_v2": {}}
    for kk in sorted(d["anchor"]["verdict"].keys()):
        a = d["anchor"]["verdict"][kk]
        b = d["v2"]["verdict"][kk]
        c2 = d["v3"]["verdict"][kk]
        if a is None or b is None or c2 is None:
            print("%-32s anchor=%s v2=%s v3=%s" % (kk, a, b, c2))
            for key in ver_diff:
                ver_diff[key][kk] = None
            continue
        for key, (x, y) in {"v2_anchor": (b, a), "v3_anchor": (c2, a),
                            "v3_v2": (c2, b)}.items():
            ver_diff[key][kk] = [x[i] - y[i] for i in range(3)]
        print("%-32s v2−a %+.3g [%+.3g,%+.3g] | v3−a %+.3g [%+.3g,%+.3g] | v3−v2 %+.3g [%+.3g,%+.3g]"
              % (kk, ver_diff["v2_anchor"][kk][0], ver_diff["v2_anchor"][kk][1],
                 ver_diff["v2_anchor"][kk][2], ver_diff["v3_anchor"][kk][0],
                 ver_diff["v3_anchor"][kk][1], ver_diff["v3_anchor"][kk][2],
                 ver_diff["v3_v2"][kk][0], ver_diff["v3_v2"][kk][1],
                 ver_diff["v3_v2"][kk][2]))

    def verdict_table(name, keys, note=""):
        print()
        print("== 显著性判定对照（%s；SIG = CI 不含 0）%s" % (name, note))
        hdr = "%-9s | " % "channel" + " | ".join("%-20s" % v for v in keys)
        print(hdr)
        for c in CHANNELS:
            cells = []
            flips = set()
            for v in keys:
                t = d[v]["verdict"][c + "|" + VERDICT_SUFFIX]
                cells.append("%s %+8.4g" % (sig(t), t[0]) if t else "n/a")
            states = {v: (sig(d[v]["verdict"][c + "|" + VERDICT_SUFFIX])
                          if d[v]["verdict"][c + "|" + VERDICT_SUFFIX] else None)
                      for v in keys}
            nonnull = [s for s in states.values() if s is not None]
            flip = "  <== FLIP" if len(set(nonnull)) > 1 else ""
            print("%-9s | " % c + " | ".join("%-20s" % x for x in cells) + flip)

    VERDICT_SUFFIX = "raw108|vs_const"
    verdict_table("T 可读（const−raw108，无编码器，三版直接可比）",
                  ("anchor", "v2", "v3"))
    VERDICT_SUFFIX = "v_tok|vs_raw108"
    verdict_table("V 更强（raw108−v_tok）",
                  ("v2", "v3"), note="（v2/v3 同 ckpt；anchor ckpt 不同不列）")
    VERDICT_SUFFIX = "vt|vs_const"
    verdict_table("vt>const", ("v2", "v3"),
                  note="（v2/v3 同 ckpt；anchor ckpt 不同不列）")

    payload = {
        "provenance": {k: v.get("provenance") for k, v in d.items()},
        "results": {v: {f"{c}|{r}": d[v]["results"][c + "|" + r]
                        for c in CHANNELS for r in READOUTS}
                    for v in d},
        "diff_results": res_diff,
        "verdict_diff": ver_diff,
        "n_valid_train_test": n_valid_all,
        "verdict": {v: d[v]["verdict"] for v in d},
    }
    Path(args.out).write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
