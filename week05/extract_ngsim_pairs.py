# -*- coding: utf-8 -*-
"""
从 NGSIM I-80 数据中提取「真实跟驰车对」。

数据本身不带 Preceding/Following 列，因此**自行重建**前车-后车关系：
对每一帧、每条车道，按 local_y_m 排序，相邻两辆车即构成一对（后车跟驰前车）。

先做一次坐标约定核对
-------------------
NGSIM 的 local_y 到底是「车头中心」还是「车辆质心」，直接决定净间距怎么算：

    车头约定：净间距 = (y_lead - y_own) - L_lead
    质心约定：净间距 = (y_lead - y_own) - (L_lead + L_own) / 2

两者在车队里差半个车身，不能靠猜。这里用数据自己判：统计
(y_lead - y_own) - L_lead 的下分位与「后车车长」的关系 ——
若是车头约定，该量下界应贴近 0 且与后车车长无关。

输出
----
results/ngsim_pairs.npz          轨迹数组（定长补齐）
results/ngsim_pairs.csv          每对车的元信息
results/ngsim_pair_diagnostic.txt / .json   坐标约定核对记录
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DEFAULT_CSV = Path(r"E:\jiaotongjianmo\week04\第4周_NGSIM-I80_课堂数据"
                   r"\NGSIM_I80_20050413_1600-1615_SI.csv")

USE_COLS = ["vehicle_id", "frame_id", "time_s", "local_y_m", "vehicle_length_m",
            "vehicle_class", "speed_m_s", "acceleration_m_s2", "lane_id"]

MIN_FRAMES = 150          # 至少 15 s
MAX_PAIRS = 240
SPEED_BINS = ((0.0, 3.0), (3.0, 6.0), (6.0, 9.0), (9.0, 30.0))

LOG = []


def say(s=""):
    LOG.append(str(s))
    print(s)


# --------------------------------------------------------------------------
def build_pairs(csv: Path) -> pd.DataFrame:
    say(f"读取 {csv.name} ...")
    df = pd.read_csv(csv, usecols=USE_COLS)
    say(f"  行数 {len(df):,}，车辆 {df.vehicle_id.nunique()}，"
        f"时间 {df.time_s.min():.1f}–{df.time_s.max():.1f} s")

    # 同帧同车道按纵向位置排序后，相邻行即为「后车 - 前车」
    df = df.sort_values(["lane_id", "frame_id", "local_y_m"],
                        kind="mergesort").reset_index(drop=True)
    g = df.groupby(["lane_id", "frame_id"], sort=False)
    df["lead_id"] = g["vehicle_id"].shift(-1)
    df["lead_y"] = g["local_y_m"].shift(-1)
    df["lead_v"] = g["speed_m_s"].shift(-1)
    df["lead_len"] = g["vehicle_length_m"].shift(-1)
    df["lead_cls"] = g["vehicle_class"].shift(-1)
    df["lead_a"] = g["acceleration_m_s2"].shift(-1)
    df["lead_frame"] = g["frame_id"].shift(-1)

    p = df.dropna(subset=["lead_id"]).copy()
    p = p.rename(columns={"vehicle_id": "own_id", "local_y_m": "own_y",
                          "speed_m_s": "own_v", "vehicle_length_m": "own_len",
                          "vehicle_class": "own_cls",
                          "acceleration_m_s2": "own_a"})
    p["lead_id"] = p["lead_id"].astype(int)
    p["d_m"] = p.lead_y - p.own_y
    say(f"  同帧同车道相邻车对记录 {len(p):,} 条")
    return p


# --------------------------------------------------------------------------
def diagnose_convention(p: pd.DataFrame):
    """判定 local_y 的坐标约定。"""
    say("")
    say("=" * 74)
    say("坐标约定核对：local_y 是车头还是质心？")
    say("=" * 74)

    d = p.d_m.values
    L_lead = p.lead_len.values
    L_own = p.own_len.values

    gap_front = d - L_lead
    gap_center = d - 0.5 * (L_lead + L_own)

    def pct(x, q):
        return float(np.percentile(x, q))

    say(f"  样本 {len(d):,} 条")
    say(f"  (y_lead - y_own)                      : 下界 {d.min():.3f} m，"
        f"0.1% {pct(d, 0.1):.3f}，1% {pct(d, 1):.3f}，50% {pct(d, 50):.3f}")
    say(f"  按车头约定 gap = d - L_lead           : 下界 {gap_front.min():.3f} m，"
        f"0.1% {pct(gap_front, 0.1):.3f}，1% {pct(gap_front, 1):.3f}，"
        f"负值占比 {np.mean(gap_front < 0) * 100:.3f}%")
    say(f"  按质心约定 gap = d - (L_lead+L_own)/2 : 下界 {gap_center.min():.3f} m，"
        f"0.1% {pct(gap_center, 0.1):.3f}，1% {pct(gap_center, 1):.3f}，"
        f"负值占比 {np.mean(gap_center < 0) * 100:.3f}%")

    # 关键判据：车头约定下 gap_front 与「后车车长」无关；质心约定下它约等于 L_own/2
    say("")
    say("  按后车车长分组看 gap_front 的 1% 分位（车头约定应恒定 ≈0）:")
    bins = [(0, 4.0), (4.0, 4.6), (4.6, 5.2), (5.2, 6.5), (6.5, 20.0)]
    rows = []
    for lo, hi in bins:
        m = (L_own >= lo) & (L_own < hi)
        if m.sum() < 200:
            continue
        gf = float(np.percentile(gap_front[m], 1))
        gc = float(np.percentile(gap_center[m], 1))
        rows.append(dict(len_lo=lo, len_hi=hi, n=int(m.sum()),
                         gap_front_p1=gf, gap_center_p1=gc))
        say(f"    后车长 [{lo:4.1f},{hi:4.1f}) m  n={int(m.sum()):>8d}   "
            f"gap_front 1% = {gf:7.3f} m   gap_center 1% = {gc:7.3f} m")

    neg_frac_front = float(np.mean(gap_front < 0))
    # 若 gap_front 极少为负且与后车车长无关 → 车头约定
    convention = "front_bumper" if neg_frac_front < 0.005 else "center"
    say("")
    say(f"  判定结果：local_y 为『车{'头中心' if convention == 'front_bumper' else '辆质心'}』"
        f"（gap_front 负值占比 {neg_frac_front * 100:.3f}%）")
    say("")
    return dict(convention=convention,
                neg_frac_gap_front=neg_frac_front,
                gap_front_min=float(gap_front.min()),
                gap_center_min=float(gap_center.min()),
                by_follower_length=rows)


# --------------------------------------------------------------------------
def extract(p: pd.DataFrame, convention: str):
    say("=" * 74)
    say("提取连续跟驰车对")
    say("=" * 74)

    if convention == "front_bumper":
        p = p.assign(gap_net=p.d_m - p.lead_len,
                     lead_rear=p.lead_y - p.lead_len)
    else:
        p = p.assign(gap_net=p.d_m - 0.5 * (p.lead_len + p.own_len),
                     lead_rear=p.lead_y - 0.5 * p.lead_len)
    p = p.assign(own_front=p.own_y)

    # 逐对统计持续时间
    say("  按 (后车, 前车) 聚合 ...")
    p["key"] = p.own_id.astype(np.int64) * 100000 + p.lead_id
    gb = p.groupby("key", sort=False)
    stat = gb.agg(n=("frame_id", "size"),
                  f_min=("frame_id", "min"),
                  f_max=("frame_id", "max"),
                  lane=("lane_id", "first"),
                  t0=("time_s", "min"),
                  own_v=("own_v", "mean"),
                  lead_v=("lead_v", "mean"),
                  gap=("gap_net", "mean"),
                  gap_min=("gap_net", "min"),
                  own_cls=("own_cls", "first"),
                  lead_cls=("lead_cls", "first"))
    stat = stat.reset_index()
    stat["span"] = stat.f_max - stat.f_min + 1
    say(f"  候选车对 {len(stat):,} 个")

    cand = stat[(stat.n >= MIN_FRAMES) & (stat.n == stat.span)].copy()
    say(f"  连续帧且 >= {MIN_FRAMES} 帧（{MIN_FRAMES * 0.1:.0f} s）: {len(cand):,} 个")

    cand = cand[(cand.own_v > 1.0) & (cand.lead_v > 1.0)]
    cand = cand[(cand.gap > 0.5) & (cand.gap < 120.0) & (cand.gap_min > 0.0)]
    say(f"  速度与间距合理: {len(cand):,} 个")

    cand["v_mean"] = 0.5 * (cand.own_v + cand.lead_v)
    cand["regime"] = pd.cut(cand.v_mean, bins=[0, 3, 6, 9, 30],
                            labels=["拥堵 <3", "低速 3-6", "中速 6-9", "畅通 9+"])

    # 分层抽样，保证各速度区间都有代表
    per_bin = max(1, MAX_PAIRS // len(SPEED_BINS))
    picked = []
    for lo, hi in SPEED_BINS:
        sub = cand[(cand.v_mean >= lo) & (cand.v_mean < hi)]
        if len(sub) == 0:
            continue
        take = sub.sample(n=min(per_bin, len(sub)), random_state=0)
        picked.append(take)
    sel = pd.concat(picked).sort_values(["lane", "t0"]).reset_index(drop=True)
    sel["pair_id"] = np.arange(len(sel))
    say(f"  分层抽样后参与评估: {len(sel)} 对")
    say("")
    say("  各速度区间样本数：")
    for lab, cnt in sel.regime.value_counts().sort_index().items():
        say(f"    {lab:10s} {cnt:4d}")
    say("")

    # 取轨迹
    p_idx = p.set_index("key")
    Tmax = int(sel.span.max())
    n = len(sel)
    arr = {k: np.full((n, Tmax), np.nan) for k in
           ("t", "lead_rear_y", "lead_v", "lead_a", "own_front_y", "own_v",
            "own_a", "gap_net")}
    lengths = np.zeros(n, dtype=int)

    cols = ["time_s", "lead_rear", "lead_v", "lead_a", "own_front",
            "own_v", "own_a", "gap_net"]
    keys = ["t", "lead_rear_y", "lead_v", "lead_a", "own_front_y",
            "own_v", "own_a", "gap_net"]

    for i, row in sel.iterrows():
        sub = p_idx.loc[row.key].sort_values("frame_id")
        m = len(sub)
        lengths[i] = m
        for c, k in zip(cols, keys):
            arr[k][i, :m] = sub[c].values
        # 时间归零
        arr["t"][i, :m] = arr["t"][i, :m] - arr["t"][i, 0]

    sel["n_frames"] = lengths
    sel["duration_s"] = (lengths - 1) * 0.1

    np.savez_compressed(RESULTS / "ngsim_pairs.npz",
                        pair_ids=sel.pair_id.values,
                        lengths=lengths,
                        **arr)
    keep = ["pair_id", "lane", "t0", "duration_s", "n_frames", "v_mean",
            "regime", "own_v", "lead_v", "gap", "gap_min", "own_cls", "lead_cls"]
    sel[keep].to_csv(RESULTS / "ngsim_pairs.csv", index=False,
                     encoding="utf-8-sig")

    say(f"  轨迹已存 results/ngsim_pairs.npz（最长 {Tmax} 帧 = {Tmax * 0.1:.1f} s）")
    say(f"  时长: 中位 {sel.duration_s.median():.1f} s，"
        f"最短 {sel.duration_s.min():.1f} s，最长 {sel.duration_s.max():.1f} s")
    say(f"  平均净间距: 中位 {sel.gap.median():.2f} m，"
        f"范围 {sel.gap.min():.2f}–{sel.gap.max():.2f} m")
    return sel


# --------------------------------------------------------------------------
def main():
    csv = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    if not csv.is_file():
        raise SystemExit(f"找不到数据文件：{csv}")

    p = build_pairs(csv)
    diag = diagnose_convention(p)
    sel = extract(p, diag["convention"])

    (RESULTS / "ngsim_pair_diagnostic.json").write_text(
        json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "extract_ngsim_pairs_report.txt").write_text(
        "\n".join(LOG), encoding="utf-8")
    print("已写出 results/ngsim_pairs.npz / .csv / ngsim_pair_diagnostic.json")


if __name__ == "__main__":
    main()
