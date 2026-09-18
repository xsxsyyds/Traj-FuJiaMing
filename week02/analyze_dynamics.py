#!/usr/bin/env python3
"""week02 · 步骤① 续 —— 速度与方向分析。

步骤① 的轨迹可视化只回答了"人往哪走"，这一节回答"走多快、朝哪个方向走"。

已知帧率 25 fps（帧间隔 0.04 s），因此相邻帧位移可以直接折算成 m/s。
相邻帧位移中位仅 5 cm，差分噪声不可忽略，故速度默认对位置先做 5 帧
（0.2 s）滑动平均再中心差分；``--smooth 1`` 可关掉。

输出的图
--------
- ``results/fig4_speed.png``       速率—时间剖面、速率沿进度剖面、速率分布
- ``results/fig5_congestion.png``  速率—邻居数、最近两人间距、到达时间
- ``results/fig6_heading.png``     横向偏移剖面、偏航角剖面、过中心侧别

用法
----
    python week02/analyze_dynamics.py
    python week02/analyze_dynamics.py --smooth 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from circle_data import (ACCENT, BASE, CYCLIC, GRID, TEXT, CircleTrial,
                         apply_style, default_data_path, draw_ring, load,
                         save, style_axes)

apply_style()


# ------------------------------------------------------------------
# 小工具
# ------------------------------------------------------------------

def band_profile(x: np.ndarray, y: np.ndarray, edges: np.ndarray,
                 qs=(10, 25, 50, 75, 90), min_n: int = 40):
    """把散点按 ``x`` 分箱，返回每个箱的分位数。

    为避免把不同长度的轨迹硬插值到公共进度轴，统一用"按 x 分箱统计"
    的方式构造剖面。
    """
    idx = np.digitize(x, edges)
    mid = 0.5 * (edges[:-1] + edges[1:])
    out = {q: np.full(mid.shape, np.nan) for q in qs}
    cnt = np.zeros(mid.shape, dtype=int)
    for i in range(1, len(edges)):
        sel = idx == i
        cnt[i - 1] = int(sel.sum())
        if cnt[i - 1] >= min_n:
            for q in qs:
                out[q][i - 1] = np.nanpercentile(y[sel], q)
    return mid, out, cnt


def _smooth_nan(v: np.ndarray, k: int = 5) -> np.ndarray:
    ok = ~np.isnan(v)
    if ok.sum() <= k:
        return v
    out = v.copy()
    vals = v[ok]
    pad = k // 2
    sm = np.convolve(np.pad(vals, pad, mode="edge"), np.ones(k) / k,
                     mode="valid")[:vals.size]
    out[ok] = sm
    return out


def annotate(ax, text: str, loc: str = "upper left", **kw) -> None:
    pos = {"upper left": (0.03, 0.96, "left", "top"),
           "upper center": (0.50, 0.96, "center", "top"),
           "upper right": (0.97, 0.96, "right", "top"),
           "center left": (0.03, 0.50, "left", "center"),
           "center": (0.50, 0.50, "center", "center"),
           "center right": (0.97, 0.50, "right", "center"),
           "lower left": (0.03, 0.06, "left", "bottom"),
           "lower center": (0.50, 0.06, "center", "bottom"),
           "lower right": (0.97, 0.06, "right", "bottom")}[loc]
    ax.text(pos[0], pos[1], text, transform=ax.transAxes, fontsize=8.2,
            color=TEXT, ha=pos[2], va=pos[3], linespacing=1.45, **kw)


# ------------------------------------------------------------------
# 图 4：速度
# ------------------------------------------------------------------

def fig_speed(tr: CircleTrial, outdir: Path, dpi: int, smooth: int) -> dict:
    V = tr.velocities(smooth)
    S = np.linalg.norm(V, axis=2)                 # (N, A)
    t = tr.t
    tmid = t[1:-1]
    Smid = S[1:-1]
    m = tr.travel_mask()
    arr = tr.arrival_time()

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.5))

    # (a) 速率—时间
    ax = axes[0]
    med = np.median(Smid, axis=1)
    p10 = np.percentile(Smid, 10, axis=1)
    p90 = np.percentile(Smid, 90, axis=1)
    ax.fill_between(tmid, p10, p90, color=BASE, alpha=0.14, lw=0,
                    label="10–90 %")
    ax.plot(tmid, med, color=BASE, lw=1.6, label="median")
    ax.axhline(np.median(S[m]), color=ACCENT, ls="--", lw=1.1,
               label=f"travel median {np.median(S[m]):.2f} m/s")
    ax.set_xlim(0, tr.duration)
    ax.set_ylim(bottom=0)
    style_axes(ax, xlabel="time [s]", ylabel="speed [m/s]", equal=False)
    ax.set_title("全程速率变化", fontsize=10, color=TEXT, pad=8)

    # 右轴：还在行进的人数比例——解释末端中位数为何塌下来
    n_moving = (Smid > 0.3).sum(axis=1) / tr.n_agent
    ax2 = ax.twinx()
    ax2.plot(tmid, n_moving, color="#8C8C8C", lw=1.0, ls=":",
             label="still moving")
    ax2.set_ylabel("fraction still moving", fontsize=8.5, color="#8C8C8C")
    ax2.tick_params(colors="#8C8C8C", labelsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#BFBFBF")
    ax2.spines["left"].set_visible(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.3,
              loc="upper right")
    annotate(ax, "the median falls at the end because\nearly arrivals are "
                 "already standing", "lower left")

    # (b) 速率沿路径进度
    ax = axes[1]
    edges = np.linspace(0, 1, 41)
    rows_x = np.concatenate([(np.arange(tr.n_frame) + 0.5) / tr.n_frame
                             for _ in range(tr.n_agent)])
    rows_y = np.concatenate([S[:, i] for i in range(tr.n_agent)])
    mid, q, _ = band_profile(rows_x, rows_y, edges)
    ax.fill_between(mid, q[10], q[90], color=BASE, alpha=0.13, lw=0,
                    label="10–90 %")
    ax.fill_between(mid, q[25], q[75], color=BASE, alpha=0.22, lw=0,
                    label="25–75 %")
    ax.plot(mid, q[50], color=ACCENT, lw=1.8, label="median")
    style_axes(ax, xlabel="normalised path progress", ylabel="speed [m/s]",
               equal=False)
    ax.set_ylim(bottom=0)
    ax.set_title("速率沿路径的变化", fontsize=10, color=TEXT, pad=8)
    ax.legend(frameon=False, fontsize=8.3, loc="upper right")
    i_pk = int(np.nanargmax(q[50]))
    annotate(ax, f"peak {q[50][i_pk]:.2f} m/s\nat {mid[i_pk]:.0%} of the way",
             "lower right")

    # (c) 速率分布
    ax = axes[2]
    vals = S[m]
    bins = np.arange(0, 5.1, 0.25)
    ax.hist(vals, bins=bins, color=BASE, alpha=0.75, edgecolor="white",
            linewidth=0.5)
    vmed = float(np.median(vals))
    vfree = float(np.median(S[m & (tr.neighbours(2.0) <= 0)]))
    for x, c, lab in ((vmed, ACCENT, f"median {vmed:.2f}"),
                      (vfree, "#2A9D8F", f"isolated {vfree:.2f}")):
        ax.axvline(x, color=c, lw=1.6, label=lab)
    style_axes(ax, xlabel="speed [m/s]", ylabel="samples during travel",
               equal=False)
    ax.set_title("速率分布", fontsize=10, color=TEXT, pad=8)
    ax.legend(frameon=False, fontsize=8.3, loc="upper right")
    annotate(ax, "walking pace 1–2.5 m/s\ndominates", "center right")

    save(fig, outdir, "fig4_speed", dpi, root=ROOT)
    plt.close(fig)

    return {"speed_median_travel": vmed, "speed_free": vfree,
            "speed_p90": float(np.percentile(vals, 90)),
            "speed_max": float(vals.max())}


# ------------------------------------------------------------------
# 图 5：拥堵
# ------------------------------------------------------------------

def fig_congestion(tr: CircleTrial, outdir: Path, dpi: int,
                   smooth: int) -> dict:
    S = tr.speeds(smooth)
    m = tr.travel_mask()
    cnt = tr.neighbours(2.0)
    t = tr.t
    arr = tr.arrival_time()

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.5))

    # (a) 速率 vs 邻居数（按区间分组画条形，避免稀疏箱把坐标轴拉歪）
    ax = axes[0]
    groups = [(0, 1, "0"), (1, 2, "1"), (2, 3, "2"), (3, 4, "3"),
              (4, 5, "4"), (5, 6, "5"), (6, 8, "6–7"), (8, 11, "8–10"),
              (11, 99, "11+")]
    xs, med_v, iqr_lo, iqr_hi, ns = [], [], [], [], []
    for k, (lo, hi, _) in enumerate(groups):
        sel = m & (cnt >= lo) & (cnt < hi)
        if sel.sum() < 40:
            continue
        xs.append(k)
        med_v.append(np.median(S[sel]))
        iqr_lo.append(np.percentile(S[sel], 25))
        iqr_hi.append(np.percentile(S[sel], 75))
        ns.append(int(sel.sum()))
    xpos = np.arange(len(xs))
    yerr = np.maximum(0.0, np.array([np.array(med_v) - np.array(iqr_lo),
                                     np.array(iqr_hi) - np.array(med_v)]))
    ax.bar(xpos, med_v, yerr=yerr, width=0.68, color=BASE, alpha=0.85,
           edgecolor="white", linewidth=0.7,
           error_kw=dict(ecolor="#8FA6BC", elinewidth=1.3, capsize=3))
    for x, v, n in zip(xpos, med_v, ns):
        ax.text(x, 0.06, f"{n}", ha="center", va="bottom", fontsize=7.2,
                color="white", rotation=90)
    ax.set_xticks(xpos)
    ax.set_xticklabels([groups[k][2] for k in xs])
    rho = stats.spearmanr(cnt[m], S[m]).statistic
    style_axes(ax, xlabel="neighbours within 2 m",
               ylabel="speed [m/s]  (median, IQR)", equal=False)
    ax.set_ylim(0, 3.4)
    ax.set_title("人群越密，走得越慢",
                 fontsize=10, color=TEXT, pad=8)
    annotate(ax, f"Spearman ρ = {rho:.2f}\nn = {int(m.sum())} samples"
                 "\n(numbers in bars = samples)", "upper right")

    # (b) 最近两人间距随时间
    ax = axes[1]
    nd = tr.nearest_distance()
    ax.plot(t, nd, color=BASE, lw=1.2)
    ax.axhline(np.median(nd), color=ACCENT, ls="--", lw=1.1,
               label=f"median {np.median(nd):.2f} m")
    i = int(np.argmin(nd))
    ax.plot(t[i], nd[i], marker="v", ms=6, color=ACCENT, mec="white", mew=0.6,
            zorder=5)
    ax.annotate(f"closest approach {nd[i]:.2f} m",
                xy=(t[i], nd[i]), xytext=(t[i] - 5.6, 0.13),
                fontsize=8.2, color=ACCENT, va="center",
                arrowprops=dict(arrowstyle="->", color="#BBBBBB", lw=0.7,
                                shrinkA=2, shrinkB=3))
    for r in (0.5, 0.25):
        ax.axhline(r, color=GRID, ls=":", lw=0.9)
    ax.text(0.25, 0.28, "0.25 m ≈ body depth", fontsize=8, color="#8C8C8C")
    style_axes(ax, xlabel="time [s]",
               ylabel="distance between the two closest people [m]",
               equal=False)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, tr.duration)
    ax.set_title("两人最贴近到什么程度",
                 fontsize=10, color=TEXT, pad=8)
    ax.legend(frameon=False, fontsize=8.3, loc="lower right")

    # (c) 到达时间分布
    ax = axes[2]
    bins = np.arange(arr.min() - 0.5, arr.max() + 0.5, 0.75)
    ax.hist(arr, bins=bins, color=BASE, alpha=0.75, edgecolor="white",
            linewidth=0.5)
    q1, q2, q3 = np.percentile(arr, [25, 50, 75])
    ax.axvline(q2, color=ACCENT, lw=1.6, label=f"median {q2:.1f} s")
    ax.axvspan(q1, q3, color=ACCENT, alpha=0.08, lw=0)
    style_axes(ax, xlabel="arrival time [s]", ylabel="number of pedestrians",
               equal=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.30)
    ax.set_title("到达时间分布", fontsize=10, color=TEXT, pad=8)
    ax.legend(frameon=False, fontsize=8.3, loc="upper left")
    annotate(ax, f"IQR {q1:.1f}–{q3:.1f} s\np10–p90 spread "
                 f"{np.percentile(arr, 90) - np.percentile(arr, 10):.1f} s",
             "upper right")

    save(fig, outdir, "fig5_congestion", dpi, root=ROOT)
    plt.close(fig)

    return {"spearman_cnt_speed": float(rho),
            "nearest_median": float(np.median(nd)),
            "nearest_min": float(nd.min()),
            "arrival_median": float(q2), "arrival_iqr": (float(q1), float(q3))}


# ------------------------------------------------------------------
# 图 6：方向
# ------------------------------------------------------------------

def fig_heading(tr: CircleTrial, outdir: Path, dpi: int, smooth: int) -> dict:
    N, A = tr.n_frame, tr.n_agent
    lat = tr.lateral()
    dev = tr.bearing_deviation(smooth)
    S = tr.speeds(smooth)
    m = tr.travel_mask()
    prog = np.linspace(0, 1, N)

    # 过中心时的侧别：取最近圆心那一帧的横向偏移
    pass_lat = np.array([lat[int(np.argmin(np.linalg.norm(
        tr.aligned()[:, i], axis=1))), i] for i in range(A)])
    n_right = int((pass_lat > 0).sum())
    pval = 2 * stats.binom.sf(max(n_right, A - n_right) - 1, A, 0.5)

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.5))

    # (a) 横向偏移剖面
    ax = axes[0]
    rows_x = np.concatenate([np.linspace(0, 1, N) for _ in range(A)])
    rows_y = lat.ravel(order="F")
    edges = np.linspace(0, 1, 41)
    mid, q, _ = band_profile(rows_x, rows_y, edges)
    ax.fill_between(mid, q[10], q[90], color=BASE, alpha=0.13, lw=0,
                    label="10–90 %")
    ax.fill_between(mid, q[25], q[75], color=BASE, alpha=0.22, lw=0,
                    label="25–75 %")
    ax.plot(mid, q[50], color=ACCENT, lw=1.8, label="median")
    ax.axhline(0, color="#9A9A9A", lw=0.9, ls=":")
    i_mid = int(np.nanargmax(q[90] - q[10]))
    ax.annotate(f"IQR {q[75][i_mid] - q[25][i_mid]:.1f} m\nat the pinch",
                xy=(mid[i_mid], q[75][i_mid]), xytext=(0.46, 0.60),
                textcoords="axes fraction", fontsize=8.2, color=TEXT,
                ha="left", va="top", linespacing=1.4,
                arrowprops=dict(arrowstyle="->", color="#BBBBBB", lw=0.7,
                                shrinkA=2, shrinkB=3))
    style_axes(ax, xlabel="normalised path progress",
               ylabel="lateral offset [m]\n(> 0 = to the right)",
               equal=False)
    ax.set_title("先散开，再收拢",
                 fontsize=10, color=TEXT, pad=8)
    ax.legend(frameon=False, fontsize=8.3, loc="lower left")

    # (b) 偏航角剖面
    ax = axes[1]
    mm = m & ~np.isnan(dev)
    dmid, dq, _ = band_profile(np.tile(prog, (A, 1)).ravel(order="F")[mm.ravel(order="F")],
                               dev[mm], edges)
    ax.fill_between(dmid, dq[25], dq[75], color=BASE, alpha=0.22, lw=0,
                    label="25–75 %")
    ax.fill_between(dmid, dq[10], dq[90], color=BASE, alpha=0.13, lw=0,
                    label="10–90 %")
    ax.plot(dmid, dq[50], color=ACCENT, lw=1.8, label="median")
    ax.axhline(0, color="#9A9A9A", lw=0.9, ls=":")
    for y, lab in ((20, "±20°"), (-20, "±20°")):
        ax.axhline(y, color=GRID, ls=":", lw=0.9)
    style_axes(ax, xlabel="normalised path progress",
               ylabel="angle to the direct line [deg]\n(> 0 = veering right)",
               equal=False)
    ax.set_ylim(-60, 60)
    ax.set_title("朝向与直线的夹角",
                 fontsize=10, color=TEXT, pad=8)
    ax.legend(frameon=False, fontsize=8.3, loc="upper left")
    frac = float(np.mean(np.abs(dev[mm]) < 20))
    annotate(ax, f"only {frac:.0%} of travel samples\nstay within 20° of the "
                 f"straight line", "lower right")

    # (c) 过中心的侧别
    ax = axes[2]
    order = np.argsort(pass_lat)
    y = np.arange(A)
    cols = np.where(pass_lat[order] > 0, BASE, ACCENT)
    ax.barh(y, pass_lat[order], color=cols, alpha=0.85, height=0.85)
    ax.axvline(0, color="#666666", lw=0.9)
    ax.set_ylim(-1, A)
    style_axes(ax, xlabel="lateral offset when passing the centre [m]",
               ylabel="pedestrians (sorted)", equal=False)
    ax.set_title("从哪一侧挤过中心",
                 fontsize=10, color=TEXT, pad=8)
    handles = [Line2D([], [], marker="s", ls="none", ms=7, color=BASE,
                      label=f"right of travel: {n_right}"),
               Line2D([], [], marker="s", ls="none", ms=7, color=ACCENT,
                      label=f"left of travel: {A - n_right}")]
    ax.legend(handles=handles, frameon=False, fontsize=8.3, loc="lower right")
    annotate(ax, f"binomial p = {pval:.4f}", "upper left")

    save(fig, outdir, "fig6_heading", dpi, root=ROOT)
    plt.close(fig)

    return {"pass_right": n_right, "pass_left": A - n_right,
            "binom_p": float(pval),
            "pass_lat_abs_median": float(np.median(np.abs(pass_lat))),
            "dev_abs_median": float(np.nanmedian(np.abs(dev[mm]))),
            "dev_p90": float(np.nanpercentile(np.abs(dev[mm]), 90)),
            "frac_within_20deg": frac,
            "turning_median": float(np.nanmedian(
                np.abs(tr.turning_rate(smooth)[mm])))}


# ------------------------------------------------------------------
# 文字报告
# ------------------------------------------------------------------

def report(tr: CircleTrial, smooth: int) -> None:
    S = tr.speeds(smooth)
    Sr = tr.speeds(1)
    m = tr.travel_mask()
    arr = tr.arrival_time()
    cnt = tr.neighbours(2.0)
    dev = tr.bearing_deviation(smooth)
    mm = m & ~np.isnan(dev)

    print(f"帧率 {tr.fps:g} fps，时长 {tr.duration:.1f} s，"
          f"速度估计：位置 {smooth} 帧滑动平均 + 中心差分")
    print()
    print("【速度】")
    print(f"  行进段速率   中位 {np.median(S[m]):.2f}   "
          f"p10 {np.percentile(S[m], 10):.2f}   p90 "
          f"{np.percentile(S[m], 90):.2f} m/s")
    iso = m & (cnt <= 0)
    print(f"  自由流速率   中位 {np.median(S[iso]):.2f} m/s  "
          f"(0 个 2 m 邻居，n={int(iso.sum())})")
    print(f"  拥堵段速率   中位 {np.median(S[m & (cnt >= 8)]):.2f} m/s  "
          f"(≥8 个 2 m 邻居，n={int((m & (cnt >= 8)).sum())})")
    print(f"  未平滑速率   中位 {np.median(Sr):.2f}   p99 "
          f"{np.percentile(Sr, 99):.2f}   最大 {Sr.max():.2f} m/s")
    print(f"  超过 3 m/s 的样本占比（未平滑）{np.mean(Sr > 3):.3f}")
    print(f"  全程静止样本占比（<0.3 m/s）{np.mean(S < 0.3):.3f}")
    print()
    print("【方向】")
    print(f"  相对直线偏航角  中位 |{np.nanmedian(np.abs(dev[mm])):.1f}|°   "
          f"p90 {np.nanpercentile(np.abs(dev[mm]), 90):.1f}°   "
          f"落在 ±20° 内的比例 {np.mean(np.abs(dev[mm]) < 20):.3f}")
    lat = tr.lateral()
    pl = np.array([lat[int(np.argmin(np.linalg.norm(tr.aligned()[:, i],
                                                    axis=1))), i]
                   for i in range(tr.n_agent)])
    n_r = int((pl > 0).sum())
    p = 2 * stats.binom.sf(max(n_r, tr.n_agent - n_r) - 1, tr.n_agent, 0.5)
    print(f"  过中心侧别      右侧 {n_r} / 左侧 {tr.n_agent - n_r} 人"
          f"   （二项检验 p = {p:.4f}）")
    print(f"  过中心横向偏移  中位 |{np.median(np.abs(pl)):.2f}| m   "
          f"p75 |{np.percentile(np.abs(pl), 75):.2f}| m")
    print(f"  转向率          中位 {np.nanmedian(np.abs(tr.turning_rate(smooth)[mm])):.1f} "
          f"deg/s   p90 "
          f"{np.nanpercentile(np.abs(tr.turning_rate(smooth)[mm]), 90):.1f}")
    print()
    print("【拥堵与到达】")
    print(f"  速率—邻居数 Spearman ρ = "
          f"{stats.spearmanr(cnt[m], S[m]).statistic:.3f}")
    nd = tr.nearest_distance()
    print(f"  全场最近两人间距  中位 {np.median(nd):.2f} m   "
          f"最小 {nd.min():.2f} m（第 {int(np.argmin(nd))} 帧）")
    print(f"  到达时间  中位 {np.median(arr):.1f} s   "
          f"IQR {np.percentile(arr, 25):.1f}–{np.percentile(arr, 75):.1f} s   "
          f"最晚 {arr.max():.1f} s")
    print(f"  到达时间 p10–p90 跨度 {np.percentile(arr, 90) - np.percentile(arr, 10):.1f} s")


def main() -> int:
    ap = argparse.ArgumentParser(description="week02 圆环对趾：速度与方向分析")
    ap.add_argument("--data", default=str(default_data_path()))
    ap.add_argument("--outdir", default=str(HERE / "results"))
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--smooth", type=int, default=5,
                    help="速度估计前对位置做的滑动平均帧数（1 = 不平滑）")
    args = ap.parse_args()

    tr = load(Path(args.data))
    outdir = Path(args.outdir)
    report(tr, args.smooth)
    print()
    fig_speed(tr, outdir, args.dpi, args.smooth)
    fig_congestion(tr, outdir, args.dpi, args.smooth)
    fig_heading(tr, outdir, args.dpi, args.smooth)
    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
