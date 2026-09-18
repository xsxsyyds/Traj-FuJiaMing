#!/usr/bin/env python3
"""week02 · 步骤① 形成行为判断 —— 圆环对趾实验的轨迹可视化。

数据格式
--------
`data/circle-10m-64-1.txt`，5 列纯文本，空格分隔：

    agent  frame  x  y  unknown

- ``agent``    行人编号 1..64
- ``frame``    帧号 37..461（每人恰好 425 帧，等时间采样）
- ``x, y``     位置，单位 cm（圆周半径约 1000 cm = 10 m）
- ``unknown``  每个行人为定值，取 160/170/180（分别 21/29/14 人），
               与行走方向不对应，本脚本不使用

⚠️ 列的含义极易读反：**第 2 列不是行人 ID，而是帧号**。只有固定第 1 列、
让第 2 列递增，轨迹才会从圆周一端平滑穿过圆心走到对侧。

输出的图
--------
- ``results/fig1_trajectories_raw.png``      原始轨迹（各行人在自己的起始方位上）
- ``results/fig2_trajectories_aligned.png``  绕圆心旋转对齐后的轨迹（起点终点重合）
- ``results/fig3_geometry_stats.png``        几何统计：半径剖面、步长剖面、迂回程度

用法
----
    python week02/analyze_circle.py
    python week02/analyze_circle.py --data week02/data/circle-10m-64-1.txt
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from circle_data import (ACCENT, BASE, CYCLIC, GRID, TEXT, CircleTrial,
                         apply_style, default_data_path, draw_ring, load,
                         save)

apply_style()


def align(tr: CircleTrial) -> dict[int, np.ndarray]:
    """把每条轨迹绕圆心旋转 −θ₀，使所有起点方位角归零、终点落在对侧。

    只做纯旋转（保持半径不变），因此起点不会严格重合于一点——起点本身
    就分布在半径 9.8~10.4 m 的薄环上，这个残差本身也是信息。
    """
    al = tr.aligned()
    return {a: al[:, i] for i, a in enumerate(tr.agents)}


def progress(n: int) -> np.ndarray:
    """归一化路径进度 0..1。"""
    return np.linspace(0.0, 1.0, n)


# ------------------------------------------------------------------
# 绘图工具
# ------------------------------------------------------------------

def _save(fig, outdir: Path, stem: str, dpi: int) -> None:
    save(fig, outdir, stem, dpi, root=ROOT)


def style(ax, xlabel="x [m]", ylabel="y [m]", equal=True) -> None:
    from circle_data import style_axes
    style_axes(ax, xlabel, ylabel, equal)


def agents_colored(tr: CircleTrial):
    """按起始方位角给行人排序并返回 (行人列表, 颜色, 归一化器)。"""
    order = sorted(tr.tracks, key=lambda a: (tr.theta0(a) + np.pi) % (2 * np.pi))
    th = np.array([np.degrees(tr.theta0(a)) for a in order])
    norm = plt.Normalize(-180, 180)
    cmap = plt.get_cmap(CYCLIC)
    return order, cmap(norm(th)), norm


# ------------------------------------------------------------------
# 图 1：原始轨迹
# ------------------------------------------------------------------

def fig_raw(tr: CircleTrial, outdir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    draw_ring(ax)

    order, colors, norm = agents_colored(tr)
    for a, c in zip(order, colors):
        p = tr.tracks[a]
        ax.plot(p[:, 0], p[:, 1], color=c, lw=1.0, alpha=0.85,
                solid_capstyle="round", zorder=2)
    for a, c in zip(order, colors):
        p = tr.tracks[a]
        ax.plot(*p[0], marker="o", ms=3.2, color=c, mec="white", mew=0.4,
                zorder=3)
        ax.plot(*p[-1], marker="s", ms=3.0, color=c, mec="white", mew=0.4,
                zorder=3)

    ax.plot(0, 0, marker="+", ms=10, color=ACCENT, mew=1.4, zorder=4)

    style(ax)
    ax.set_title(f"圆环对趾实验：原始轨迹（旋转前）\n"
                 f"{tr.n_agent} 位行人，圆周半径 10 m",
                 fontsize=10.5, color=TEXT, pad=10)

    handles = [
        Line2D([], [], marker="o", ls="none", ms=4.5, mfc="#555555",
               mec="white", mew=0.5, label="start"),
        Line2D([], [], marker="s", ls="none", ms=4.5, mfc="#555555",
               mec="white", mew=0.5, label="goal"),
        Line2D([], [], marker="+", ls="none", ms=9, color=ACCENT, mew=1.5,
               label="circle centre"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.5,
              handletextpad=0.5, borderpad=0.2)

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(CYCLIC), norm=norm)
    cb = fig.colorbar(sm, ax=ax, fraction=0.036, pad=0.03)
    cb.set_label("start angle  [deg]", fontsize=9, color=TEXT)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=8)

    _save(fig, outdir, "fig1_trajectories_raw", dpi)
    plt.close(fig)


# ------------------------------------------------------------------
# 图 2：旋转对齐后的轨迹
# ------------------------------------------------------------------

def fig_aligned(tr: CircleTrial, outdir: Path, dpi: int) -> None:
    aligned = align(tr)
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    draw_ring(ax)

    order, colors, norm = agents_colored(tr)

    # 理想直线参考
    ax.plot([10, -10], [0, 0], ls=":", color="#B0B0B0", lw=1.1, zorder=1)

    for a, c in zip(order, colors):
        p = aligned[a]
        ax.plot(p[:, 0], p[:, 1], color=c, lw=1.0, alpha=0.55,
                solid_capstyle="round", zorder=2)

    # 横向偏移的 10–90% 包络与中位线，用来读出"人流在中点散开多宽"。
    # 直接按 x 分箱统计所有轨迹点，避免对非单调（绕圈）轨迹做插值。
    X = np.concatenate([aligned[a][:, 0] for a in order])
    Y = np.concatenate([aligned[a][:, 1] for a in order])
    edges = np.linspace(-10.0, 10.0, 81)
    mid = 0.5 * (edges[:-1] + edges[1:])
    which = np.digitize(X, edges)

    def band(q):
        out = np.full(mid.shape, np.nan)
        for i in range(1, len(edges)):
            sel = which == i
            if sel.sum() >= 8:
                out[i - 1] = np.percentile(Y[sel], q)
        # 轻度平滑，抑制分箱造成的锯齿
        k = 7
        ok = ~np.isnan(out)
        v = out[ok]
        if v.size > k:
            pad = k // 2
            out[ok] = np.convolve(np.pad(v, pad, mode="edge"),
                                  np.ones(k) / k, mode="valid")[:v.size]
        return out

    q10, q50, q90 = band(10), band(50), band(90)
    ax.fill_between(mid, q10, q90, color=BASE, alpha=0.12, lw=0, zorder=3,
                    label="10–90 % of trajectories")
    ax.plot(mid, q50, color=ACCENT, lw=1.7, zorder=6, label="median")

    ic = int(np.argmin(np.abs(mid)))
    w_mid = q90[ic] - q10[ic]
    print(f"      对齐后过 y 轴处的横向散布(10-90%) {w_mid:.2f} m")

    # 所有起点落在 (≈10, 0)，终点落在 (≈-10, 0)
    ax.plot(10, 0, marker="o", ms=7, color="#444444", mec="white", mew=0.8,
            zorder=5)
    ax.plot(-10, 0, marker="s", ms=7, color="#444444", mec="white", mew=0.8,
            zorder=5)
    ax.annotate("all starts\ncollapse here", xy=(10, 0), xytext=(7.0, 4.2),
                fontsize=8.5, color="#444444", ha="center", linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color="#999999", lw=0.7))
    ax.annotate("all goals\ncollapse here", xy=(-10, 0), xytext=(-6.8, 4.2),
                fontsize=8.5, color="#444444", ha="center", linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color="#999999", lw=0.7))

    ax.plot(0, 0, marker="+", ms=10, color=ACCENT, mew=1.4, zorder=4)

    style(ax)
    ax.set_title("绕圆心旋转对齐后\n"
                 "所有起点与终点重合",
                 fontsize=10.5, color=TEXT, pad=10)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5,
              handletextpad=0.6, borderpad=0.2,
              title=f"lateral spread on the y-axis:  {w_mid:.1f} m",
              title_fontsize=8.5)

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(CYCLIC), norm=norm)
    cb = fig.colorbar(sm, ax=ax, fraction=0.036, pad=0.03)
    cb.set_label("start angle  [deg]", fontsize=9, color=TEXT)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=8)

    _save(fig, outdir, "fig2_trajectories_aligned", dpi)
    plt.close(fig)


# ------------------------------------------------------------------
# 图 3：几何统计
# ------------------------------------------------------------------

def fig_stats(tr: CircleTrial, outdir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))
    order = sorted(tr.tracks)

    # (a) 半径剖面
    ax = axes[0]
    R_all = []
    for a in order:
        r = tr.radius(a)
        ax.plot(progress(len(r)), r, color=BASE, lw=0.7, alpha=0.22)
        R_all.append(np.interp(np.linspace(0, 1, 200),
                               progress(len(r)), r))
    Rm = np.median(np.array(R_all), axis=0)
    ax.plot(np.linspace(0, 1, 200), Rm, color=ACCENT, lw=1.8,
            label="median", zorder=5)
    style(ax, xlabel="normalised path progress", ylabel="distance to centre [m]",
          equal=False)
    ax.set_ylim(0, 11)
    ax.set_title("半径剖面", fontsize=10, color=TEXT, pad=8)
    ax.legend(frameon=False, fontsize=8.5)

    # (b) 最近圆心距离分布：谁从中心穿过、谁贴外圈绕行
    ax = axes[1]
    rmin = np.array([tr.r_min(a) for a in order])
    bins = np.arange(0, 10.5, 0.5)
    ax.hist(rmin, bins=bins, color=BASE, alpha=0.75, edgecolor="white",
            linewidth=0.6)
    ax.axvline(np.median(rmin), color=ACCENT, lw=1.8,
               label=f"median {np.median(rmin):.1f} m")
    ax.axvline(2.0, color="#7A7A7A", ls="--", lw=0.9)
    ax.text(2.15, ax.get_ylim()[1] * 0.55, "centre\ncorridor",
            fontsize=8, color="#7A7A7A", va="top", linespacing=1.3)
    style(ax, xlabel="closest approach to centre [m]",
          ylabel="number of pedestrians", equal=False)
    ax.set_title("路线选择：穿过圆心，还是绕行外圈",
                 fontsize=10, color=TEXT, pad=8)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    # (c) 迂回程度
    ax = axes[2]
    rmin = np.array([tr.r_min(a) for a in order])
    ratio = np.array([tr.path_length(a) / tr.net_length(a) for a in order])
    sc = ax.scatter(rmin, ratio, s=34, c=ratio, cmap="viridis",
                    edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(1.0, color="#999999", ls="--", lw=0.8)
    style(ax, xlabel="closest approach to centre [m]",
          ylabel="path length / straight-line distance", equal=False)
    ax.set_ylim(0.98, max(ratio) * 1.1)
    ax.set_title("迂回程度与离中心远近",
                 fontsize=10, color=TEXT, pad=8)
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("detour ratio", fontsize=8.5, color=TEXT)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=8)

    _save(fig, outdir, "fig3_geometry_stats", dpi)
    plt.close(fig)


# ------------------------------------------------------------------

def report(tr: CircleTrial) -> None:
    order = sorted(tr.tracks)
    th0 = np.array([np.degrees(tr.theta0(a)) for a in order])
    dth = np.array([(np.degrees(tr.theta0(a)) + 180) % 360 - 180 for a in order])
    r0 = np.array([tr.r_start_end(a)[0] for a in order])
    r1 = np.array([tr.r_start_end(a)[1] for a in order])
    rmin = np.array([tr.r_min(a) for a in order])
    L = np.array([tr.path_length(a) for a in order])
    net = np.array([tr.net_length(a) for a in order])
    ratio = L / net

    print(f"行人数 {tr.n_agent}   每轨迹采样点 {len(next(iter(tr.tracks.values())))}")
    print(f"起点半径   {r0.min():.2f} ~ {r0.max():.2f} m   中位 {np.median(r0):.2f}")
    print(f"终点半径   {r1.min():.2f} ~ {r1.max():.2f} m   中位 {np.median(r1):.2f}")
    print(f"最近圆心   {rmin.min():.2f} ~ {rmin.max():.2f} m   中位 {np.median(rmin):.2f}")
    print(f"首末直线   {net.min():.1f} ~ {net.max():.1f} m   中位 {np.median(net):.1f}  (理想 2R=20 m)")
    print(f"路径长度   {L.min():.1f} ~ {L.max():.1f} m   中位 {np.median(L):.1f}")
    print(f"迂回系数   {ratio.min():.3f} ~ {ratio.max():.3f}   中位 {np.median(ratio):.3f}")
    print(f"通过中心者(最近圆心<2m): {(rmin < 2).sum()} / {tr.n_agent}")
    print(f"贴边绕行者(最近圆心>5m): {(rmin > 5).sum()} / {tr.n_agent}")

    # 帧号等时间采样，故相邻帧位移正比于速度
    allstep = np.concatenate([tr.steps(a) for a in order])
    med = float(np.median(allstep))
    print(f"相邻帧步长 中位 {med:.4f} m/帧  最大 {allstep.max():.4f}")
    print(f"  按 {tr.fps:g} fps 折算   中位速度 {med * tr.fps:.2f} m/s"
          f"   最大 {allstep.max() * tr.fps:.2f} m/s")
    print("  （速度与方向的完整分析见 analyze_dynamics.py）")


def main() -> int:
    ap = argparse.ArgumentParser(description="week02 圆环对趾实验轨迹可视化")
    ap.add_argument("--data", default=str(ROOT / "week02" / "data"
                                          / "circle-10m-64-1.txt"))
    ap.add_argument("--outdir", default=str(ROOT / "week02" / "results"))
    ap.add_argument("--dpi", type=int, default=300, help="PNG 输出分辨率")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tr = load(Path(args.data))
    report(tr)
    print()
    fig_raw(tr, outdir, args.dpi)
    fig_aligned(tr, outdir, args.dpi)
    fig_stats(tr, outdir, args.dpi)
    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
