#!/usr/bin/env python3
"""A1 画轨迹 —— 每个个体一条线，输出各场景的轨迹图。

用法
----
    python week01/a1_trajectories.py                 # 全部 5 个场景
    python week01/a1_trajectories.py --scenes zara01 seq_eth
    python week01/a1_trajectories.py --color-by id   # 按行人编号着色

产物写入 ``week01/figures/``：每个场景一张图，外加一张 5 场景总览。
同时输出 PNG（预览）与 PDF（矢量，可直接用于报告）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.traj_io import SCENES, Scene, load_scene  # noqa: E402

# ---- 样式 ---------------------------------------------------------
BASE_COLOR = "#1F4E79"      # 深蓝，线条主色
ACCENT = "#C00000"          # 深红，用于强调与标注
GRID = "#D9D9D9"
TEXT = "#333333"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#666666",
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "savefig.bbox": "tight",
})


# ------------------------------------------------------------------
# 绘图
# ------------------------------------------------------------------

def _style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.9)
    ax.set_axisbelow(True)


def plot_scene(ax, sc: Scene, color_by: str = "uniform",
               lw: float = 0.7, alpha: float = 0.35) -> None:
    """在 ``ax`` 上画出该场景全部行人的轨迹，每个个体一条线。"""
    pids = sorted(sc.tracks)

    if color_by == "id":
        cmap = plt.get_cmap("turbo")
        denom = max(len(pids) - 1, 1)
        for i, pid in enumerate(pids):
            p = sc.tracks[pid]
            ax.plot(p[:, 1], p[:, 2], color=cmap(i / denom),
                    lw=lw, alpha=0.75, solid_capstyle="round")
    elif color_by == "length":
        lens = np.array([sc.track_length(p) for p in pids])
        lo, hi = np.percentile(lens, [5, 95])
        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(lo, max(hi, lo + 1e-6))
        for pid, L in zip(pids, lens):
            p = sc.tracks[pid]
            ax.plot(p[:, 1], p[:, 2], color=cmap(norm(L)),
                    lw=lw, alpha=0.85, solid_capstyle="round")
    else:  # uniform
        for pid in pids:
            p = sc.tracks[pid]
            ax.plot(p[:, 1], p[:, 2], color=BASE_COLOR,
                    lw=lw, alpha=alpha, solid_capstyle="round")

    _style_axes(ax)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    # 标题与副标题放在坐标区外，避免遮挡轨迹
    ax.set_title(f"{sc.name}   ({sc.dataset})", fontsize=10, color=TEXT, pad=22)
    ax.text(0.5, 1.015,
            f"{sc.n_ped} pedestrians   ·   {sc.duration:.0f} s",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.5, color="#777777")


def _figsize_for(sc: Scene, height: float = 4.6) -> tuple[float, float]:
    """按场景长宽比定尺寸，避免等比例下留大片空白。"""
    x0, x1, y0, y1 = sc.bounds()
    w = max(x1 - x0, 1e-6)
    h = max(y1 - y0, 1e-6)
    width = height * w / h
    return float(np.clip(width, 3.4, 8.5)), height


def plot_overview(scenes: list[Scene], outdir: Path, color_by: str,
                  dpi: int) -> None:
    """5 个场景 + 统计面板的总览图。"""
    n = len(scenes)
    ncol = 3
    nrow = int(np.ceil((n + 1) / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.2 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, sc in zip(axes, scenes):
        plot_scene(ax, sc, color_by=color_by, lw=0.55, alpha=0.32)

    # 最后一个格子放统计面板
    ax = axes[n]
    ax.axis("off")
    ax.set_title("summary", fontsize=10, color=TEXT, pad=8)

    rows = [("scene", "set", "N", "dur [s]", "span [m]")]
    for sc in scenes:
        x0, x1, y0, y1 = sc.bounds()
        rows.append((sc.name, sc.dataset, str(sc.n_ped),
                     f"{sc.duration:.0f}", f"{x1 - x0:.0f} x {y1 - y0:.0f}"))

    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   loc="upper center", cellLoc="center",
                   bbox=[0.0, 0.42, 1.0, 0.52])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.5)
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_facecolor("#E8EEF5")
            cell.set_text_props(color=BASE_COLOR, weight="bold")

    ax.text(0.5, 0.22,
            f"ETH + UCY  |  {sum(s.n_ped for s in scenes)} pedestrians  |  "
            f"{sum(s.n_point for s in scenes)} annotated points  |  25 fps",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=8.5, color=TEXT)
    ax.text(0.5, 0.08,
            "one line per pedestrian, ground-plane coordinates [m]",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=8, color="#777777")

    for ax in axes[n + 1:]:
        ax.axis("off")

    fig.suptitle("A1  Pedestrian trajectory overview — ETH / UCY",
                 fontsize=12, color=TEXT, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    for ext in ("png", "pdf"):
        p = outdir / f"a1_overview.{ext}"
        fig.savefig(p, dpi=dpi, facecolor="white")
        print(f"[写出] {p.relative_to(ROOT)}")
    plt.close(fig)


# ------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="A1 画轨迹：每个个体一条线")
    ap.add_argument("--scenes", nargs="+", default=list(SCENES),
                    choices=list(SCENES), help="要绘制的场景")
    ap.add_argument("--outdir", default=str(ROOT / "week01" / "figures"))
    ap.add_argument("--color-by", default="uniform",
                    choices=["uniform", "id", "length"],
                    help="线条着色方式")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--no-overview", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scenes = [load_scene(n) for n in args.scenes]

    for sc in scenes:
        fig, ax = plt.subplots(figsize=_figsize_for(sc))
        plot_scene(ax, sc, color_by=args.color_by)
        for ext in ("png", "pdf"):
            p = outdir / f"a1_traj_{sc.name}.{ext}"
            fig.savefig(p, dpi=args.dpi, facecolor="white")
            print(f"[写出] {p.relative_to(ROOT)}")
        plt.close(fig)

    if not args.no_overview and len(scenes) > 1:
        plot_overview(scenes, outdir, args.color_by, args.dpi)

    print(f"\n完成，共 {len(scenes)} 个场景。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
