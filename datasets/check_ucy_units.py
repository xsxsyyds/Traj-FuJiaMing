#!/usr/bin/env python3
"""验证 UCY 原始标注 (`.vsp`) 的坐标单位。

输出三条证据并绘制与米制转换版的对比图：

1. 坐标范围是否等于视频像素边界（720x576 居中）
2. 行人 y 位置与像素速度是否显著负相关（透视压缩的特征）
3. 按像素当厘米换算得到的步速是否合理

用法
----
    python datasets/check_ucy_units.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
UCY = ROOT / "ucy"
FIGDIR = ROOT / "figures"

FPS = 25.0
VIDEO_W, VIDEO_H = 720, 576

SCENES = {
    "zara01": "crowds_zara01",
    "zara02": "crowds_zara02",
    "students03": "students003",
}

BASE = "#1F4E79"
GRID = "#D9D9D9"


def parse_vsp(path: Path) -> list[np.ndarray]:
    """解析 `.vsp`，返回每条样条的 ``(k, 4)`` 数组 [x, y, frame, gaze]。

    首行声明样条总数；每个样条由「控制点数」一行加若干控制点行组成。
    `students003.vsp` 在样条之后还有障碍物段落，靠首行声明的样条数界定范围。
    """
    raw = [ln.split(" - ")[0].strip() for ln in path.read_text().splitlines()]
    raw = [ln for ln in raw if ln]
    n_declared = int(float(raw[0].split()[0]))

    splines: list[np.ndarray] = []
    i = 1
    while i < len(raw) and len(splines) < n_declared:
        head = raw[i].split()
        if len(head) != 1:
            break
        k = int(float(head[0]))
        block = raw[i + 1: i + 1 + k]
        if len(block) < k or any(len(b.split()) != 4 for b in block):
            break  # 进入障碍物等其他段落
        splines.append(np.array([[float(t) for t in b.split()] for b in block]))
        i += 1 + k

    if len(splines) != n_declared:
        print(f"  [警告] {path.name} 声明 {n_declared} 条样条，实际解析 {len(splines)} 条")
    return splines


def check(scene: str, stem: str) -> dict:
    vsp = UCY / scene / f"{stem}.vsp"
    splines = parse_vsp(vsp)
    pts = np.vstack(splines)

    # 证据 1：坐标范围 vs 视频边界
    xlim = (-VIDEO_W / 2, VIDEO_W / 2)
    ylim = (-VIDEO_H / 2, VIDEO_H / 2)
    in_x = xlim[0] - 30 <= pts[:, 0].min() and pts[:, 0].max() <= xlim[1] + 30
    in_y = ylim[0] - 30 <= pts[:, 1].min() and pts[:, 1].max() <= ylim[1] + 30

    # 证据 2：y 位置 与 像素速度 的相关性
    ys, vs = [], []
    for a in splines:
        if len(a) < 3:
            continue
        d = np.linalg.norm(np.diff(a[:, :2], axis=0), axis=1)
        dt = np.diff(a[:, 2]) / FPS
        m = dt > 0
        if m.sum() < 2:
            continue
        ys.append(a[:, 1].mean())
        vs.append(float(np.median(d[m] / dt[m])))
    rho, p = spearmanr(ys, vs)

    # 证据 3：按 cm 解释的步速
    allv = []
    for a in splines:
        if len(a) < 2:
            continue
        d = np.linalg.norm(np.diff(a[:, :2], axis=0), axis=1)
        dt = np.diff(a[:, 2]) / FPS
        m = dt > 0
        allv.extend((d[m] / dt[m]).tolist())
    med_px = float(np.median(allv))

    return dict(scene=scene, n_ped=len(splines), n_pt=len(pts),
                bounds=(pts[:, 0].min(), pts[:, 0].max(),
                        pts[:, 1].min(), pts[:, 1].max()),
                in_frame=in_x and in_y, rho=rho, p=p,
                med_px=med_px, med_as_cm=med_px / 100.0)


def main() -> int:
    print("UCY `.vsp` 坐标单位检验")
    print("=" * 62)
    print(f"参考：视频 720x576，居中后应为 x∈[-360,360]  y∈[-288,288]\n")

    results = []
    for scene, stem in SCENES.items():
        r = check(scene, stem)
        results.append(r)
        b = r["bounds"]
        print(f"[{scene}]")
        print(f"  坐标范围  x [{b[0]:7.1f}, {b[1]:7.1f}]   "
              f"y [{b[2]:7.1f}, {b[3]:7.1f}]   "
              f"{'落在视频边界内 ✓' if r['in_frame'] else '超出边界 ✗'}")
        print(f"  透视检验  y 位置 vs 像素速度  Spearman rho = {r['rho']:+.3f}  "
              f"p = {r['p']:.1e}  {'显著负相关 ✓' if r['p'] < 0.01 else '不显著 ✗'}")
        print(f"  步速      中位 {r['med_px']:.1f} 单位/s  "
              f"→ 若按 cm 计为 {r['med_as_cm']:.2f} m/s "
              f"({'不合理，正常步速约 1.2-1.4 m/s' if r['med_as_cm'] < 0.9 else '合理'})")
        print()

    print("结论：`.vsp` 为图像像素坐标，不可直接用于物理量计算。")
    print("      请使用 build_ucy_metric.py 生成的米制 <场景>.txt。\n")

    # ---- 对比图 --------------------------------------------------
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    for a in parse_vsp(UCY / "zara01" / "crowds_zara01.vsp"):
        ax.plot(a[:, 0], a[:, 1], color=BASE, lw=0.7, alpha=0.35)
    ax.set_title("zara01 — raw .vsp (image pixels)", fontsize=10)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")

    ax = axes[1]
    metric = np.loadtxt(UCY / "zara01" / "zara01.txt")
    for pid in np.unique(metric[:, 1]):
        d = metric[metric[:, 1] == pid]
        ax.plot(d[:, 2], d[:, 3], color=BASE, lw=0.7, alpha=0.35)
    ax.set_title("zara01 — metric version (meters)", fontsize=10)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    ax = axes[2]
    for r in results:
        ax.scatter(r["med_px"] / 100.0, r["rho"], s=60, color=BASE, zorder=3)
        ax.annotate(r["scene"], (r["med_px"] / 100.0, r["rho"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.axvspan(1.2, 1.4, color="#C00000", alpha=0.10)
    ax.text(1.3, -0.46, "normal\nwalking", ha="center", va="center",
            fontsize=8, color="#C00000", linespacing=1.4)
    ax.axhline(0, color="#999999", lw=0.8, ls="--")
    ax.set_xlim(0, 2.0)
    ax.set_ylim(-0.55, 0.08)
    ax.set_title("if .vsp were centimetres", fontsize=10)
    ax.set_xlabel("implied median speed [m/s]")
    ax.set_ylabel("Spearman rho (y vs pixel speed)")

    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)

    fig.suptitle("UCY coordinate system check: .vsp is in image pixels, not metres",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    out = FIGDIR / "ucy_coordinate_check.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[写出] {out.relative_to(ROOT.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
