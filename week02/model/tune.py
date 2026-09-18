#!/usr/bin/env python3
"""week02 · 社会力模型参数筛选。

分两阶段：

- ``screen``  单因子扫描：在其他参数固定时逐个改 v⁰、A、B、τ，看清每个
  旋钮各自控制什么行为（"做什么"）。
- ``search``  在有希望的区域内做 v⁰ × A 网格，按与实测的偏差打分，给出
  一组可用的参数组合。

打分（越小越好）由四项归一化误差平均而来：

    e1  中位到达时间的相对偏差
    e2  中心拥堵时长（实测为 0）
    e3  横向散布 IQR 的相对偏差
    e4  过中心右侧人数的相对偏差

用法
----
    python week02/model/tune.py --stage screen
    python week02/model/tune.py --stage search
    python week02/model/tune.py --stage both
"""

from __future__ import annotations

import argparse
import json
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
sys.path.insert(0, str(ROOT))

from circle_data import (ACCENT, BASE, GRID, TEXT, apply_style,
                         default_data_path, load, save, style_axes)
from social_force import SFMParams, arrival_curve, centrality, simulate

apply_style()

GREEN = "#2A9D8F"
GREY = "#8C8C8C"

# 实测基线（由 analyze_dynamics.py / simulate.py 给出）
MEAS = {
    "t50": 12.8,          # 中位到达时间 [s]
    "jam": 0.0,           # 中心区 >80% 的时长 [s]
    "lat_iqr": 3.25,      # 行至各自半程时横向偏移的跨人 IQR [m]
    "right": 46,          # 过中心右侧人数
    "n": 64,
}


def evaluate(tr, params: SFMParams, duration: float,
             sample: int = 10) -> dict:
    """跑一次仿真并算出与实测的偏差。sample 为降采样步数。"""
    traj = simulate(tr.positions()[0], tr.goals(), params,
                    duration=duration, sample_every=sample)
    traj = traj[::max(1, int(round(0.04 / (params.dt * sample))))]
    t = np.arange(traj.shape[0]) * 0.04
    goal = tr.goals()
    arr = arrival_curve(traj, goal)
    cen = centrality(traj)
    idx = np.where(arr >= 32)[0]
    t50 = float(t[idx[0]]) if idx.size else float("nan")
    jam = float((cen > 0.8).sum() * 0.04)

    from social_force import to_trial
    tt = to_trial(traj, fps=25.0)
    iqr = tt.lateral_iqr()
    al = tt.aligned()
    lat = tt.lateral()
    pl = np.array([lat[int(np.argmin(np.linalg.norm(al[:, i], axis=1))), i]
                   for i in range(tt.n_agent)])
    right = int((pl > 0).sum())

    e1 = abs(t50 - MEAS["t50"]) / MEAS["t50"] if np.isfinite(t50) else 1.0
    e2 = jam / 10.0
    e3 = abs(iqr - MEAS["lat_iqr"]) / MEAS["lat_iqr"]
    e4 = abs(right - MEAS["right"]) / MEAS["right"]

    # 额外记录：模型的最高速率（用于判断参数是否已经不物理）
    sp = tt.speeds(1)
    return {
        "t50": t50, "t100": float(t[np.where(arr >= 64)[0][0]])
        if (arr >= 64).any() else float("nan"),
        "jam": jam, "lat_iqr": iqr, "right": right,
        "vmax": float(sp.max()),
        "e1": e1, "e2": e2, "e3": e3, "e4": e4,
        "score": float(np.mean([e1, e2, e3, e4])),
        "arrived": int(arr[-1]),
    }


# ------------------------------------------------------------------
# 阶段一：单因子扫描
# ------------------------------------------------------------------

KNOBS = [
    ("v0", "desired speed v⁰ [m/s]", [1.5, 2.0, 2.5, 3.0, 3.5]),
    ("A", "repulsion strength A [N]", [500, 1000, 2000, 4000, 8000]),
    ("B", "repulsion range B [m]", [0.04, 0.08, 0.12, 0.20, 0.30]),
    ("tau", "relaxation time τ [s]", [0.15, 0.3, 0.5, 1.0, 2.0]),
]


def screen(tr, duration: float) -> list[dict]:
    rows = []
    for key, label, values in KNOBS:
        for v in values:
            r = evaluate(tr, SFMParams(**{key: v}), duration)
            rows.append({"knob": key, "label": label, "value": v, **r})
            print(f"  {key:<4} = {v:<6g} → t50 {r['t50']:5.1f}s  "
                  f"拥堵 {r['jam']:4.1f}s  IQR {r['lat_iqr']:.2f}m  "
                  f"右 {r['right']:2d}/64  到达 {r['arrived']:2d}  "
                  f"score {r['score']:.3f}")
    return rows


# ------------------------------------------------------------------
# 阶段二：v0 × A 网格
# ------------------------------------------------------------------

def search(tr, duration: float, base: dict) -> list[dict]:
    rows = []
    for v0 in (2.0, 2.5, 3.0, 3.5, 4.0):
        for A in (1000, 2000, 4000, 8000, 16000):
            p = SFMParams(v0=v0, A=A, **base)
            r = evaluate(tr, p, duration)
            rows.append({"v0": v0, "A": A, **r})
            print(f"  v⁰={v0:<4g} A={A:<6g} → t50 {r['t50']:5.1f}s  "
                  f"拥堵 {r['jam']:4.1f}s  IQR {r['lat_iqr']:.2f}m  "
                  f"右 {r['right']:2d}/64  到达 {r['arrived']:2d}  "
                  f"score {r['score']:.3f}")
    return rows


# ------------------------------------------------------------------

def fig_screen(rows: list[dict], outdir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    for ax, (key, label, values) in zip(axes.ravel(), KNOBS):
        sub = [r for r in rows if r["knob"] == key]
        x = np.array([r["value"] for r in sub], dtype=float)
        jam = np.array([r["jam"] for r in sub])
        t50 = np.array([r["t50"] for r in sub])
        ax.plot(x, jam, "o-", color=ACCENT, lw=1.7, ms=5, mec="white",
                mew=0.6, label="gridlock [s]")
        ax.axhline(MEAS["jam"], color=GREY, ls="--", lw=1.2,
                   label="measured 0 s")
        ax.set_xlabel(label, fontsize=8.8)
        ax.set_ylabel("gridlock at the centre [s]", color=ACCENT, fontsize=8.8)
        ax.tick_params(axis="y", colors=ACCENT)
        ax.set_xscale("log" if key in ("A", "B", "tau") else "linear")
        style_axes(ax, xlabel=label, ylabel="gridlock at the centre [s]",
                   equal=False)

        ax2 = ax.twinx()
        ax2.plot(x, t50, "s--", color=BASE, lw=1.5, ms=4.5, mec="white",
                 mew=0.6, label="half-time [s]")
        ax2.axhline(MEAS["t50"], color="#9DB2C6", ls=":", lw=1.2)
        ax2.set_ylabel("half-time arrival [s]", color=BASE, fontsize=8.8)
        ax2.tick_params(axis="y", colors=BASE)
        ax2.spines["top"].set_visible(False)

        bad = [r for r in sub if not np.isfinite(r["t100"])]
        for r in bad:
            ax.annotate("not all\narrive", xy=(r["value"], 0.35),
                        xytext=(r["value"], 2.4), fontsize=7.4,
                        color=GREY, ha="center", linespacing=1.3,
                        arrowprops=dict(arrowstyle="->", color=GREY, lw=0.6))
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8,
                  loc="upper center", ncol=2)

    fig.suptitle("One-at-a-time parameter screen  (thick red = gridlock, "
                 "blue = arrival half-time)", fontsize=11, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, outdir, "fig9_param_screen", dpi, root=ROOT.parent)
    plt.close(fig)


def fig_search(rows: list[dict], outdir: Path, dpi: int) -> None:
    v0s = sorted({r["v0"] for r in rows})
    As = sorted({r["A"] for r in rows})
    Z = np.full((len(v0s), len(As)), np.nan)
    J = np.full_like(Z, np.nan)
    for r in rows:
        i, j = v0s.index(r["v0"]), As.index(r["A"])
        Z[i, j] = r["score"]
        J[i, j] = r["jam"]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for ax, M, title, cmap, fmt in (
            (axes[0], Z, "mismatch score (lower is better)", "viridis_r",
             "%.2f"),
            (axes[1], J, "gridlock at the centre [s]", "magma_r", "%.1f")):
        im = ax.imshow(M, cmap=cmap, aspect="auto", origin="lower")
        ax.set_xticks(range(len(As)))
        ax.set_xticklabels([f"{a:g}" for a in As])
        ax.set_yticks(range(len(v0s)))
        ax.set_yticklabels([f"{v:g}" for v in v0s])
        ax.set_xlabel("repulsion strength A [N]", fontsize=9)
        ax.set_ylabel("desired speed v⁰ [m/s]", fontsize=9)
        ax.set_title(title, fontsize=10, color=TEXT, pad=8)
        for i in range(len(v0s)):
            for j in range(len(As)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, fmt % M[i, j], ha="center", va="center",
                            fontsize=8, color="white" if M[i, j] > np.nanmean(M)
                            else "#222222")
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cb.outline.set_linewidth(0.5)
    fig.suptitle("Two-knob search: how much of the gap can tuning close?",
                 fontsize=11, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, outdir, "fig10_param_search", dpi, root=ROOT.parent)
    plt.close(fig)


def refine(tr, duration: float) -> list[dict]:
    """在"还能讲得通"的参数区间里细化。

    可辩护的边界：期望速度不超过实测自由流的上四分位，排斥力强度不超过
    文献值的若干倍，作用尺度不超过身高的量级。
    """
    rows = []
    for v0 in (2.5, 2.8):
        for A in (8000, 12000, 16000, 24000):
            for B in (0.06, 0.08, 0.10):
                p = SFMParams(v0=v0, A=A, B=B)
                r = evaluate(tr, p, duration)
                rows.append({"v0": v0, "A": A, "B": B, **r})
                print(f"  v⁰={v0:<4g} A={A:<6g} B={B:<5g} → "
                      f"t50 {r['t50']:5.1f}s  拥堵 {r['jam']:4.1f}s  "
                      f"IQR {r['lat_iqr']:.2f}m  右 {r['right']:2d}/64  "
                      f"v_max {r['vmax']:.1f}  到达 {r['arrived']:2d}  "
                      f"score {r['score']:.3f}")
    return rows


# ------------------------------------------------------------------

def fig_refine(rows: list[dict], outdir: Path, dpi: int) -> None:
    """三张指标随 A 变化的曲线（每个 v⁰/B 一条），标出可选区间。"""
    combos = sorted({(r["v0"], r["B"]) for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.5))
    cmap = plt.get_cmap("viridis")
    for k, (v0, B) in enumerate(combos):
        sub = sorted([r for r in rows if r["v0"] == v0 and r["B"] == B],
                     key=lambda r: r["A"])
        x = [r["A"] for r in sub]
        col = cmap(k / max(1, len(combos) - 1))
        lab = f"v⁰={v0:g}, B={B:g}"
        axes[0].plot(x, [r["jam"] for r in sub], "o-", color=col, lw=1.5,
                     ms=4.5, mec="white", mew=0.5, label=lab)
        axes[1].plot(x, [r["t50"] for r in sub], "o-", color=col, lw=1.5,
                     ms=4.5, mec="white", mew=0.5)
        axes[2].plot(x, [r["arrived"] for r in sub], "o-", color=col, lw=1.5,
                     ms=4.5, mec="white", mew=0.5)
    for ax in axes:
        style_axes(ax, xlabel="repulsion strength A [N]",
                   ylabel="", equal=False)
        ax.set_xscale("log")
    axes[0].axhline(MEAS["jam"], color=GREY, ls="--", lw=1.3,
                    label="measured 0 s")
    axes[0].set_ylabel("gridlock at the centre [s]")
    axes[0].set_title("Gridlock can be tuned away…", fontsize=10, pad=8)
    axes[1].axhline(MEAS["t50"], color=GREY, ls="--", lw=1.3,
                    label="measured 12.8 s")
    axes[1].set_ylabel("half-time arrival [s]")
    axes[1].set_title("…but the arrival time stays late", fontsize=10, pad=8)
    axes[2].axhline(64, color=GREY, ls="--", lw=1.3, label="all 64")
    axes[2].set_ylabel("pedestrians arriving")
    axes[2].set_title("…and pushing too hard strands people", fontsize=10,
                      pad=8)
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    axes[1].legend(frameon=False, fontsize=8.4, loc="upper left")
    axes[2].legend(frameon=False, fontsize=8.4, loc="lower left")
    fig.suptitle("Refinement inside the defensible parameter range",
                 fontsize=11, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, outdir, "fig11_param_refine", dpi, root=ROOT.parent)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="week02 社会力模型参数筛选")
    ap.add_argument("--data", default=str(default_data_path()))
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    ap.add_argument("--duration", type=float, default=22.0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--stage",
                    choices=["screen", "search", "refine", "both", "all"],
                    default="all")
    args = ap.parse_args()

    tr = load(Path(args.data))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    stages = {"screen", "search", "refine"} if args.stage in ("all", "both") \
        else {args.stage}

    if "screen" in stages:
        print("阶段一 · 单因子扫描")
        rows = screen(tr, args.duration)
        summary["screen"] = rows
        fig_screen(rows, outdir, args.dpi)
        print()

    if "search" in stages:
        print("阶段二 · v⁰ × A 网格")
        rows2 = search(tr, args.duration, base={})
        summary["search"] = rows2
        fig_search(rows2, outdir, args.dpi)
        best = min(rows2, key=lambda r: r["score"])
        print(f"\n打分最优: v⁰={best['v0']:g} A={best['A']:g} "
              f"score={best['score']:.3f}  t50={best['t50']:.1f}s "
              f"拥堵={best['jam']:.1f}s  到达={best['arrived']}")
        summary["best_score"] = {k: best[k] for k in
                                 ("v0", "A", "t50", "t100", "jam", "lat_iqr",
                                  "right", "score", "arrived")}
        print()

    if "refine" in stages:
        print("阶段三 · 可辩护区间细化")
        rows3 = refine(tr, args.duration)
        summary["refine"] = rows3
        fig_refine(rows3, outdir, args.dpi)
        ok = [r for r in rows3 if r["arrived"] == 64]
        best = min(ok, key=lambda r: r["score"]) if ok else None
        if best:
            print(f"\n可辩护区间内最优: v⁰={best['v0']:g} A={best['A']:g} "
                  f"B={best['B']:g}  score={best['score']:.3f}  "
                  f"t50={best['t50']:.1f}s  拥堵={best['jam']:.1f}s  "
                  f"IQR={best['lat_iqr']:.2f}m  右={best['right']}/64")
            summary["best_defensible"] = {
                k: best[k] for k in ("v0", "A", "B", "t50", "t100", "jam",
                                     "lat_iqr", "right", "score", "vmax")}

    (outdir / "tuning.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[写出] {outdir / 'tuning.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
