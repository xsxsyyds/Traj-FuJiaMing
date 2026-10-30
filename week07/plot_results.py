# -*- coding: utf-8 -*-
"""
week07 结果图。全部 dpi=300 PNG（图题中文，坐标/图例英文）。

    fig1_trajectories_3d.png   三个模型的三维轨迹（球面）
    fig2_metrics.png           五项指标对比
    fig3_min_separation.png    最小间隔随时间变化
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "#D9D9D9"
plt.rcParams["grid.linewidth"] = 0.7
plt.rcParams["axes.axisbelow"] = True
plt.rcParams["axes.edgecolor"] = "#4D4D4D"
plt.rcParams["font.size"] = 10

MODELS = ["orca", "direct", "repulsion"]
LABEL = {"orca": "ORCA", "direct": "Direct", "repulsion": "Repulsion"}
COLOR = {"orca": "#1F4E79", "direct": "#C00000", "repulsion": "#2A9D8F"}
RADIUS = 10.0


def load(model):
    return np.load(RESULTS / f"traj_{model}.npz")


def draw_sphere(ax, radius=RADIUS, color="#C9C9C9", lw=0.6):
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color=color, linewidth=lw, alpha=0.25,
                      rstride=6, cstride=6)


def style3d(ax, title):
    ax.set_xlim(-RADIUS, RADIUS)
    ax.set_ylim(-RADIUS, RADIUS)
    ax.set_zlim(-RADIUS, RADIUS)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X [m]", labelpad=8)
    ax.set_ylabel("Y [m]", labelpad=10)
    ax.set_zlabel("")          # 3D 的 z 标签在最左侧面板会被裁掉，改为全图统一标注
    ax.set_title(title, fontsize=11, pad=2)
    ax.tick_params(labelsize=6.5, pad=1)
    ax.view_init(elev=22, azim=38)


# ==========================================================================
def fig1():
    fig = plt.figure(figsize=(16.0, 4.9))
    for k, m in enumerate(MODELS, start=1):
        ax = fig.add_subplot(1, 3, k, projection="3d")
        draw_sphere(ax)
        z = load(m)
        traj = z["traj"]
        zcolors = z["colors"] if "colors" in z else [COLOR[m]] * traj.shape[1]
        for i in range(traj.shape[1]):
            ax.plot(traj[:, i, 0], traj[:, i, 1], traj[:, i, 2],
                    color=zcolors[i], lw=0.45, alpha=0.48)
        ax.scatter(traj[0, :, 0], traj[0, :, 1], traj[0, :, 2],
                   s=6, color="#555555", depthshade=False)
        ax.scatter(traj[-1, :, 0], traj[-1, :, 1], traj[-1, :, 2],
                   s=10, color=zcolors, edgecolor="white", linewidth=0.3,
                   depthshade=False)
        style3d(ax, f"{LABEL[m]}（{traj.shape[1]} 架）")
    fig.subplots_adjust(left=0.085, right=0.955, bottom=0.06, top=0.88,
                        wspace=0.0)
    fig.text(0.038, 0.47, "Z [m]", rotation=90, va="center", ha="center",
             fontsize=9)
    fig.suptitle("三维球面对趾：三个模型的轨迹（灰点 = 起点，彩点 = 终点）",
                 fontsize=13.5, y=0.98)
    fig.savefig(RESULTS / "fig1_trajectories_3d.png")
    plt.close(fig)
    print("写出 fig1_trajectories_3d.png")


# ==========================================================================
def fig2():
    ms = {m: json.loads((RESULTS / f"metrics_{m}.json").read_text(encoding="utf-8"))
          for m in MODELS}
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.1))
    x = np.arange(len(MODELS))
    colors = [COLOR[m] for m in MODELS]
    labels = [LABEL[m] for m in MODELS]

    def bars(ax, vals, title, ylabel, fmt="{:.2f}", hline=None):
        b = ax.bar(x, vals, 0.55, color=colors, edgecolor="white", linewidth=0.9)
        ax.bar_label(b, fmt=fmt, fontsize=9, padding=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        if hline is not None:
            ax.axhline(hline, color="#B8860B", lw=1.2, ls=":")
        return b

    bars(axes[0], [ms[m]["arrival_rate"] * 100 for m in MODELS],
         "到达率", "Arrival rate [%]", "{:.0f}%")

    bars(axes[1], [ms[m]["min_separation"] for m in MODELS],
         "最小间隔", "Minimum separation [m]", "{:.3f}", hline=1.0)
    axes[1].text(2.35, 1.02, "required 1.0 m", fontsize=7.5,
                 color="#8B6508", ha="right")

    bars(axes[2], [ms[m]["violating_pairs"] for m in MODELS],
         "侵犯间隔的机对数", "Pairs violating 1 m", "{:.0f}")
    axes[2].set_ylim(0, max(ms[m]["violating_pairs"] for m in MODELS) * 1.18 + 1)

    tvals = [ms[m]["t_all_arrive"] if ms[m]["t_all_arrive"] is not None
             else ms[m]["t_end"] for m in MODELS]
    bars(axes[3], tvals, "全部到达时间", "Time [s]", "{:.1f}")

    fig.suptitle("三维球面对趾：100 架无人机的指标对比（安全间隔 1 m）",
                 fontsize=13.5, y=1.03)
    fig.savefig(RESULTS / "fig2_metrics.png")
    plt.close(fig)
    print("写出 fig2_metrics.png")


# ==========================================================================
def fig3():
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for m in MODELS:
        z = load(m)
        traj, t = z["traj"], z["t"]
        mins = []
        for k in range(len(traj)):
            d = np.linalg.norm(traj[k][:, None, :] - traj[k][None, :, :], axis=2)
            np.fill_diagonal(d, np.inf)
            mins.append(d.min())
        ax.plot(t, mins, color=COLOR[m], lw=1.8, label=LABEL[m])
    ax.axhline(1.0, color="#B8860B", lw=1.3, ls=":")
    ax.text(24.3, 1.06, "required separation 1.0 m", fontsize=8,
            color="#8B6508", ha="right")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Minimum pairwise separation [m]")
    ax.set_ylim(0, 6)
    ax.set_xlim(0, 25)
    ax.legend(fontsize=9, loc="upper right")
    fig.suptitle("三类模型在冲突最密集阶段的最近距离", fontsize=13.5, y=1.02)
    fig.savefig(RESULTS / "fig3_min_separation.png")
    plt.close(fig)
    print("写出 fig3_min_separation.png")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
