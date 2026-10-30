# -*- coding: utf-8 -*-
"""
生成 mp4 动画：左 ORCA，右 Direct（无避让）。

输出 results/sphere_orca_vs_direct.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FFMpegWriter

import imageio_ffmpeg
from orca3d import region_colors

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

RADIUS = 10.0
def draw_sphere(ax):
    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(0, np.pi, 24)
    x = RADIUS * np.outer(np.cos(u), np.sin(v))
    y = RADIUS * np.outer(np.sin(u), np.sin(v))
    z = RADIUS * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="#D0D0D0", linewidth=0.45, alpha=0.22,
                      rstride=6, cstride=6)


def style(ax, title):
    ax.set_xlim(-RADIUS, RADIUS)
    ax.set_ylim(-RADIUS, RADIUS)
    ax.set_zlim(-RADIUS, RADIUS)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X [m]", labelpad=0)
    ax.set_ylabel("Y [m]", labelpad=0)
    ax.set_zlabel("Z [m]", labelpad=0)
    ax.set_title(title, fontsize=11)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=22, azim=38)


def main():
    zo = np.load(RESULTS / "traj_orca.npz")
    zd = np.load(RESULTS / "traj_direct.npz")
    traj_o, traj_d = np.asarray(zo["traj"]), np.asarray(zd["traj"])
    arrived_o, arrived_d = np.asarray(zo["arrived"]), np.asarray(zd["arrived"])
    To, Td = len(traj_o), len(traj_d)
    print("ORCA frames", To, " Direct frames", Td)

    fig = plt.figure(figsize=(11.2, 5.4))
    axes = [fig.add_subplot(1, 2, k, projection="3d") for k in (1, 2)]
    for ax in axes:
        draw_sphere(ax)
    style(axes[0], "ORCA（带避让）")
    style(axes[1], "Direct（无避让）")

    # 颜色由起始球面位置决定，并在整个仿真过程中固定。
    # 方位角决定色相，纬度决定明暗，因此相邻球面区域自然形成渐变。
    colors = np.asarray(zo["colors"]) if "colors" in zo else region_colors(traj_o[0])
    scat_o = axes[0].scatter(traj_o[0][:, 0], traj_o[0][:, 1],
                             traj_o[0][:, 2], s=22, color=colors,
                             edgecolor="white", linewidth=0.4, depthshade=False)
    scat_d = axes[1].scatter(traj_d[0][:, 0], traj_d[0][:, 1],
                             traj_d[0][:, 2], s=22, color=colors,
                             edgecolor="white", linewidth=0.4, depthshade=False)
    # 每架无人机一条独立轨迹线；线条与机体使用同一个固定颜色。
    lines_o = [axes[0].plot([], [], [], color=colors[i], lw=0.65,
                            alpha=0.78)[0] for i in range(traj_o.shape[1])]
    lines_d = [axes[1].plot([], [], [], color=colors[i], lw=0.65,
                            alpha=0.78)[0] for i in range(traj_d.shape[1])]
    txt_o = axes[0].text2D(0.02, 0.95, "", transform=axes[0].transAxes,
                           fontsize=10, color="#333333", fontweight="bold")
    txt_d = axes[1].text2D(0.02, 0.95, "", transform=axes[1].transAxes,
                           fontsize=10, color="#333333", fontweight="bold")
    fig.suptitle("三维球面对趾：100 架无人机（同一起点、同一目标）",
                 fontsize=13, y=0.98)

    out = RESULTS / "sphere_orca_vs_direct.mp4"
    writer = FFMpegWriter(fps=20, bitrate=2800)
    n = max(To, Td)
    with writer.saving(fig, str(out), dpi=150):
        for k in range(n):
            ko = min(k, To - 1)
            kd = min(k, Td - 1)
            po = traj_o[ko]
            pd = traj_d[kd]
            scat_o._offsets3d = (po[:, 0], po[:, 1], po[:, 2])
            scat_d._offsets3d = (pd[:, 0], pd[:, 1], pd[:, 2])
            # 显示历史轨迹，便于观察三维绕行和冲突解脱过程。
            for i, line in enumerate(lines_o):
                q = traj_o[:ko + 1, i]
                line.set_data_3d(q[:, 0], q[:, 1], q[:, 2])
            for i, line in enumerate(lines_d):
                q = traj_d[:kd + 1, i]
                line.set_data_3d(q[:, 0], q[:, 1], q[:, 2])
            t = k * 0.1
            ao = int(arrived_o[ko].sum())
            ad = int(arrived_d[kd].sum())
            txt_o.set_text(f"t = {t:4.1f} s   arrived {ao}/100")
            txt_d.set_text(f"t = {t:4.1f} s   arrived {ad}/100")
            writer.grab_frame()
            if k % 40 == 0:
                print(f"  frame {k}/{n}", flush=True)

    plt.close(fig)
    size = out.stat().st_size / 1e6
    print(f"写出 {out.name}  {size:.1f} MB")


if __name__ == "__main__":
    main()
