#!/usr/bin/env python3
"""week02 · 步骤③ 实现与验证 —— 用社会力模型复现圆环对趾。

做法
----
仿真从**实测的初始位置**出发（而不是自己造一个均匀圆环），目标取实测
起点的对趾点，然后看模型能否走完实测那条路。这样比较的是"同样的初始
条件、同样的目标下，模型的行为与真人差在哪"，而不是"两个不同实验的
统计量差在哪"。

参数预设
--------
- ``literature``  Helbing 原始值（A=2000 N、B=0.08 m、τ=0.5 s），
                  期望速度按实测自由流标定为 2.0 m/s
- ``tuned``       在文献值基础上按 tune.py 的筛选结果调整（默认）

所有参数都可以在命令行上单独覆盖：

    python week02/model/simulate.py --set v0=2.5 --set A=8000

三种力配置
----------
- 默认            课程要求的三项力，排斥力各向同性
- ``front_only``  只保留来自前半平面的排斥力（失效分析探针，非三项力之一）

用法
----
    python week02/model/simulate.py                  # 对比 + 失效分析
    python week02/model/simulate.py --preset literature
    python week02/model/simulate.py --set v0=3.0
    python week02/model/simulate.py --sweep          # 附加参数扫描
    python week02/model/simulate.py --gif            # 另存全程动图 GIF
    python week02/model/simulate.py --only gif       # 只重做动图
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
sys.path.insert(0, str(ROOT))

from circle_data import (ACCENT, BASE, GRID, TEXT, apply_style,
                         default_data_path, draw_ring, load, save, style_axes)
from social_force import (PRESETS, SFMParams, arrival_curve, centrality,
                          min_pair_distance, simulate, to_trial)

apply_style()

GREEN = "#2A9D8F"
GREY = "#8C8C8C"


# ------------------------------------------------------------------
# 参数：支持 --preset 与 --set key=value 覆盖
# ------------------------------------------------------------------

def build_params(preset: str, overrides: list[str] | None) -> SFMParams:
    p = PRESETS[preset]()
    for item in overrides or []:
        if "=" not in item:
            raise SystemExit(f"--set 需要 key=value 形式，收到 {item!r}")
        key, val = item.split("=", 1)
        key = key.strip()
        if not hasattr(p, key):
            raise SystemExit(
                f"未知参数 {key!r}。可调参数："
                + ", ".join(sorted(p.__dataclass_fields__)))
        cur = getattr(p, key)
        setattr(p, key,
                bool(int(val)) if isinstance(cur, bool) else type(cur)(val))
    return p


def describe_params(p: SFMParams) -> str:
    return (f"A={p.A:g} N, B={p.B:g} m, τ={p.tau:g} s, v⁰={p.v0:g} m/s, "
            f"k={p.k:g}, κ={p.kappa:g}, m={p.mass:g} kg, r={p.radius:g} m"
            + (", front-only" if p.front_only else ""))


# ------------------------------------------------------------------
# 运行与派生量
# ------------------------------------------------------------------

def run_config(tr, params: SFMParams, duration: float,
               sample: int = 20) -> dict:
    """从实测初始条件仿真，返回轨迹与派生量。"""
    pos0 = tr.positions()[0]
    goal = tr.goals()
    traj = simulate(pos0, goal, params, duration=duration, sample_every=sample)
    t = np.arange(traj.shape[0]) * (params.dt * sample)
    return {
        "params": params,
        "traj": traj,
        "t": t,
        "goal": goal,
        "arrivals": arrival_curve(traj, goal),
        "central": centrality(traj),
        "minpair": min_pair_distance(traj),
        "radius": np.linalg.norm(traj, axis=2),
    }


def measured_centrality(tr) -> np.ndarray:
    return (np.linalg.norm(tr.positions(), axis=2) < 3.0).mean(axis=1)


def jam_duration(central: np.ndarray, dt: float, thresh: float = 0.8) -> float:
    """中心区人数占比超过 thresh 的累计时长 [s]。"""
    return float((central > thresh).sum() * dt)


def lateral_spread(trial) -> float:
    """行至各自半程时横向偏移的跨人 IQR [m]，与轨迹时长无关。

    因此实测（17 s）与仿真（25 s）可以直接比较；定义见
    ``circle_data.CircleTrial.lateral_iqr``。
    """
    return trial.lateral_iqr()


def pass_sides(trial) -> tuple[int, int]:
    """(右侧人数, 左侧人数)：按最近圆心那一帧的横向偏移定侧别。"""
    al = trial.aligned()
    lat = trial.lateral()
    pl = np.array([lat[int(np.argmin(np.linalg.norm(al[:, i], axis=1))), i]
                   for i in range(trial.n_agent)])
    return int((pl > 0).sum()), int((pl <= 0).sum())


def time_at(arrivals: np.ndarray, t: np.ndarray, k: int) -> float:
    idx = np.where(arrivals >= k)[0]
    return float(t[idx[0]]) if idx.size else float("nan")


# ------------------------------------------------------------------
# 图 7：模型 vs 实测（含旋转之前的原始轨迹）
# ------------------------------------------------------------------

def _align(arr: np.ndarray, th0: np.ndarray) -> np.ndarray:
    """把轨迹绕圆心旋转 −θ₀，使所有起点落到 (+r, 0)。"""
    a = np.empty_like(arr)
    for i in range(arr.shape[1]):
        c, s = np.cos(th0[i]), np.sin(th0[i])
        a[:, i, 0] = arr[:, i, 0] * c + arr[:, i, 1] * s
        a[:, i, 1] = -arr[:, i, 0] * s + arr[:, i, 1] * c
    return a


def _panel_world(ax, arr, color, title, alpha=0.5, lw=0.7) -> None:
    """世界系（旋转之前）的轨迹。"""
    draw_ring(ax)
    for i in range(arr.shape[1]):
        ax.plot(arr[:, i, 0], arr[:, i, 1], lw=lw, alpha=alpha, color=color)
    ax.plot(arr[0, :, 0], arr[0, :, 1], "o", ms=2.6, color=color,
            mec="white", mew=0.25, zorder=4)
    ax.plot(0, 0, "+", ms=9, color=ACCENT, mew=1.3, zorder=5)
    style_axes(ax)
    ax.set_title(title, fontsize=10, color=TEXT, pad=8)


def _panel_aligned(ax, arr, color, title) -> None:
    draw_ring(ax)
    for i in range(arr.shape[1]):
        ax.plot(arr[:, i, 0], arr[:, i, 1], lw=0.7, alpha=0.5, color=color)
    ax.axhline(0, color="#B8B8B8", lw=0.8, ls=":")
    ax.plot(10, 0, "o", ms=6, color="#444444", mec="white", mew=0.7, zorder=5)
    ax.plot(-10, 0, "s", ms=6, color="#444444", mec="white", mew=0.7,
            zorder=5)
    ax.plot(0, 0, "+", ms=9, color=ACCENT, mew=1.3, zorder=5)
    style_axes(ax)
    ax.set_title(title, fontsize=10, color=TEXT, pad=8)


def fig_compare(tr, sim: dict, params, outdir: Path, dpi: int) -> None:
    th0 = np.arctan2(tr.positions()[0][:, 1], tr.positions()[0][:, 0])
    t, traj = sim["t"], sim["traj"]
    meas_raw, sim_raw = tr.positions(), traj
    meas_al, sim_al = _align(meas_raw, th0), _align(sim_raw, th0)

    fig, axes = plt.subplots(2, 3, figsize=(11.8, 10.6))
    (axA, axB, axC), (axD, axE, axF) = axes

    # ---- 上排：世界系（旋转之前）----
    _panel_world(axA, meas_raw, BASE, "Measured — before rotation")
    _panel_world(axB, sim_raw, ACCENT, "Model — before rotation")
    _panel_world(axC, meas_raw, GREY, "Overlay — before rotation",
                 alpha=0.30, lw=0.6)
    for i in range(sim_raw.shape[1]):
        axC.plot(sim_raw[:, i, 0], sim_raw[:, i, 1], lw=0.7, alpha=0.5,
                 color=ACCENT)
    axC.legend(handles=[Line2D([], [], color=GREY, lw=1.2, label="measured"),
                        Line2D([], [], color=ACCENT, lw=1.2, label="model")],
               frameon=False, fontsize=8.3, loc="upper left")

    # ---- 下排：对齐系 ----
    _panel_aligned(axD, meas_al, BASE, "Measured — starts aligned")
    _panel_aligned(axE, sim_al, ACCENT, "Model — starts aligned")

    # ---- 累计到达 ----
    ax = axF
    arr_m = np.sort(tr.arrival_time())
    ax.plot(arr_m, np.arange(1, tr.n_agent + 1), color=BASE, lw=1.7,
            label="measured")
    ax.plot(t, sim["arrivals"], color=ACCENT, lw=1.7, label="model")
    ax.set_xlim(0, max(tr.duration, t[-1]))
    ax.set_ylim(0, 68)
    style_axes(ax, xlabel="time [s]", ylabel="cumulative arrivals [of 64]",
               equal=False)
    ax.set_title("Arrivals", fontsize=10, color=TEXT, pad=8)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    td = time_at(sim["arrivals"], t, 32)
    tm = float(np.median(tr.arrival_time()))
    ax.axvline(tm, color=GREY, ls="--", lw=1.0)
    ax.axvline(td, color=ACCENT, ls="--", lw=1.0)
    ax.text(tm - 0.5, 46, "measured\nhalf-time", fontsize=8, color=GREY,
            ha="right", linespacing=1.35)
    ax.text(td + 0.5, 27, "model\nhalf-time", fontsize=8, color=ACCENT,
            ha="left", linespacing=1.35)

    fig.suptitle("Social-force model vs. the measured ring experiment\n"
                 f"preset «{params_name(params)}»: {describe_params(params)}",
                 fontsize=11, color=TEXT, y=0.995, linespacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save(fig, outdir, "fig7_sfm_vs_measured", dpi, root=ROOT.parent)
    plt.close(fig)


def params_name(p: SFMParams) -> str:
    return "tuned" if (p.v0, p.A, p.B) == (PRESETS["tuned"]().v0,
                                          PRESETS["tuned"]().A,
                                          PRESETS["tuned"]().B) else "custom"


# ------------------------------------------------------------------
# 图 8：失效分析
# ------------------------------------------------------------------

def fig_failure(tr, lit: dict, tuned: dict, front: dict, sweeps: dict | None,
                params, outdir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 9.4),
                             gridspec_kw=dict(width_ratios=[1.0, 1.15]))
    (axA, axB), (axC, axD) = axes

    # (a) 半径剖面
    ax = axA
    tc = np.arange(tr.n_frame) * tr.dt
    ax.plot(tc, np.median(np.linalg.norm(tr.positions(), axis=2), axis=1),
            color=BASE, lw=1.7, label="measured")
    for k, r, col in (("literature", lit, ACCENT), ("tuned", tuned, GREEN)):
        ax.plot(r["t"], np.median(r["radius"], axis=1), color=col, lw=1.6,
                ls="-" if k == "literature" else "--", label=f"{k} preset")
    ax.axhspan(0, 3.0, color=ACCENT, alpha=0.06, lw=0)
    ax.text(0.6, 1.3, "centre zone", fontsize=8, color="#9A8A8A", ha="left")
    style_axes(ax, xlabel="time [s]",
               ylabel="median distance to centre [m]", equal=False)
    ax.set_ylim(0, 11)
    ax.set_xlim(0, lit["t"][-1])
    ax.set_title("Tuning flattens the dive but keeps it",
                 fontsize=10, color=TEXT, pad=8)
    ax.legend(frameon=False, fontsize=8.3, loc="lower right")

    # (b) 速率 vs 邻居数
    ax = axB
    groups = [(0, 1, "0"), (1, 2, "1"), (2, 3, "2"), (3, 4, "3"),
              (4, 6, "4–5"), (6, 9, "6–8"), (9, 99, "9+")]

    def profile(trial, mask):
        S = trial.speeds(5)
        cnt = trial.neighbours(2.0)
        return np.array([
            np.median(S[mask & (cnt >= lo) & (cnt < hi)])
            if (mask & (cnt >= lo) & (cnt < hi)).sum() >= 30 else np.nan
            for lo, hi, _ in groups])

    def sim_mask(trial, goal):
        return np.linalg.norm(trial.positions() - goal[None], axis=2) > 1.0

    tt_t = to_trial(tuned["traj"])
    tt_f = to_trial(front["traj"])
    tt_l = to_trial(lit["traj"])
    series = [("measured", BASE, "o", "-", profile(tr, tr.travel_mask())),
              ("model (literature)", ACCENT, "s", "--",
               profile(tt_l, sim_mask(tt_l, lit["goal"]))),
              ("model (tuned)", GREEN, "^", "-.",
               profile(tt_t, sim_mask(tt_t, tuned["goal"]))),
              ("model (front only)", "#9467BD", "v", ":",
               profile(tt_f, sim_mask(tt_f, front["goal"])))]
    xp = np.arange(len(groups))
    for lab, col, mk, ls, v in series:
        ok = ~np.isnan(v)
        ax.plot(xp[ok], v[ok], marker=mk, ls=ls, color=col, lw=1.5, ms=5,
                mec="white", mew=0.6, label=lab)
    ax.set_xticks(xp)
    ax.set_xticklabels([g[2] for g in groups])
    style_axes(ax, xlabel="neighbours within 2 m", ylabel="speed [m/s]",
               equal=False)
    ax.set_ylim(bottom=0)
    ax.set_title("Density–speed relation", fontsize=10, color=TEXT, pad=8)
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=2)

    # (c) 中心区人数占比
    ax = axC
    ax.plot(tc, measured_centrality(tr), color=BASE, lw=1.7, label="measured")
    ax.plot(lit["t"], lit["central"], color=ACCENT, lw=1.7,
            label="literature preset")
    ax.plot(tuned["t"], tuned["central"], color=GREEN, lw=1.7, ls="--",
            label="tuned preset")
    ax.plot(front["t"], front["central"], color="#9467BD", lw=1.6, ls=":",
            label="front only")
    ax.axhline(0.8, color=GREY, ls=":", lw=1.0)
    ax.set_xlim(0, min(lit["t"][-1], 25))
    ax.set_ylim(0, 1.02)
    style_axes(ax, xlabel="time [s]",
               ylabel="fraction within 3 m of the centre", equal=False)
    ax.set_title("How full the centre gets", fontsize=10, color=TEXT, pad=8)
    ax.legend(frameon=False, fontsize=8.3, loc="center right")
    dts = lit["t"][1] - lit["t"][0]
    j_lit = jam_duration(lit["central"], dts)
    i_j = int(np.argmax(lit["central"] > 0.8))
    j_end = int(np.max(np.where(lit["central"] > 0.8)))
    ax.annotate("", xy=(lit["t"][i_j], 0.90), xytext=(lit["t"][j_end], 0.90),
                arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.2))
    ax.text(0.5 * (lit["t"][i_j] + lit["t"][j_end]), 0.925,
            f"{j_lit:.0f} s over 80 %", fontsize=8.4, color=ACCENT,
            ha="center")

    # (d) 各配置的中心拥堵时长
    ax = axD
    labels = ["literature preset", "tuned preset", "literature + front only"]
    vals = [jam_duration(lit["central"], dts),
            jam_duration(tuned["central"], tuned["t"][1] - tuned["t"][0]),
            jam_duration(front["central"], front["t"][1] - front["t"][0])]
    if sweeps:
        labels += ["—— " + l for l in sweeps["labels"]]
        vals += sweeps["jam"]
    y = np.arange(len(labels))
    cols = []
    for l in labels:
        if "front" in l:
            cols.append("#9467BD")
        elif l.startswith("tuned") or ("A=12k" in l and "B=" in l):
            cols.append(GREEN)
        else:
            cols.append(ACCENT)
    ax.barh(y, vals, color=cols, alpha=0.88, height=0.66)
    ref = jam_duration(measured_centrality(tr), tr.dt)
    ax.axvline(ref, color=GREY, ls="--", lw=1.4, label=f"measured {ref:.1f} s")
    for yy, v in zip(y, vals):
        ax.text(v + 0.18, yy, f"{v:.1f}", va="center", fontsize=8.4,
                color=TEXT)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.3)
    ax.set_ylim(len(labels) - 0.45, -0.55)
    style_axes(ax, xlabel="time with the centre over 80 % full  [s]",
               ylabel="", equal=False)
    ax.set_xlim(0, max(list(vals) + [ref]) * 1.32)
    ax.set_title("Tuning nearly removes it — at a cost", fontsize=10,
                 color=TEXT, pad=8)
    ax.legend(frameon=False, fontsize=8.3, loc="lower right")

    fig.suptitle(f"Failure analysis   (tuned preset: {describe_params(params)})",
                 fontsize=11, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save(fig, outdir, "fig8_sfm_failure", dpi, root=ROOT.parent)
    plt.close(fig)


# ------------------------------------------------------------------
# 动图：全程 GIF
# ------------------------------------------------------------------

def fig_gif(tr, sim: dict, params, outdir: Path, t_end: float = 22.0,
            stride: int = 2, dpi: int = 64, n_colors: int = 96,
            stem: str = "anim_sfm_ring") -> Path:
    """把整场仿真录成 GIF：左=实测，右=模型，同一世界坐标系与时间轴。

    为控制体积：分辨率压到约 700×330 像素、每 2 帧取 1 帧播放（等效
    12.5 fps）、全部帧共用一张 96 色调色板。实测记录只有 17 s，之后左侧
    画面保持不动，右侧模型继续走到 ``t_end``。
    """
    from io import BytesIO

    from matplotlib.collections import LineCollection
    from PIL import Image

    fps = tr.fps
    meas, simt, goal = tr.positions(), sim["traj"], tr.goals()
    n = tr.n_agent
    n_frames = min(int(round(t_end * fps)) + 1, simt.shape[0])
    idx = list(range(0, n_frames, stride))
    step_ms = int(round(1000.0 * stride / fps))

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.6))
    panels = []
    for ax, color, title in ((axes[0], BASE, "Measured"),
                             (axes[1], ACCENT, "Social-force model")):
        draw_ring(ax)
        ax.plot(goal[:, 0], goal[:, 1], "o", ms=2.4, mfc="none",
                mec="#BBBBBB", mew=0.5, zorder=3)
        trails = LineCollection([], colors=color, lw=0.35, alpha=0.32,
                                zorder=2)
        ax.add_collection(trails)
        dots = ax.scatter(meas[0][:, 0], meas[0][:, 1], s=13, color=color,
                          edgecolor="white", linewidth=0.25, zorder=4)
        ax.plot(0, 0, "+", ms=9, color="#777777", mew=1.2, zorder=5)
        style_axes(ax)
        ax.set_xlim(-11.6, 11.6)
        ax.set_ylim(-11.6, 11.6)
        ax.set_title(title, fontsize=9.5, color=TEXT, pad=6)
        panels.append((trails, dots))

    stamp = fig.text(0.5, 0.045, "", ha="center", fontsize=9, color=TEXT)
    fig.text(0.5, 0.008, "measured record ends at 17 s", ha="center",
             fontsize=7.6, color=GREY)
    fig.suptitle("Circle antipodal — measured vs. social-force model   "
                 f"({describe_params(params)})", fontsize=9.5, color=TEXT,
                 y=0.985)
    fig.tight_layout(rect=(0.01, 0.075, 0.99, 0.945))

    frames = []
    for j, k in enumerate(idx):
        km = min(k, meas.shape[0] - 1)
        for (trails, dots), arr, kk in ((panels[0], meas, km),
                                        (panels[1], simt, k)):
            trails.set_segments([arr[:kk + 1, i, :] for i in range(n)])
            dots.set_offsets(arr[kk])
        a_m = int(np.sum(np.linalg.norm(meas[km] - goal, axis=1) < 0.6))
        a_s = int(np.sum(np.linalg.norm(simt[k] - goal, axis=1) < 0.6))
        stamp.set_text(f"t = {k * tr.dt:4.1f} s        arrived   "
                       f"measured {a_m:2d}/64   ·   model {a_s:2d}/64")
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
        if (j + 1) % 60 == 0:
            print(f"      已渲染 {j + 1}/{len(idx)} 帧")
    plt.close(fig)

    # 所有帧共用同一张调色板，帧间色表一致，压缩率更高
    pal = frames[len(frames) // 2].convert("P", palette=Image.ADAPTIVE,
                                          colors=n_colors)
    q = [f.quantize(palette=pal) for f in frames]
    out = Path(outdir) / f"{stem}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    q[0].save(out, save_all=True, append_images=q[1:], duration=step_ms,
              loop=0, optimize=True, disposal=2)
    size_mb = out.stat().st_size / 1e6
    print(f"[写出] {out.relative_to(ROOT.parent)}  "
          f"（{len(q)} 帧，{size_mb:.1f} MB，{1000 / step_ms:.0f} fps 播放）")
    return out


# ------------------------------------------------------------------
# 参数扫描（失效分析第四格用）
# ------------------------------------------------------------------

def run_sweep(tr, duration: float) -> dict:
    labels, jam = [], []
    for A in (2000, 4000, 8000, 12000, 16000):
        r = run_config(tr, SFMParams(A=A), duration)
        labels.append(f"v⁰=2.0  A={A / 1000:g}k")
        jam.append(jam_duration(r["central"], r["t"][1] - r["t"][0]))
        print(f"    A={A:<6g} 中心拥堵 {jam[-1]:5.1f} s   "
              f"t50 {time_at(r['arrivals'], r['t'], 32):5.1f} s")
    for B in (0.08, 0.10, 0.12):
        r = run_config(tr, SFMParams(v0=2.8, A=12000.0, B=B), duration)
        labels.append(f"v⁰=2.8 A=12k B={B:g}")
        jam.append(jam_duration(r["central"], r["t"][1] - r["t"][0]))
        print(f"    B={B:<5g} 中心拥堵 {jam[-1]:5.1f} s   "
              f"t50 {time_at(r['arrivals'], r['t'], 32):5.1f} s")
    r = run_config(tr, SFMParams(front_only=True), duration)
    print(f"    front-only   中心拥堵 "
          f"{jam_duration(r['central'], r['t'][1] - r['t'][0]):5.1f} s   "
          f"t50 {time_at(r['arrivals'], r['t'], 32):5.1f} s")
    return {"labels": labels, "jam": jam}


# ------------------------------------------------------------------
# 文字报告
# ------------------------------------------------------------------

def report(tr, cases: dict) -> None:
    """cases: 名称 -> run_config 的结果。第一列永远是实测。"""
    from social_force import contact_stats, measured_contacts
    r = 0.25
    width = 12
    print(f"  {'':<18}{'实测':>10}"
          + "".join(f"{k:>{width}}" for k in cases))

    def line(label, meas, fmt, fn):
        s = f"  {label:<18}{fmt.format(meas):>10}"
        for v in cases.values():
            s += f"{fmt.format(fn(v)):>{width}}"
        print(s)

    line("中位到达时间[s]", float(np.median(tr.arrival_time())), "{:.1f}",
         lambda v: time_at(v["arrivals"], v["t"], 32))
    line("全部到达时间[s]", float(tr.arrival_time().max()), "{:.1f}",
         lambda v: time_at(v["arrivals"], v["t"], 64))
    line("中心拥堵时长[s]", jam_duration(measured_centrality(tr), tr.dt),
         "{:.1f}", lambda v: jam_duration(v["central"], v["t"][1] - v["t"][0]))
    line("中心区占比峰值", float(measured_centrality(tr).max()), "{:.2f}",
         lambda v: float(v["central"].max()))
    line("最近两人间距[m]", float(tr.nearest_distance().min()), "{:.3f}",
         lambda v: float(v["minpair"].min()))
    line("横向散布 IQR[m]", lateral_spread(tr), "{:.2f}",
         lambda v: lateral_spread(to_trial(v["traj"])))

    ps = pass_sides(tr)
    s = f"  {'过中心 右/左':<18}{ps[0]:>6d}/{ps[1]:<3d}"
    for v in cases.values():
        p = pass_sides(to_trial(v["traj"]))
        s += f"{p[0]:>{width - 4}d}/{p[1]:<3d}"
    print(s)

    print()
    print("【接触力是否真的起作用】（两人中心距 < 2r = 0.5 m 记为接触）")
    line("有接触的时刻占比", measured_contacts(tr, r)["touching_frames"],
         "{:.1%}", lambda v: contact_stats(v["traj"], r)["touching_frames"])
    line("最大同时接触对数", float(measured_contacts(tr, r)["max_pairs"]),
         "{:.0f}", lambda v: float(contact_stats(v["traj"], r)["max_pairs"]))
    line("最大挤压深度[m]", measured_contacts(tr, r)["max_overlap"], "{:.3f}",
         lambda v: contact_stats(v["traj"], r)["max_overlap"])


# ------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="week02 社会力模型复现圆环对趾")
    ap.add_argument("--data", default=str(default_data_path()))
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    ap.add_argument("--duration", type=float, default=25.0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="tuned",
                    help="对比图里展示 / 动图里播放的那一组参数")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE", help="覆盖单个参数，可重复")
    ap.add_argument("--sweep", action="store_true",
                    help="失效分析里加参数扫描")
    ap.add_argument("--gif", action="store_true", help="另存全程动图 GIF")
    ap.add_argument("--only", choices=["compare", "failure", "gif"],
                    default=None)
    ap.add_argument("--gif-end", type=float, default=22.0,
                    help="动图覆盖到多少秒")
    ap.add_argument("--gif-stride", type=int, default=2,
                    help="动图每 N 帧取 1 帧（越大文件越小）")
    ap.add_argument("--gif-dpi", type=int, default=56,
                    help="动图分辨率（dpi）")
    ap.add_argument("--gif-colors", type=int, default=48,
                    help="动图调色板颜色数")
    args = ap.parse_args()

    tr = load(Path(args.data))
    outdir = Path(args.outdir)

    prim = build_params(args.preset, args.overrides)
    print(f"主用参数（{args.preset}）：{describe_params(prim)}")
    sim_prim = run_config(tr, prim, args.duration)

    if args.only == "gif":
        fig_gif(tr, sim_prim, prim, outdir, t_end=args.gif_end,
                stride=args.gif_stride, dpi=args.gif_dpi,
                n_colors=args.gif_colors)
        return 0
    if args.only == "compare":
        fig_compare(tr, sim_prim, prim, outdir, args.dpi)
        return 0

    print("运行 文献值预设…")
    sim_lit = run_config(tr, PRESETS["literature"](), args.duration)
    sim_tuned = (sim_prim if args.preset == "tuned"
                 else run_config(tr, PRESETS["tuned"](), args.duration))
    print("运行 文献值 + 仅前方排斥（失效分析探针）…")
    sim_front = run_config(tr, SFMParams(front_only=True), args.duration)

    print()
    report(tr, {"文献值": sim_lit, "调参后": sim_tuned, "仅前方": sim_front})

    sweeps = None
    if args.sweep or args.only == "failure":
        print("\n参数扫描…")
        sweeps = run_sweep(tr, args.duration)

    print()
    fig_compare(tr, sim_prim, prim, outdir, args.dpi)
    fig_failure(tr, sim_lit, sim_tuned, sim_front, sweeps, prim, outdir,
                args.dpi)

    if args.gif:
        print()
        fig_gif(tr, sim_prim, prim, outdir, t_end=args.gif_end,
                stride=args.gif_stride, dpi=args.gif_dpi,
                n_colors=args.gif_colors)
    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
