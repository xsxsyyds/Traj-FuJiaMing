#!/usr/bin/env python3
"""week02 · 步骤③ 实现与验证 —— 用社会力模型复现圆环对趾。

做法
----
仿真从**实测的初始位置**出发（而不是自己造一个均匀圆环），目标取实测
起点的对趾点，然后看模型能否走完实测那条路。这样比较的是"同样的初始
条件、同样的目标下，模型的行为与真人差在哪"，而不是"两个不同实验的
统计量差在哪"。

三种配置
--------
- ``isotropic`` 课程要求的三项力，排斥力各向同性（默认）
- ``front``     只保留来自前半平面的排斥力（失效分析探针）
- 以及 ``--sweep`` 下的参数扫描

用法
----
    python week02/model/simulate.py
    python week02/model/simulate.py --front-only
    python week02/model/simulate.py --sweep
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

from circle_data import (ACCENT, BASE, CYCLIC, GRID, TEXT, apply_style,
                         default_data_path, draw_ring, load, save, style_axes)
from social_force import (SFMParams, accelerations, arrival_curve, centrality,
                          min_pair_distance, simulate)

apply_style()

GREEN = "#2A9D8F"
GREY = "#8C8C8C"


# ------------------------------------------------------------------
# 运行
# ------------------------------------------------------------------

def run_config(tr, params: SFMParams, duration: float,
               sample: int = 20) -> dict:
    """从实测初始条件仿真，返回轨迹与派生量。"""
    pos0 = tr.positions()[0]
    goal = tr.goals()
    traj = simulate(pos0, goal, params, duration=duration, sample_every=sample)
    t = np.arange(traj.shape[0]) * (params.dt * sample)
    return {
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


def lateral_spread(trial, lo: float = 0.35, hi: float = 0.55) -> float:
    """行进中段横向偏移的 IQR [m]，衡量"人流散开多宽"。"""
    lat = trial.lateral()
    a, b = int(lo * trial.n_frame), int(hi * trial.n_frame)
    v = lat[a:b].ravel()
    return float(np.percentile(v, 75) - np.percentile(v, 25))


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
# 图 7：模型 vs 实测
# ------------------------------------------------------------------

def fig_compare(tr, sim: dict, outdir: Path, dpi: int) -> None:
    th0 = np.arctan2(tr.positions()[0][:, 1], tr.positions()[0][:, 0])
    def align(arr):
        a = np.empty_like(arr)
        for i in range(arr.shape[1]):
            c, s = np.cos(th0[i]), np.sin(th0[i])
            a[:, i, 0] = arr[:, i, 0] * c + arr[:, i, 1] * s
            a[:, i, 1] = -arr[:, i, 0] * s + arr[:, i, 1] * c
        return a

    t = sim["t"]
    traj = sim["traj"]
    dur = tr.duration

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 10.0))
    (axA, axB), (axC, axD) = axes

    # (a)(b) 对齐后的轨迹
    for ax, arr, title, col in (
            (axA, align(tr.positions()), "Measured", None),
            (axB, align(traj), "Social-force model", None)):
        draw_ring(ax)
        for i in range(arr.shape[1]):
            if col is None:
                ax.plot(arr[:, i, 0], arr[:, i, 1], lw=0.7, alpha=0.45,
                        color=BASE)
            else:
                ax.plot(arr[:, i, 0], arr[:, i, 1], lw=0.7, alpha=0.45,
                        color=col)
        ax.axhline(0, color="#B8B8B8", lw=0.8, ls=":")
        ax.plot(10, 0, "o", ms=6, color="#444444", mec="white", mew=0.7,
                zorder=5)
        ax.plot(-10, 0, "s", ms=6, color="#444444", mec="white", mew=0.7,
                zorder=5)
        ax.plot(0, 0, "+", ms=9, color=ACCENT, mew=1.3, zorder=5)
        style_axes(ax)
        ax.set_title(title, fontsize=10, color=TEXT, pad=8)

    # (c) 半径剖面：全场中位
    ax = axC
    Rm = np.median(np.linalg.norm(tr.positions(), axis=2), axis=1)
    tc = np.arange(tr.n_frame) * tr.dt
    ax.plot(tc, Rm, color=BASE, lw=1.7, label="measured")
    Rs = np.median(sim["radius"], axis=1)
    ax.plot(t, Rs, color=ACCENT, lw=1.7, label="model")
    ax.axhspan(0, 3.0, color=ACCENT, alpha=0.06, lw=0)
    ax.text(21.5, 1.4, "centre zone", fontsize=8, color="#9A8A8A", ha="right")
    style_axes(ax, xlabel="time [s]",
               ylabel="median distance to centre [m]", equal=False)
    ax.set_ylim(0, 11)
    ax.set_xlim(0, max(t[-1], tc[-1]))
    ax.set_title("The model overshoots into the centre",
                 fontsize=10, color=TEXT, pad=8)
    ax.legend(frameon=False, fontsize=8.5, loc="center right")

    # (d) 累计到达
    ax = axD
    arr_m = np.sort(tr.arrival_time())
    ax.plot(arr_m, np.arange(1, tr.n_agent + 1), color=BASE, lw=1.7,
            label="measured")
    ax.plot(t, sim["arrivals"], color=ACCENT, lw=1.7, label="model")
    ax.set_xlim(0, max(dur, t[-1]))
    ax.set_ylim(0, 68)
    style_axes(ax, xlabel="time [s]", ylabel="cumulative arrivals [of 64]",
               equal=False)
    ax.set_title("Everyone still gets there — but later",
                 fontsize=10, color=TEXT, pad=8)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    td = time_at(sim["arrivals"], t, 32)
    ax.axvline(float(np.median(tr.arrival_time())), color=GREY, ls="--", lw=1.0)
    ax.axvline(td, color=ACCENT, ls="--", lw=1.0)
    ax.text(float(np.median(tr.arrival_time())) - 0.5, 44,
            "measured\nhalf-time", fontsize=8, color=GREY, ha="right",
            linespacing=1.35)
    ax.text(td + 0.5, 26, "model\nhalf-time", fontsize=8, color=ACCENT,
            ha="left", linespacing=1.35)

    fig.suptitle("Social-force model vs. the measured ring experiment",
                 fontsize=12, color=TEXT, y=0.995)
    save(fig, outdir, "fig7_sfm_vs_measured", dpi, root=ROOT.parent)
    plt.close(fig)


# ------------------------------------------------------------------
# 图 8：失效分析
# ------------------------------------------------------------------

def fig_failure(tr, iso: dict, front: dict, sweeps: dict | None,
                outdir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8),
                             gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.25]))

    # (a) 速率 vs 邻居数：实测 vs 模型
    ax = axes[0]
    groups = [(0, 1, "0"), (1, 2, "1"), (2, 3, "2"), (3, 4, "3"),
              (4, 6, "4–5"), (6, 9, "6–8"), (9, 99, "9+")]

    def profile(trial, mask):
        S = trial.speeds(5)
        cnt = trial.neighbours(2.0)
        out = []
        for lo, hi, _ in groups:
            sel = mask & (cnt >= lo) & (cnt < hi)
            out.append(np.median(S[sel]) if sel.sum() >= 30 else np.nan)
        return np.array(out), cnt, S

    from social_force import to_trial
    t_iso = to_trial(iso["traj"])
    t_front = to_trial(front["traj"])

    def sim_mask(trial, goal):
        """行进段：还没到达目标就还在走，排除到达后站立的样本。"""
        d = np.linalg.norm(trial.positions() - goal[None], axis=2)
        return d > 1.0

    xp = np.arange(len(groups))
    series = [
        ("measured", BASE, "o", "-", profile(tr, tr.travel_mask())[0]),
        ("model (isotropic)", ACCENT, "s", "--",
         profile(t_iso, sim_mask(t_iso, iso["goal"]))[0]),
        ("model (front only)", GREEN, "^", ":",
         profile(t_front, sim_mask(t_front, front["goal"]))[0]),
    ]
    for lab, col, mk, ls, v in series:
        ok = ~np.isnan(v)
        ax.plot(xp[ok], v[ok], marker=mk, ls=ls, color=col, lw=1.6, ms=5,
                mec="white", mew=0.6, label=lab)
    ax.set_xticks(xp)
    ax.set_xticklabels([g[2] for g in groups])
    style_axes(ax, xlabel="neighbours within 2 m", ylabel="speed [m/s]",
               equal=False)
    ax.set_ylim(bottom=0)
    ax.set_title("Density–speed relation", fontsize=10, color=TEXT, pad=8)
    ax.legend(frameon=False, fontsize=8.3, loc="lower left")

    # (b) 中心区人数占比随时间
    ax = axes[1]
    tc = np.arange(tr.n_frame) * tr.dt
    ax.plot(tc, measured_centrality(tr), color=BASE, lw=1.7,
            label="measured")
    ax.plot(iso["t"], iso["central"], color=ACCENT, lw=1.7,
            label="model (isotropic)")
    ax.plot(front["t"], front["central"], color=GREEN, lw=1.7,
            label="model (front only)")
    ax.axhline(0.8, color=GREY, ls=":", lw=1.0)
    ax.set_xlim(0, min(iso["t"][-1], 25))
    ax.set_ylim(0, 1.02)
    style_axes(ax, xlabel="time [s]",
               ylabel="fraction within 3 m of the centre", equal=False)
    ax.set_title("The model parks the crowd at the centre",
                 fontsize=10, color=TEXT, pad=8)
    ax.legend(frameon=False, fontsize=8.3, loc="lower center")
    j_iso = jam_duration(iso["central"], iso["t"][1] - iso["t"][0])
    i_j = int(np.argmax(iso["central"] > 0.8))
    j_end = int(np.max(np.where(iso["central"] > 0.8)))
    ax.annotate("", xy=(iso["t"][i_j], 0.88), xytext=(iso["t"][j_end], 0.88),
                arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.2))
    ax.text(0.5 * (iso["t"][i_j] + iso["t"][j_end]), 0.905,
            f"{j_iso:.0f} s of gridlock", fontsize=8.4, color=ACCENT,
            ha="center")

    # (c) 各配置的中心拥堵时长
    ax = axes[2]
    if sweeps:
        labels, vals = sweeps["labels"], sweeps["jam"]
    else:
        labels, vals = ["isotropic\n(course three forces)", "front only"], [
            jam_duration(iso["central"], iso["t"][1] - iso["t"][0]),
            jam_duration(front["central"], front["t"][1] - front["t"][0])]
    y = np.arange(len(labels))
    cols = [GREEN if "front" in l else (ACCENT if "isotropic" in l else BASE)
            for l in labels]
    ax.barh(y, vals, color=cols, alpha=0.85, height=0.72)
    ref = jam_duration(measured_centrality(tr), tr.dt)
    ax.axvline(ref, color=GREY, ls="--", lw=1.4,
               label=f"measured {ref:.1f} s")
    for yy, v in zip(y, vals):
        ax.text(v + 0.18, yy, f"{v:.1f}", va="center", fontsize=8.4, color=TEXT)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.3, linespacing=1.3)
    ax.set_ylim(len(labels) - 0.45, -0.55)
    style_axes(ax, xlabel="time with the centre over 80 % full  [s]",
               ylabel="", equal=False)
    ax.set_xlim(0, max(list(vals) + [ref]) * 1.30)
    ax.set_title("Gridlock is a direction problem, not a tuning one",
                 fontsize=10, color=TEXT, pad=8)
    ax.legend(frameon=False, fontsize=8.3, loc="lower right")

    save(fig, outdir, "fig8_sfm_failure", dpi, root=ROOT.parent)
    plt.close(fig)


# ------------------------------------------------------------------
# 参数扫描
# ------------------------------------------------------------------

def run_sweep(tr, duration: float) -> dict:
    labels, jam = [], []
    for A in (500, 2000, 8000):
        r = run_config(tr, SFMParams(A=A), duration)
        labels.append(f"isotropic   A={A:g} N")
        jam.append(jam_duration(r["central"], r["t"][1] - r["t"][0]))
        print(f"    A={A:<6g} 中心拥堵 {jam[-1]:5.1f} s   "
              f"t50 {time_at(r['arrivals'], r['t'], 32):5.1f} s")
    for v0 in (1.0, 3.0, 4.0):
        r = run_config(tr, SFMParams(v0=v0), duration)
        labels.append(f"isotropic   v⁰={v0:g} m/s")
        jam.append(jam_duration(r["central"], r["t"][1] - r["t"][0]))
        print(f"    v0={v0:<5g} 中心拥堵 {jam[-1]:5.1f} s   "
              f"t50 {time_at(r['arrivals'], r['t'], 32):5.1f} s")
    r = run_config(tr, SFMParams(front_only=True), duration)
    labels.append("front only")
    jam.append(jam_duration(r["central"], r["t"][1] - r["t"][0]))
    print(f"    front-only   中心拥堵 {jam[-1]:5.1f} s   "
          f"t50 {time_at(r['arrivals'], r['t'], 32):5.1f} s")
    return {"labels": labels, "jam": jam}


# ------------------------------------------------------------------

def report(tr, iso: dict, front: dict) -> None:
    p = SFMParams()
    dts = iso["t"][1] - iso["t"][0]
    print(f"模型参数  A={p.A:g} N  B={p.B:g} m  k={p.k:g} kg/s²  "
          f"κ={p.kappa:g} kg/(m·s)  τ={p.tau:g} s  v⁰={p.v0:g} m/s  "
          f"m={p.mass:g} kg  r={p.radius:g} m  内部步长 {p.dt*1000:g} ms")
    print()
    meas_t50 = float(np.median(tr.arrival_time()))
    print(f"                     实测      模型(各向同性)   模型(仅前方)")
    print(f"  中位到达时间[s] {meas_t50:8.1f}  "
          f"{time_at(iso['arrivals'], iso['t'], 32):14.1f}  "
          f"{time_at(front['arrivals'], front['t'], 32):14.1f}")
    print(f"  全部到达时间[s] {tr.arrival_time().max():8.1f}  "
          f"{time_at(iso['arrivals'], iso['t'], 64):14.1f}  "
          f"{time_at(front['arrivals'], front['t'], 64):14.1f}")
    print(f"  中心拥堵时长[s] {jam_duration(measured_centrality(tr), tr.dt):8.1f}  "
          f"{jam_duration(iso['central'], dts):14.1f}  "
          f"{jam_duration(front['central'], dts):14.1f}")
    print(f"  最近两人间距[m] {tr.nearest_distance().min():8.3f}  "
          f"{iso['minpair'].min():14.3f}  {front['minpair'].min():14.3f}")
    from social_force import to_trial
    t_iso, t_front = to_trial(iso["traj"]), to_trial(front["traj"])
    print(f"  横向散布IQR[m]  {lateral_spread(tr):8.2f}  "
          f"{lateral_spread(t_iso):14.2f}  {lateral_spread(t_front):14.2f}")
    rm_ = pass_sides(tr)
    ri = pass_sides(t_iso)
    rf = pass_sides(t_front)
    print(f"  过中心 右/左    {rm_[0]:4d}/{rm_[1]:<3d}  "
          f"{ri[0]:10d}/{ri[1]:<3d}  {rf[0]:10d}/{rf[1]:<3d}")
    print()
    rm = np.median(np.linalg.norm(tr.positions(), axis=2), axis=1).min()
    rs = iso["radius"].min(axis=1)
    print(f"  半径剖面最小值[m]     实测 {rm:.2f}   "
          f"模型 {np.median(rs):.2f}")
    print()
    print("【接触力是否真的起作用】（两人中心距 < 2r = 0.5 m 记为接触）")
    from social_force import contact_stats, measured_contacts
    cs_m = measured_contacts(tr, p.radius)
    cs_i = contact_stats(iso["traj"], p.radius)
    cs_f = contact_stats(front["traj"], p.radius)
    print("                     实测      模型(各向同性)   模型(仅前方)")
    print(f"  有接触的时刻占比 {cs_m['touching_frames']:8.1%}  "
          f"{cs_i['touching_frames']:14.1%}  {cs_f['touching_frames']:14.1%}")
    print(f"  同时接触对子 最大 {cs_m['max_pairs']:8d}  "
          f"{cs_i['max_pairs']:14d}  {cs_f['max_pairs']:14d}")
    print(f"  最大挤压深度[m]  {cs_m['max_overlap']:8.3f}  "
          f"{cs_i['max_overlap']:14.3f}  {cs_f['max_overlap']:14.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="week02 社会力模型复现圆环对趾")
    ap.add_argument("--data", default=str(default_data_path()))
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    ap.add_argument("--duration", type=float, default=25.0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--front-only", action="store_true",
                    help="只仿真「仅前方排斥」这一种配置")
    ap.add_argument("--sweep", action="store_true", help="额外做参数扫描")
    ap.add_argument("--only", choices=["compare", "failure"], default=None)
    args = ap.parse_args()

    tr = load(Path(args.data))
    outdir = Path(args.outdir)

    if args.only == "compare":
        iso = run_config(tr, SFMParams(), args.duration)
        fig_compare(tr, iso, outdir, args.dpi)
        return 0

    print("运行 各向同性（课程要求的三项力）…")
    iso = run_config(tr, SFMParams(), args.duration)
    print("运行 仅前方排斥（失效分析探针）…")
    front = run_config(tr, SFMParams(front_only=True), args.duration)

    sweeps = None
    if args.sweep:
        print("参数扫描…")
        sweeps = run_sweep(tr, args.duration)

    report(tr, iso, front)
    print()
    fig_compare(tr, iso, outdir, args.dpi)
    fig_failure(tr, iso, front, sweeps, outdir, args.dpi)
    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
