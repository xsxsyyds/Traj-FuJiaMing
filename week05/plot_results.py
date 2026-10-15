# -*- coding: utf-8 -*-
"""
week05 对比图。全部输出 dpi=300 的 PNG（约定：图题中文，坐标/图例英文）。

    fig01_speed_response.png     场景 A 前车匀速：速度响应时间历程
    fig02_startup_metrics.png    场景 A 指标：起步加速度与上升时间
    fig03_braking_response.png   场景 B 前车急刹：净间距与速度时间历程
    fig04_braking_metrics.png    场景 B 指标：最小净间距与碰撞相对速度
    fig05_ngsim_errors.png       场景 C 误差分布：位置 / 速度 RMSE
    fig06_ngsim_examples.png     场景 C 三条真实车对的轨迹重放
    fig07_ring_stability.png     圆周稳定性：实测增长比 vs 线性判据
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))

from car_following_models import build_models, simulate_follower  # noqa: E402

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

MODELS = ["ovm", "fvdm", "idm"]
COLOR = {"ovm": "#1F4E79", "fvdm": "#C00000", "idm": "#2A9D8F",
         "baseline": "#9E9E9E", "lead": "#6B6B6B"}
LABEL = {"ovm": "OVM", "fvdm": "FVDM", "idm": "IDM",
         "baseline": "Constant velocity"}


def load_json(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def save(fig, name):
    p = RESULTS / name
    fig.savefig(p)
    plt.close(fig)
    print("写出", p.name)


# ==========================================================================
# fig01  场景 A 速度响应
# ==========================================================================
def fig01():
    z = np.load(RESULTS / "exp_a_series.npz")
    speeds = [6.0, 10.0, 14.0]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.1))
    for ax, vl in zip(axes, speeds):
        t = z[f"ovm|{vl:g}__t"]
        # 每个面板按本面板的峰值定纵轴，避免曲线被裁掉
        peak = max(float(z[f"{mk}|{vl:g}__v"].max()) for mk in MODELS)
        ax.plot(t, z[f"ovm|{vl:g}__lead_v"], color=COLOR["lead"], lw=1.8,
                ls="--", label="Leader", zorder=3)
        for mk in MODELS:
            ax.plot(t, z[f"{mk}|{vl:g}__v"], color=COLOR[mk], lw=1.9,
                    label=LABEL[mk], zorder=4)
        ax.axhline(vl, color="#B0B0B0", lw=0.9, ls=":", zorder=2)
        ax.set_xlim(0, 45)
        ax.set_ylim(0, peak * 1.12)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Speed [m/s]")
        ax.set_title(f"前车匀速 v_lead = {vl:g} m/s", fontsize=11)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    fig.suptitle("场景 A  前车匀速：跟驰者从静止起步的速度响应",
                 fontsize=13.5, y=1.045)
    save(fig, "fig01_speed_response.png")


# ==========================================================================
# fig02  场景 A 指标
# ==========================================================================
def fig02():
    rows = load_json("exp_a.json")
    speeds = sorted({r["v_lead"] for r in rows})
    x = np.arange(len(speeds))
    w = 0.26
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3))
    ax = axes[0]
    for i, mk in enumerate(MODELS):
        vals = [next(r["a_start"] for r in rows
                     if r["model"] == mk and r["v_lead"] == v) for v in speeds]
        b = ax.bar(x + (i - 1) * w, vals, w, color=COLOR[mk],
                   edgecolor="white", linewidth=0.8, label=LABEL[mk])
        ax.bar_label(b, fmt="%.2f", fontsize=7.5, padding=1.5)
    ax.axhline(3.5, color="#B8860B", lw=1.2, ls=":")
    ax.annotate("emergency braking limit 3.5 m/s2", xy=(0.98, 3.5),
                xytext=(0.98, 26), textcoords="data", fontsize=7,
                color="#8B6508", ha="right", va="center",
                arrowprops=dict(arrowstyle="-", color="#B8860B", lw=0.8))
    ax.set_yscale("log")
    ax.set_ylim(0.4, 60)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in speeds])
    ax.set_xlabel("Leader speed [m/s]")
    ax.set_ylabel("Initial acceleration [m/s2]  (log)")
    ax.set_title("起步加速度", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left")

    ax = axes[1]
    for i, mk in enumerate(MODELS):
        vals = [next(r["t_rise_s"] for r in rows
                     if r["model"] == mk and r["v_lead"] == v) for v in speeds]
        b = ax.bar(x + (i - 1) * w, vals, w, color=COLOR[mk],
                   edgecolor="white", linewidth=0.8, label=LABEL[mk])
        ax.bar_label(b, fmt="%.1f", fontsize=7.5, padding=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in speeds])
    ax.set_xlabel("Leader speed [m/s]")
    ax.set_ylabel("Rise time 10%-90% [s]")
    ax.set_title("速度上升时间", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left")

    ax = axes[2]
    for i, mk in enumerate(MODELS):
        vals = [next(r["overshoot_pct"] for r in rows
                     if r["model"] == mk and r["v_lead"] == v) for v in speeds]
        b = ax.bar(x + (i - 1) * w, vals, w, color=COLOR[mk],
                   edgecolor="white", linewidth=0.8, label=LABEL[mk])
        ax.bar_label(b, fmt="%.1f%%", fontsize=7.5, padding=1.5)
    ax.axhline(0.0, color="#404040", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in speeds])
    ax.set_xlabel("Leader speed [m/s]")
    ax.set_ylabel("Speed overshoot [%]")
    ax.set_title("速度超调量", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")

    fig.suptitle("场景 A  前车匀速：速度响应指标对比", fontsize=13.5, y=1.03)
    save(fig, "fig02_startup_metrics.png")


# ==========================================================================
# fig03  场景 B 时间历程
# ==========================================================================
def _get(series, key):
    parts = series[key]
    return parts


def fig03(b=6.0):
    z = np.load(RESULTS / "exp_b_series.npz")
    rows = load_json("exp_b.json")
    # 失效后状态被冻结，画出来会被误读成「还在开」，故在失效时刻截断
    fail_at = {}
    for mk in MODELS:
        r = next(r for r in rows if r["model"] == mk and r["brake"] == b)
        fail_at[mk] = r["t_fail"]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2))
    ax = axes[0]
    for mk in MODELS:
        t = z[f"{mk}|{b:g}__t"]
        g = z[f"{mk}|{b:g}__gap"]
        tf = fail_at[mk]
        n = len(t) if tf is None else int(np.searchsorted(t, tf)) + 1
        ax.plot(t[:n], g[:n], color=COLOR[mk], lw=2.0, label=LABEL[mk])
        if tf is not None:
            ax.plot(t[n - 1], g[n - 1], "X", color=COLOR[mk], ms=9,
                    markeredgecolor="white", markeredgewidth=1.0, zorder=6)
    ax.axhline(0.0, color="#404040", lw=1.4)
    ax.text(0.3, -2.0, "collision threshold  gap = 0", fontsize=7.5,
            color="#404040")
    ax.text(0.97, 0.06, "X = collision", transform=ax.transAxes,
            fontsize=7.5, color="#404040", ha="right")
    ax.set_xlim(0, 12)
    ax.set_ylim(-3, 21)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Net spacing [m]")
    ax.set_title(f"前车以 {b:g} m/s2 刹停", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")

    ax = axes[1]
    ax.plot(z[f"ovm|{b:g}__t"], z[f"ovm|{b:g}__lead_v"], color=COLOR["lead"],
            lw=1.8, ls="--", label="Leader")
    for mk in MODELS:
        t = z[f"{mk}|{b:g}__t"]
        v = z[f"{mk}|{b:g}__v"]
        tf = fail_at[mk]
        n = len(t) if tf is None else int(np.searchsorted(t, tf)) + 1
        ax.plot(t[:n], v[:n], color=COLOR[mk], lw=2.0, label=LABEL[mk])
        if tf is not None:
            ax.plot(t[n - 1], v[n - 1], "X", color=COLOR[mk], ms=9,
                    markeredgecolor="white", markeredgewidth=1.0, zorder=6)
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.6, 14)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [m/s]")
    ax.set_title("速度历程（曲线止于碰撞时刻）", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")

    fig.suptitle("场景 B  前车急刹：跟驰者净间距与速度的时间历程",
                 fontsize=13.5, y=1.03)
    save(fig, "fig03_braking_response.png")


# ==========================================================================
# fig04  场景 B 指标
# ==========================================================================
def fig04():
    rows = load_json("exp_b.json")
    bs = sorted({r["brake"] for r in rows})
    x = np.arange(len(bs))
    w = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2))

    ax = axes[0]
    COLL = -0.30          # 碰撞柱的示意高度（真实值≈0，此处仅为可见性）
    for i, mk in enumerate(MODELS):
        xs = x + (i - 1) * w
        for j, b in enumerate(bs):
            r = next(r for r in rows if r["model"] == mk and r["brake"] == b)
            v = r["gap_min"]
            if v > 0:
                ax.bar(xs[j], v, w, color=COLOR[mk], edgecolor="white",
                       linewidth=0.8,
                       label=(LABEL[mk] if j == 0 else None))
                ax.text(xs[j], v + 0.05, f"{v:.2f}", ha="center",
                        fontsize=7.5, color="#2B2B2B")
            else:
                # 接触：真实最小值≈0，柱子会消失，故用红底斜纹块示意
                ax.bar(xs[j], COLL, w, color="#F2C6C6",
                       edgecolor=COLOR[mk], linewidth=1.0, hatch="///",
                       label=(LABEL[mk] if j == 0 else None))
    ax.text(0.03, 0.10, "hatched = follower hit the leader (gap <= 0)",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=7.5, color="#A00000")
    ax.axhline(0.0, color="#404040", lw=1.5)
    ax.set_ylim(COLL - 0.42, 1.45)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:g}" for b in bs])
    ax.set_xlabel("Leader deceleration [m/s2]")
    ax.set_ylabel("Minimum net spacing [m]")
    ax.set_title("最小净间距（斜纹块 = 跟驰者接触到前车）", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")

    ax = axes[1]
    for i, mk in enumerate(MODELS):
        vals, xs, ok = [], [], []
        for j, b in enumerate(bs):
            r = next(r for r in rows if r["model"] == mk and r["brake"] == b)
            v = r["v_at_fail"]
            if v is None:
                continue
            xs.append(x[j] + (i - 1) * w)
            vals.append(max(float(v), 0.02))
            ok.append(r["failed"])
        ax.bar(xs, vals, w, color=COLOR[mk], edgecolor="white",
               linewidth=0.8, label=LABEL[mk])
        for xx, vv in zip(xs, vals):
            ax.text(xx, vv * 1.15, f"{vv:.2f}", ha="center", fontsize=7.5)
    ax.set_yscale("log")
    ax.set_ylim(0.015, 12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:g}" for b in bs])
    ax.set_xlabel("Leader deceleration [m/s2]")
    ax.set_ylabel("Impact speed [m/s]  (log)")
    ax.set_title("接触瞬间相对速度（无柱子 = 未发生接触）", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left")

    fig.suptitle("场景 B  前车减速：安全裕度与接触严重程度", fontsize=13.5, y=1.03)
    save(fig, "fig04_braking_metrics.png")


# ==========================================================================
# fig05  场景 C 误差分布
# ==========================================================================
def fig05():
    d = load_json("exp_c.json")
    rows = d["pairs"]
    order = ["baseline", "ovm", "fvdm", "idm"]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))
    ax = axes[0]
    data = [[r["rmse_pos"] for r in rows if r["model"] == k] for k in order]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                    medianprops=dict(color="white", lw=1.6),
                    flierprops=dict(marker="o", markersize=2.4,
                                    markerfacecolor="#9E9E9E",
                                    markeredgecolor="none", alpha=0.5))
    for patch, k in zip(bp["boxes"], order):
        patch.set_facecolor(COLOR[k])
        patch.set_edgecolor("#3A3A3A")
        patch.set_linewidth(0.9)
        patch.set_alpha(0.92 if k != "baseline" else 0.7)
    ax.set_xticklabels([LABEL[k] for k in order], fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("Position RMSE [m]  (log)")
    ax.set_title("轨迹位置误差（= 净间距误差）", fontsize=11)
    for i, k in enumerate(order):
        v = np.array(data[i])
        ax.text(i + 1, np.median(v) * 1.35, f"{np.median(v):.2f}",
                ha="center", fontsize=8, color="#2B2B2B")

    ax = axes[1]
    data2 = [[r["rmse_v"] for r in rows if r["model"] == k]
             for k in ("ovm", "fvdm", "idm")]
    bp = ax.boxplot(data2, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", lw=1.6),
                    flierprops=dict(marker="o", markersize=2.4,
                                    markerfacecolor="#9E9E9E",
                                    markeredgecolor="none", alpha=0.5))
    for patch, k in zip(bp["boxes"], ("ovm", "fvdm", "idm")):
        patch.set_facecolor(COLOR[k])
        patch.set_edgecolor("#3A3A3A")
        patch.set_linewidth(0.9)
    ax.set_xticklabels([LABEL[k] for k in ("ovm", "fvdm", "idm")], fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("Speed RMSE [m/s]  (log)")
    ax.set_title("速度误差", fontsize=11)
    for i, k in enumerate(("ovm", "fvdm", "idm")):
        v = np.array(data2[i])
        ax.text(i + 1, np.median(v) * 1.35, f"{np.median(v):.2f}",
                ha="center", fontsize=8, color="#2B2B2B")

    fig.suptitle("场景 C  真实跟驰（NGSIM I-80，205 对车）：复现误差分布",
                 fontsize=13.5, y=1.03)
    save(fig, "fig05_ngsim_errors.png")


# ==========================================================================
# fig06  场景 C 示例轨迹
# ==========================================================================
def fig06():
    import csv as _csv
    z = np.load(RESULTS / "ngsim_pairs.npz")
    with open(RESULTS / "ngsim_pairs.csv", encoding="utf-8-sig") as fh:
        meta = list(_csv.DictReader(fh))
    v_mean = np.array([float(r["v_mean"]) for r in meta])
    pair_id = [int(r["pair_id"]) for r in meta]
    lane = [int(float(r["lane"])) for r in meta]
    order = np.argsort(v_mean)
    picks = []
    for target in (3.0, 7.0, 11.0):
        j = order[np.argmin(np.abs(v_mean[order] - target))]
        picks.append(int(j))

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.0))
    for ax, j in zip(axes, picks):
        m = int(z["lengths"][j])
        t = z["t"][j, :m]
        t = t - t[0]
        gap_true = z["gap_net"][j, :m]
        lead_rear = z["lead_rear_y"][j, :m]
        lead_v = z["lead_v"][j, :m]
        own_front = z["own_front_y"][j, :m]
        own_v = z["own_v"][j, :m]

        tg = np.arange(0.0, t[-1] + 1e-9, 0.1)
        lead_rear = np.interp(tg, t, lead_rear)
        lead_v = np.interp(tg, t, lead_v)
        own_front = np.interp(tg, t, own_front)
        gap_true = np.interp(tg, t, gap_true)

        ax.plot(tg, gap_true, color="#111111", lw=2.4, label="Measured",
                zorder=5)
        for mk in MODELS:
            mdl = next(m for m in build_models() if m.key == mk)
            r = simulate_follower(mdl, tg, lead_rear, lead_v, own_front[0],
                                  own_v[0], gap_true[0], dt=0.05, v_min=0.0)
            ax.plot(tg, r["gap"], color=COLOR[mk], lw=1.6,
                    label=LABEL[mk], zorder=4)
        ax.plot(tg, lead_v, color=COLOR["lead"], lw=0, label=None)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Net spacing [m]")
        ax.set_title(f"pair {pair_id[j]}  (lane {lane[j]}, "
                     f"v ~ {v_mean[j]:.1f} m/s)", fontsize=10)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("场景 C  三条真实跟驰车对的净间距重放", fontsize=13.5, y=1.03)
    save(fig, "fig06_ngsim_examples.png")


# ==========================================================================
# fig07  圆周稳定性
# ==========================================================================
def fig07():
    d = load_json("validation.json")
    runs = d["stability"]["runs"]
    h = np.array([r["headway"] for r in runs])
    f = np.array([r["f"] for r in runs])
    go = np.array([r["growth_ovm"] for r in runs])
    gf = np.array([r["growth_fvdm"] for r in runs])

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ax.plot(h, go, "o-", color=COLOR["ovm"], lw=1.6, ms=6, label="OVM measured")
    ax.plot(h, gf, "s-", color=COLOR["fvdm"], lw=1.6, ms=6,
            label="FVDM measured")
    ax.axhline(3.0, color="#404040", lw=1.2, ls=":")
    ax.text(h.min(), 3.4, "instability threshold (growth = 3)", fontsize=7.5,
            color="#404040")
    ax.set_yscale("log")
    ax.set_xlabel("Headway [m]")
    ax.set_ylabel("Perturbation growth ratio  (log)")
    ax.set_title("环形路扰动增长比", fontsize=11)
    ax.legend(fontsize=8.5)

    ax = axes[1]
    ax.plot(h, 2 * f / 0.85, "o-", color=COLOR["ovm"], lw=1.6, ms=6,
            label="OVM  2f / kappa")
    ax.plot(h, 2 * f / (0.41 + 2 * 0.5), "s-", color=COLOR["fvdm"], lw=1.6,
            ms=6, label="FVDM  2f / (kappa + 2 lambda)")
    ax.axhline(1.0, color="#404040", lw=1.2, ls=":")
    ax.text(h.min(), 1.07, "stability boundary = 1", fontsize=7.5,
            color="#404040")
    ax.set_xlabel("Headway [m]")
    ax.set_ylabel("Stability index  (dimensionless)")
    ax.set_title("线性稳定性判据", fontsize=11)
    ax.legend(fontsize=8.5)

    fig.suptitle("复现校验  环形路稳定性：实测扰动增长与线性判据一致",
                 fontsize=13.5, y=1.03)
    save(fig, "fig07_ring_stability.png")


# ==========================================================================
if __name__ == "__main__":
    fig01()
    fig02()
    fig03()
    fig04()
    fig05()
    fig06()
    fig07()
    print("全部图片已写出到", RESULTS)
