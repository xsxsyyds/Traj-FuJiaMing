# -*- coding: utf-8 -*-
"""week04 课堂作业：用 NGSIM I-80 车辆轨迹绘制交通流基本图（流量—密度图）。

对应 PPT 三步：
  1 固定分析范围 —— 同一方向、同一车道、100 米路段；统一单位，检查缺失与重复
  2 绘制基本图   —— 用多种方法算密度与流量，画两张同坐标范围的流量—密度图
  3 解释差异     —— 比较点云覆盖、散布与趋势

「多种方法」体现在两个维度，各出一张图：
  · 窗口长度：T = 10 s（90 个样本） vs T = 60 s（15 个样本）—— 同一估计量、不同聚合粒度
  · 估计量  ：A Edie 广义定义（用 100 m 路段） vs B 断面计数法（虚拟检测器，只用一条断面）

计算口径
--------
A. Edie 广义定义（PPT 计算提示）：
    k = Σt_i / (L · T)      q = Σd_i / (L · T)      v = q / k
其中 t_i 为第 i 辆车在窗口内的**停留时间**，d_i 为其在窗口内的**纵向行驶距离**，
L = 100 m 为路段长度，T 为窗口长度。

B. 断面计数法（模拟单车道环形线圈，断面设在路段中点）：
    q   = N_cross / T                        （辆/h）
    v_s = 跨越断面车辆速度的**调和平均**      （空间平均速度，km/h）
    k   = q / v_s                            （辆/km）
与 A 的根本差别：B 只用「一条断面 + 速度」，结果与 L 无关；A 显式使用 L，
把每辆车的行驶距离逐段计入。两者在无换道的均匀流下应当一致。

离散实现（10 Hz 采样网格，Δt = 0.1 s）：
  把每辆车按 time_s 排序，取相邻两帧组成一个「区间」；
  **仅当两帧都在所选车道、且都在所选路段内**时，该区间才计入，
  贡献 Δt = t_{右} - t_{左} 与 Δy = y_{右} - y_{左}。
  这样车辆驶入/驶出路段或换道的那个区间会被自然丢弃，每处最多截断 0.1 s。
  区间跨窗口边界时按时间比例拆到两个窗口。

用法：
    python analyze_fundamental_diagram.py                 # 默认 车道 6 @ 158-258 m
    python analyze_fundamental_diagram.py --lane 4 --y0 270   # 换分析范围只改这两个参数
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = Path(r"E:\jiaotongjianmo\week04\第4周_NGSIM-I80_课堂数据"
                   r"\NGSIM_I80_20050413_1600-1615_SI.csv")
RESULTS = HERE / "results"

USE_COLS = ["vehicle_id", "frame_id", "time_s", "local_x_m", "local_y_m",
            "vehicle_length_m", "vehicle_class", "speed_m_s",
            "acceleration_m_s2", "lane_id"]

L_SEG = 100.0          # 路段长度 m
WINDOWS = (10.0, 60.0)  # 两种窗口长度 s


# --------------------------------------------------------------------------
# 1) 读数据 + 质量检查
# --------------------------------------------------------------------------
def load_and_check(csv: Path, lane: int, y0: float, y1: float):
    if not csv.is_file():
        raise SystemExit(f"找不到数据文件：{csv}")
    df = pd.read_csv(csv, usecols=USE_COLS)

    checks = {
        "file": str(csv),
        "file_mb": round(csv.stat().st_size / 1e6, 1),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "vehicles_total": int(df.vehicle_id.nunique()),
        "missing_total": int(df.isna().sum().sum()),
        "dup_full_rows": int(df.duplicated().sum()),
        "dup_vehicle_frame": int(df.duplicated(subset=["vehicle_id", "frame_id"]).sum()),
        "time_s": [float(df.time_s.min()), float(df.time_s.max())],
        "n_timestamps": int(df.time_s.nunique()),
        "dt_median_s": float(np.median(np.diff(np.sort(df.time_s.unique())))),
        "local_y_m": [float(df.local_y_m.min()), float(df.local_y_m.max())],
        "local_x_m": [float(df.local_x_m.min()), float(df.local_x_m.max())],
        "lane_ids": sorted(int(v) for v in df.lane_id.unique()),
        "segment": {"lane": int(lane), "y0": float(y0), "y1": float(y1),
                    "length_m": float(y1 - y0)},
    }
    return df, checks


def lane_coverage(df: pd.DataFrame, lane: int):
    """所选车道在所选路段的覆盖情况（PPT 第 1 步要求先看）。"""
    sub = df[df.lane_id == lane]
    y = sub.local_y_m.values
    out = {
        "rows_in_lane": int(len(sub)),
        "vehicles_in_lane": int(sub.vehicle_id.nunique()),
        "y_span_m": [float(y.min()), float(y.max())],
        "lane_speed_median_m_s": float(sub.speed_m_s.median()),
    }
    return out


# --------------------------------------------------------------------------
# 2) Edie 估计量
# --------------------------------------------------------------------------
def edie_windows(df: pd.DataFrame, lane: int, y0: float, y1: float, T: float,
                 t_end: float | None = None):
    """返回各窗口的 Edie 估计量。

    注意 t_end 必须用**数据集全局**的时间上界：df 是按 (vehicle_id, time_s) 排序的，
    `t[-1]` 只是最后一辆车的最后时刻，不能拿来定窗口总数。
    """
    if t_end is None:
        t_end = float(df.time_s.max()) + 0.1
    m = ((df.lane_id == lane)
         & (df.local_y_m >= y0) & (df.local_y_m < y1))
    s = df.loc[m, ["vehicle_id", "time_s", "local_y_m"]].sort_values(
        ["vehicle_id", "time_s"])

    veh = s.vehicle_id.values
    t = s.time_s.values
    y = s.local_y_m.values

    # 相邻两帧：同一辆车、且两帧都满足「车道 + 路段」条件
    same_veh = veh[1:] == veh[:-1]
    t_l, t_r = t[:-1][same_veh], t[1:][same_veh]
    y_l, y_r = y[:-1][same_veh], y[1:][same_veh]
    v_l = veh[:-1][same_veh]

    dt = t_r - t_l
    dy = y_r - y_l
    # 丢掉异常区间（时间倒序或非 0.1 s 量级）
    ok = (dt > 0) & (dt <= 1.0)
    t_l, t_r, dt, dy, v_l = t_l[ok], t_r[ok], dt[ok], dy[ok], v_l[ok]
    n_neg_dy = int((dy < 0).sum())

    # 跨窗口边界时按时间比例拆分
    w1 = np.floor(t_l / T).astype(np.int64)
    w2 = np.floor((t_r - 1e-9) / T).astype(np.int64)
    split = w1 != w2

    tot = int(np.ceil(t_end / T - 1e-9))
    if tot <= 0:
        raise SystemExit(f"窗口数计算异常：t_end={t_end}, T={T}")
    sum_t = np.zeros(tot + 1)
    sum_d = np.zeros(tot + 1)

    # 不跨界的区间：整块计入
    np.add.at(sum_t, w1[~split], dt[~split])
    np.add.at(sum_d, w1[~split], dy[~split])

    # 跨界的区间：按时间比例拆成两段
    if split.any():
        b = (w1[split] + 1).astype(float) * T          # 边界时刻
        dt1 = b - t_l[split]
        frac = np.divide(dt1, dt[split], out=np.zeros_like(dt1), where=dt[split] > 0)
        np.add.at(sum_t, w1[split], dt1)
        np.add.at(sum_d, w1[split], dy[split] * frac)
        np.add.at(sum_t, w2[split], dt[split] - dt1)
        np.add.at(sum_d, w2[split], dy[split] * (1.0 - frac))

    # 每窗口的车辆数（去重）
    n_veh = np.zeros(tot + 1, dtype=int)
    wd = pd.DataFrame({"w": w1, "v": v_l}).drop_duplicates()
    cnt = wd.groupby("w").size()
    n_veh[cnt.index.values] = cnt.values

    sum_t, sum_d = sum_t[:tot], sum_d[:tot]
    n_veh = n_veh[:tot]

    k = sum_t / (L_SEG * T)          # veh/m
    q = sum_d / (L_SEG * T)          # veh/s
    v = np.divide(q, k, out=np.full_like(q, np.nan), where=k > 0)

    info = {
        "T_s": float(T),
        "n_windows": int(tot),
        "intervals": int(len(dt)),
        "intervals_split": int(split.sum()),
        "negative_dy_intervals": n_neg_dy,
    }
    return {
        "k_veh_per_km": k * 1000.0,
        "q_veh_per_h": q * 3600.0,
        "v_km_per_h": v * 3.6,
        "sum_t_s": sum_t,
        "sum_d_m": sum_d,
        "n_vehicles_per_window": n_veh,
        "info": info,
    }


# --------------------------------------------------------------------------
# 2b) 断面计数法（虚拟检测器）
# --------------------------------------------------------------------------
def detector_windows(df: pd.DataFrame, lane: int, y0: float, y1: float, T: float,
                     t_end: float | None = None, y_det: float | None = None,
                     speed_col: str = "speed_m_s"):
    """断面计数法（模拟单车道环形线圈）估计量。

    在路段内设一条横向断面 y_det（默认路段中点），统计车辆跨越该断面的次数：
        q   = N_cross / T
        v_s = 跨越断面车辆速度的调和平均（空间平均速度）
        k   = q / v_s
    与 Edie 广义定义的差别见文件头说明。
    """
    if t_end is None:
        t_end = float(df.time_s.max()) + 0.1
    if y_det is None:
        y_det = 0.5 * (y0 + y1)

    m = ((df.lane_id == lane)
         & (df.local_y_m >= y0) & (df.local_y_m < y1))
    s = df.loc[m, ["vehicle_id", "time_s", "local_y_m", speed_col]] \
          .sort_values(["vehicle_id", "time_s"])

    veh = s.vehicle_id.values
    t = s.time_s.values
    y = s.local_y_m.values
    v = s[speed_col].values

    same_veh = veh[1:] == veh[:-1]
    t_l, t_r = t[:-1][same_veh], t[1:][same_veh]
    y_l, y_r = y[:-1][same_veh], y[1:][same_veh]
    v_l, v_r = v[:-1][same_veh], v[1:][same_veh]
    dt = t_r - t_l
    ok = (dt > 0) & (dt <= 1.0)
    t_l, t_r, y_l, y_r, v_l, v_r, dt = (a[ok] for a in
                                        (t_l, t_r, y_l, y_r, v_l, v_r, dt))

    # 向上跨越断面（本数据纵向递增，Δy ≥ 0）
    cross = (y_l < y_det) & (y_r >= y_det)
    span = (y_r - y_l)[cross]
    frac = np.divide(y_det - y_l[cross], span,
                     out=np.zeros(int(cross.sum())), where=span > 0)
    t_c = t_l[cross] + frac * dt[cross]
    v_c = v_l[cross] + frac * (v_r - v_l)[cross]        # 断面处速度按比例插值

    tot = int(np.ceil(t_end / T - 1e-9))
    idx = np.floor(t_c / T).astype(np.int64)
    keep = (idx >= 0) & (idx < tot)
    idx, v_c = idx[keep], v_c[keep]

    n_cross = np.bincount(idx, minlength=tot)[:tot].astype(float)
    inv = np.zeros(tot)
    good = v_c > 1e-6
    np.add.at(inv, idx[good], 1.0 / v_c[good])
    with np.errstate(divide="ignore", invalid="ignore"):
        v_s_ms = np.where(inv > 0, n_cross / inv, np.nan)   # 调和平均，m/s
    q = n_cross / T * 3600.0                                # veh/h
    v_s_kmh = v_s_ms * 3.6
    k = np.divide(q, v_s_kmh, out=np.zeros_like(q),
                  where=np.isfinite(v_s_kmh) & (v_s_kmh > 0))

    info = {
        "T_s": float(T),
        "n_windows": int(tot),
        "y_detector_m": float(y_det),
        "crossings_total": int(len(t_c)),
        "windows_zero_crossing": int((n_cross == 0).sum()),
    }
    return {
        "k_veh_per_km": k,
        "q_veh_per_h": q,
        "v_km_per_h": v_s_kmh,
        "n_vehicles_per_window": n_cross,
        "info": info,
    }


# --------------------------------------------------------------------------
# 3) 绘图
# --------------------------------------------------------------------------
COLORS = {10.0: "#C00000", 60.0: "#1F4E79"}
DEFAULT_FIGSIZE = (12.6, 6.2)


def build_time_cmap(base_hex: str, n: int = 96):
    """浅灰（低饱和、高亮度）→ base_hex（高饱和、低亮度）的单色渐变。

    用来把「时间先后」编码成颜色：早的窗口浅、晚的窗口深，
    灰度与饱和度同时单调变化，所以不连线也能看出时间维度的走向。
    """
    import colorsys
    from matplotlib.colors import LinearSegmentedColormap, to_rgb

    r, g, b = to_rgb(base_hex)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s0, v0 = 0.10, 0.86                    # 起点：浅灰、几乎无饱和
    stops = [colorsys.hsv_to_rgb(h, s0 + (s - s0) * i / n,
                                 v0 + (v - v0) * i / n)
             for i in range(n + 1)]
    return LinearSegmentedColormap.from_list("t" + base_hex.lstrip("#"), stops)


def shared_lims(series_list):
    """按若干组结果算统一坐标范围（供「两图同坐标范围」使用）。"""
    ks = np.concatenate([r["k_veh_per_km"] for s in series_list
                         for r in s.values()])
    qs = np.concatenate([r["q_veh_per_h"] for s in series_list
                         for r in s.values()])
    ks, qs = ks[np.isfinite(ks)], qs[np.isfinite(qs)]
    xlim = (0.0, max(5.0, np.ceil(ks.max() / 10) * 10 + 10))
    ylim = (0.0, max(200.0, np.ceil(qs.max() / 200) * 200 + 200))
    return xlim, ylim


def build_panels(res, t_range, tag: str):
    """把估计结果整理成绘图用的逐窗口数据（含各面板角标统计）。"""
    panels = {}
    for T in WINDOWS:
        r = res[T]
        k, q = r["k_veh_per_km"], r["q_veh_per_h"]
        n = r["info"]["n_windows"]
        vv = r["v_km_per_h"][np.isfinite(r["v_km_per_h"])]
        nv = r["n_vehicles_per_window"]
        if tag == "edie":
            q_line = f"q: {np.nanmin(q):.0f}–{np.nanmax(q):.0f} veh/h"
            unit = [f"每窗口车辆数 {int(np.min(nv))}–{int(np.max(nv))}",
                    f"v 中位 {np.median(vv):.0f} km/h"]
        else:
            q_line = (f"q: {np.nanmin(q):.0f}–{np.nanmax(q):.0f} veh/h"
                      f"（步长 {3600.0 / T:.0f}）")
            unit = [f"每窗口跨越断面 {int(np.min(nv))}–{int(np.max(nv))} 辆",
                    f"v 空间中位 {np.median(vv):.0f} km/h"]
        panels[T] = {
            "k": k, "q": q,
            "t0": np.arange(n) * T,
            "sub": [f"n = {n} 个窗口",
                    f"k: {np.nanmin(k):.1f}–{np.nanmax(k):.1f} veh/km",
                    q_line] + unit,
            "n_zero": int((nv == 0).sum()) if tag != "edie" else 0,
        }
    return panels


def plot(panels, seg, out_png: Path, *, method_cn: str, method_en: str,
         lims, t_range, colors=None):
    """画流量—密度图。两个面板同坐标范围，颜色深浅表示时间先后（不连线）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import Normalize

    fam = "Microsoft YaHei"
    if fam not in {f.name for f in font_manager.fontManager.ttflist}:
        fam = "DejaVu Sans"
    plt.rcParams["font.family"] = fam
    plt.rcParams["axes.unicode_minus"] = False

    xlim, ylim = lims
    colors = colors or COLORS
    norm = Normalize(t_range[0], t_range[1])

    fig = plt.figure(figsize=DEFAULT_FIGSIZE)
    # 2×2：上排两个面板，下排各配一条横置色标（横置可避免与相邻面板的刻度打架）
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.030], hspace=0.55,
                          wspace=0.085, left=0.070, right=0.975,
                          top=0.845, bottom=0.135)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    cax0 = fig.add_subplot(gs[1, 0])
    cax1 = fig.add_subplot(gs[1, 1])

    for ax, cax, T in ((ax0, cax0, WINDOWS[0]), (ax1, cax1, WINDOWS[1])):
        p = panels[T]
        k, q, t0 = p["k"], p["q"], p["t0"]
        m = np.isfinite(k) & np.isfinite(q)

        # 参考等速线 v = q/k（灰虚线）
        for v_kph in (20, 40, 60):
            ax.plot(np.array([0.0, xlim[1]]), np.array([0.0, v_kph * xlim[1]]),
                    ":", color="#CFCFCF", lw=0.9, zorder=1)
            x_lab = min(xlim[1] * 0.86, ylim[1] / v_kph * 0.88)
            if x_lab > xlim[1] * 0.15:
                ax.text(x_lab, v_kph * x_lab, f"v={v_kph}", fontsize=7.5,
                        color="#AAAAAA", ha="right", va="bottom", zorder=1)

        sc = ax.scatter(k[m], q[m], c=t0[m], cmap=build_time_cmap(colors[T]),
                        norm=norm, s=40, edgecolors="#8C8C8C",
                        linewidths=0.35, zorder=3)
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cb.set_label("window start time (s)", fontsize=9)
        cb.ax.tick_params(labelsize=8, length=3, pad=2)
        cb.outline.set_edgecolor("#BBBBBB")

        sub = list(p["sub"])
        if p["n_zero"]:
            sub.append(f"其中 {p['n_zero']} 个窗口无车通过（q=0）")
        ax.set_title(f"窗口 T = {T:.0f} s（{len(k)} 个样本）")
        ax.set_xlabel("density k (veh/km)")
        ax.grid(True, color="#E8E8E8", lw=0.6)
        ax.set_axisbelow(True)
        # 统计框放在右上角：四个面板的点云都在左下—中部，右上始终是空的
        ax.text(0.965, 0.965, "\n".join(sub), transform=ax.transAxes,
                va="top", ha="right", fontsize=8.5, color="#3A3A3A",
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec="#CCCCCC", alpha=0.9))

    ax0.set_ylabel("flow q (veh/h)")
    for ax in (ax0, ax1):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    fig.suptitle(
        f"交通流基本图（{method_cn}）：车道 {seg['lane']}，"
        f"路段 {seg['y0']:.0f}–{seg['y1']:.0f} m（L = {seg['length_m']:.0f} m）",
        fontsize=13)
    fig.text(0.5, 0.902,
             f"颜色深浅 = 时间先后（0 → {t_range[1]:.0f} s）；"
             f"两图同坐标范围 k 0–{xlim[1]:.0f} veh/km、q 0–{ylim[1]:.0f} veh/h；"
             f"估计量 = {method_en}",
             ha="center", fontsize=9.5, color="#555555")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------
def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="NGSIM I-80 交通流基本图")
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--lane", type=int, default=6)
    ap.add_argument("--y0", type=float, default=158.0)
    ap.add_argument("--y1", type=float, default=258.0)
    args = ap.parse_args()

    csv = Path(args.csv)
    lane, y0, y1 = args.lane, args.y0, args.y1
    if y1 - y0 != L_SEG:
        print(f"[warn] 路段长度 {y1-y0:.0f} m ≠ {L_SEG:.0f} m，"
              f"但归一化仍按 L={L_SEG:.0f} m 计算，请自行确认")

    print(f"读取 {csv.name} ...")
    df, checks = load_and_check(csv, lane, y0, y1)
    cov = lane_coverage(df, lane)
    t_end = float(df.time_s.max()) + 0.1      # 全局时间上界（= 900.0 s）

    print("=" * 74)
    print("第 1 步 固定分析范围与数据检查")
    print("=" * 74)
    print(f"  数据文件   : {checks['file']}")
    print(f"  规模       : {checks['rows']:,} 行 × {len(checks['columns'])} 列"
          f"，{checks['vehicles_total']:,} 辆车，{checks['file_mb']} MB")
    print(f"  时间       : {checks['time_s'][0]:.1f}–{checks['time_s'][1]:.1f} s，"
          f"{checks['n_timestamps']:,} 个时间戳，步长 {checks['dt_median_s']:.3f} s")
    print(f"  缺失/重复  : 缺失 {checks['missing_total']}，"
          f"整行重复 {checks['dup_full_rows']}，"
          f"(vehicle_id,frame_id) 重复 {checks['dup_vehicle_frame']}")
    print(f"  分析范围   : 车道 {lane}，纵向 {y0:.0f}–{y1:.0f} m（L = {y1-y0:.0f} m）")
    print(f"  该车道     : {cov['rows_in_lane']:,} 条记录、"
          f"{cov['vehicles_in_lane']} 辆车，"
          f"y 跨度 {cov['y_span_m'][0]:.1f}–{cov['y_span_m'][1]:.1f} m，"
          f"中位车速 {cov['lane_speed_median_m_s']:.2f} m/s")

    print()
    resA, resB = {}, {}
    for T in WINDOWS:
        resA[T] = edie_windows(df, lane, y0, y1, T, t_end=t_end)
        resB[T] = detector_windows(df, lane, y0, y1, T, t_end=t_end)
        i = resA[T]["info"]
        print(f"第 2 步 T={T:.0f} s：{i['n_windows']} 个窗口，"
              f"计入区间 {i['intervals']:,} 个（跨界拆分 {i['intervals_split']:,}，"
              f"Δy<0 的噪声区间 {i['negative_dy_intervals']}）")
        for tag, r in (("A 广义(Edie)", resA[T]), ("B 断面计数 ", resB[T])):
            k, q = r["k_veh_per_km"], r["q_veh_per_h"]
            print(f"        {tag}  k 中位 {np.nanmedian(k):6.1f}"
                  f"（IQR {np.nanpercentile(k, 25):6.1f}–"
                  f"{np.nanpercentile(k, 75):6.1f}）veh/km，"
                  f"q 中位 {np.nanmedian(q):6.0f}"
                  f"（IQR {np.nanpercentile(q, 25):6.0f}–"
                  f"{np.nanpercentile(q, 75):6.0f}）veh/h")

    t_range = (0.0, float(np.ceil(t_end / 60.0) * 60.0))
    limsA = shared_lims([resA])
    limsB = shared_lims([resB])
    lims_all = shared_lims([resA, resB])
    print()
    print(f"        坐标范围：A  k 0–{limsA[0][1]:.0f} / q 0–{limsA[1][1]:.0f}"
          f"；B  k 0–{limsB[0][1]:.0f} / q 0–{limsB[1][1]:.0f}"
          f"；出图统一用 A∪B  k 0–{lims_all[0][1]:.0f} / q 0–{lims_all[1][1]:.0f}")

    # A 与 B 的全时段汇总对照（与窗口长度无关，只看总账）
    sum_d_total = float(np.sum(resA[60.0]["sum_d_m"]))
    n_cross_total = int(resB[60.0]["info"]["crossings_total"])
    q_a_mean = sum_d_total * 3600.0 / (L_SEG * t_end)
    q_b_mean = n_cross_total * 3600.0 / t_end
    mean_run = sum_d_total / n_cross_total
    ab = {
        "sum_d_m": sum_d_total,
        "crossings": n_cross_total,
        "q_A_mean_veh_per_h": q_a_mean,
        "q_B_mean_veh_per_h": q_b_mean,
        "q_B_over_q_A": q_b_mean / q_a_mean,
        "mean_run_length_m": mean_run,
        "L_over_mean_run": L_SEG / mean_run,
        "y_detector_m": resB[60.0]["info"]["y_detector_m"],
    }
    print()
    print("        第 2b 步 A/B 全时段总账（解释 B 为何系统性偏高）：")
    print(f"          断面设在 y = {ab['y_detector_m']:.0f} m，"
          f"全时段穿越 {ab['crossings']} 次")
    print(f"          A 平均流量 Σd*3600/(L*T_total) = {q_a_mean:.0f} veh/h")
    print(f"          B 平均流量 N_cross*3600/T_total = {q_b_mean:.0f} veh/h"
          f"  -> B/A = {ab['q_B_over_q_A']:.3f}")
    print(f"          平均每次穿越的纵向长度 Σd/N_cross = {mean_run:.1f} m"
          f"，而 L = {L_SEG:.0f} m"
          f"  -> L/均值 = {ab['L_over_mean_run']:.3f}（与 B/A 一致）")

    seg = checks["segment"]
    outA = RESULTS / "fd_10s_vs_60s.png"
    outB = RESULTS / "fd_detector_10s_vs_60s.png"
    # 两张**图**也共用同一坐标范围，这样并排看不会因为坐标尺度不同而误读散布
    plot(build_panels(resA, t_range, "edie"), seg, outA,
         method_cn="A 广义定义（Edie）",
         method_en="Edie generalised definition",
         lims=lims_all, t_range=t_range)
    plot(build_panels(resB, t_range, "detector"), seg, outB,
         method_cn="B 断面计数法（虚拟检测器）",
         method_en="section counting / virtual loop detector",
         lims=lims_all, t_range=t_range)
    print()
    print("第 3 步 出图：")
    print(f"        {outA.name}  ← 方法 A（Edie 广义定义）")
    print(f"        {outB.name}  ← 方法 B（断面计数法）")

    # 落盘：逐窗口明细 + 汇总
    RESULTS.mkdir(parents=True, exist_ok=True)
    for T in WINDOWS:
        r = resA[T]
        pd.DataFrame({
            "window_index": np.arange(r["info"]["n_windows"]),
            "t_start_s": np.arange(r["info"]["n_windows"]) * T,
            "sum_t_s": r["sum_t_s"],
            "sum_d_m": r["sum_d_m"],
            "n_vehicles": r["n_vehicles_per_window"],
            "k_veh_per_km": r["k_veh_per_km"],
            "q_veh_per_h": r["q_veh_per_h"],
            "v_km_per_h": r["v_km_per_h"],
        }).to_csv(RESULTS / f"fd_windows_{int(T)}s.csv", index=False,
                  encoding="utf-8-sig")

        r = resB[T]
        pd.DataFrame({
            "window_index": np.arange(r["info"]["n_windows"]),
            "t_start_s": np.arange(r["info"]["n_windows"]) * T,
            "n_crossings": r["n_vehicles_per_window"],
            "k_veh_per_km": r["k_veh_per_km"],
            "q_veh_per_h": r["q_veh_per_h"],
            "v_km_per_h": r["v_km_per_h"],
        }).to_csv(RESULTS / f"fd_windows_detector_{int(T)}s.csv", index=False,
                  encoding="utf-8-sig")

    def _summ(res):
        out = {}
        for T in WINDOWS:
            r = res[T]
            k, q = r["k_veh_per_km"], r["q_veh_per_h"]
            out[f"{int(T)}s"] = {
                "T_s": T,
                "n_windows": r["info"]["n_windows"],
                "k_veh_per_km": [float(np.nanmin(k)), float(np.nanmax(k))],
                "k_median": float(np.nanmedian(k)),
                "k_iqr": [float(np.nanpercentile(k, 25)),
                          float(np.nanpercentile(k, 75))],
                "q_veh_per_h": [float(np.nanmin(q)), float(np.nanmax(q))],
                "q_median": float(np.nanmedian(q)),
                "q_iqr": [float(np.nanpercentile(q, 25)),
                          float(np.nanpercentile(q, 75))],
                "v_km_per_h_median": float(np.nanmedian(r["v_km_per_h"])),
                "n_per_window": [float(np.min(r["n_vehicles_per_window"])),
                                 float(np.max(r["n_vehicles_per_window"]))],
                "info": r["info"],
            }
        return out

    summary = {
        "checks": checks,
        "lane_coverage": cov,
        "convention": {
            "A_edie": ("相邻两帧都在所选车道且都在所选路段内才计入；"
                       "贡献 Δt 与 Δy；跨窗口边界按时间比例拆分；"
                       "k=Σt/(L·T)，q=Σd/(L·T)，v=q/k"),
            "B_detector": ("在路段中点设虚拟断面，统计跨越次数；"
                           "q=N_cross/T；v_s=跨越车辆速度的调和平均；k=q/v_s"),
            "time_colour": "点色由窗口起始时间决定（浅→深），不使用时间顺序连线",
        },
        "method_A_edie": _summ(resA),
        "method_B_detector": _summ(resB),
        "A_vs_B": ab,
        "figures": [str(outA), str(outB)],
        "axes": {"A": limsA, "B": limsB, "A_and_B": lims_all},
    }
    (RESULTS / "fd_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("        明细：fd_windows_{10s,60s}.csv、"
          "fd_windows_detector_{10s,60s}.csv；汇总：fd_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
