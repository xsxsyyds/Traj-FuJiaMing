# -*- coding: utf-8 -*-
"""
三组对比实验：把 OVM / FVDM / IDM 放在同一套场景与同一套指标下比较。

对应 PPT 的三个场景与四个对比维度
--------------------------------
场景 A  前车匀速        —— 跟驰者从静止起步，测「速度响应」与起步加速度
场景 B  前车减速        —— 前车以不同减速度刹停，测「最小净间距」与「失败」
场景 C  真实跟驰(NGSIM) —— 用上一周 NGSIM I-80 的真实前车轨迹驱动，测「与真实轨迹误差」

指标
----
velocity response : 上升时间 t_rise(10%->90%)、稳态速度误差、峰值加速度
min net spacing   : 全程最小净间距（<=0 即碰撞）
trajectory error  : 相对真实跟驰者轨迹的 RMSE（位置 / 净间距 / 速度）
failure           : 碰撞、倒车、数值发散

输出
----
results/exp_a.json / exp_a_series.npz
results/exp_b.json / exp_b_series.npz
results/exp_c.json / exp_c_pairs.csv
results/experiments_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(HERE))
from car_following_models import (  # noqa: E402
    VEHICLE_LENGTH_M, build_models, equilibrium_speed, simulate_follower,
)

INT_DT = 0.05          # 内部积分步长
LOG = []


def say(s=""):
    LOG.append(str(s))
    print(s)


# ==========================================================================
# 工具
# ==========================================================================
def lead_profile_from_accel(t, x_front0, v0, a_of_t):
    """由前车加速度历史积分出前车车头位置历史（弹道格式）。"""
    x = np.empty_like(t)
    v = np.empty_like(t)
    x[0], v[0] = x_front0, v0
    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        a = float(a_of_t(t[i]))
        x[i + 1] = x[i] + v[i] * h + 0.5 * a * h * h
        v[i + 1] = v[i] + a * h
    return x, v


def rise_time(t, v, v_from, v_to, frac_lo=0.1, frac_hi=0.9):
    """10%->90% 上升时间；若未达到目标 90% 返回 nan。"""
    span = v_to - v_from
    if abs(span) < 1e-9:
        return float("nan")
    lo, hi = v_from + frac_lo * span, v_from + frac_hi * span
    if span > 0:
        i_lo = np.flatnonzero(v >= lo)
        i_hi = np.flatnonzero(v >= hi)
    else:
        i_lo = np.flatnonzero(v <= lo)
        i_hi = np.flatnonzero(v <= hi)
    if i_lo.size == 0 or i_hi.size == 0:
        return float("nan")
    return float(t[i_hi[0]] - t[i_lo[0]])


# ==========================================================================
# 场景 A —— 前车匀速
# ==========================================================================
def exp_a(v_lead_list=(6.0, 10.0, 14.0), t_end=150.0, t_step=0.05):
    say("=" * 78)
    say("场景 A  前车匀速：跟驰者从静止起步")
    say("=" * 78)
    say("  前车以恒定速度行驶，跟驰者初速 0、初始净间距取该模型对 v_lead 的平衡间距。")
    say("")
    say("  v_lead  模型   起步加速度  峰值加速度  t_rise(10-90%)  峰值速度  超调  稳态速度  稳态误差  最小净间距  失败")
    rows = []
    series = {}
    for v_lead in v_lead_list:
        n = int(round(t_end / t_step)) + 1
        t = np.arange(n) * t_step
        lead_rear = v_lead * t
        lead_v = np.full(n, float(v_lead))
        for mdl in build_models():
            gap0 = float(np.ravel(mdl.equilibrium_gap(v_lead))[0])
            x0 = 0.0 - gap0                       # 前车后杠 t=0 时在 0
            r = simulate_follower(mdl, t, lead_rear, lead_v, x0, 0.0, gap0,
                                  dt=INT_DT, v_min=0.0)
            v = r["v"]
            a = r["a"]
            v_ss = float(v[-1])
            err = abs(v_ss - v_lead)
            tr = rise_time(t, v, 0.0, v_lead)
            gap_min = float(np.min(r["gap"]))
            v_peak = float(np.max(v))
            overshoot = (v_peak - v_lead) / v_lead * 100.0
            rows.append(dict(
                v_lead=v_lead, model=mdl.key, gap0=gap0,
                a_start=float(a[1]), a_peak_acc=float(r["a_peak_acc"]),
                a_peak_dec=float(r["a_peak_dec"]),
                t_rise_s=tr, v_peak=v_peak, overshoot_pct=overshoot,
                t_peak_s=float(t[int(np.argmax(v))]),
                v_ss=v_ss, ss_err=err,
                gap_min=gap_min, failed=bool(r["failed"]),
                fail_reason=r["fail_reason"]))
            say(f"  {v_lead:5.1f}  {mdl.key:5s}  {a[1]:9.3f}  {r['a_peak_acc']:9.3f}"
                f"  {tr:13.2f}  {v_peak:8.3f}  {overshoot:6.1f}%  {v_ss:8.3f}"
                f"  {err:8.4f}  {gap_min:10.3f}"
                f"  {'是' if r['failed'] else '否'}")
            series[f"{mdl.key}|{v_lead:g}"] = dict(
                t=t, v=v, gap=r["gap"], a=a, lead_v=lead_v, lead_rear=lead_rear)
        say("")
    return rows, series


# ==========================================================================
# 场景 B —— 前车减速刹停
# ==========================================================================
def exp_b(v0=13.0, brake_list=(1.5, 3.0, 4.5, 6.0), t_brake_start=1.0,
          t_end=60.0, t_step=0.05):
    say("=" * 78)
    say("场景 B  前车减速：以恒定减速度刹停后停住")
    say("=" * 78)
    say("  初态：两车均以 v0 稳态行驶（间距 = 该模型平衡间距）。t=1 s 起前车以 b 减速到 0。")
    say("  跟驰者加「不倒车」约束（v >= 0）；另跑一次无约束版本，用于探测模型是否想倒车。")
    say("")
    say("  b[m/s2] 模型   最小净间距  终态净间距  峰值减速  失败  碰撞相对速度  失败时刻  想倒车")
    rows = []
    series = {}
    for b in brake_list:
        n = int(round(t_end / t_step)) + 1
        t = np.arange(n) * t_step
        t_stop = v0 / b                            # 前车停住时刻
        t_abs_stop = t_brake_start + t_stop

        def a_lead(tt, _b=b, _ts=t_brake_start, _tabs=t_abs_stop):
            if tt < _ts:
                return 0.0
            if tt < _tabs:
                return -_b
            return 0.0

        x_front, v_lead = lead_profile_from_accel(t, 0.0, v0, a_lead)
        for mdl in build_models():
            gap0 = float(np.ravel(mdl.equilibrium_gap(v0))[0])
            x0 = x_front[0] - gap0
            r = simulate_follower(mdl, t, x_front - VEHICLE_LENGTH_M, v_lead,
                                  x0, v0, gap0, dt=INT_DT, v_min=0.0)
            r_free = simulate_follower(mdl, t, x_front - VEHICLE_LENGTH_M,
                                       v_lead, x0, v0, gap0, dt=INT_DT,
                                       v_min=None)
            gap_min = float(np.min(r["gap"]))
            reverse = bool(np.min(r_free["v"]) < -1e-6)
            rows.append(dict(
                brake=b, model=mdl.key, gap0=gap0,
                gap_min=gap_min, gap_final=float(r["gap"][-1]),
                a_peak_dec=float(r["a_peak_dec"]),
                failed=bool(r["failed"]), fail_reason=r["fail_reason"],
                reverse_tendency=reverse,
                v_min_free=float(np.min(r_free["v"])),
                v_at_fail=(None if not r["failed"] else float(r["v_at_fail"])),
                t_fail=(None if not r["failed"] else float(r["t_fail"]))))
            vf = (f"{r['v_at_fail']:8.3f}" if r["failed"]
                  else f"{'—':>8s}")
            tf = (f"{r['t_fail']:9.2f}" if r["failed"] else f"{'—':>9s}")
            say(f"  {b:6.1f}  {mdl.key:5s}  {gap_min:10.3f}  {r['gap'][-1]:10.3f}"
                f"  {r['a_peak_dec']:8.3f}  {'是' if r['failed'] else '否':4s}"
                f"  {vf}  {tf}  {'是' if reverse else '否'}")
            series[f"{mdl.key}|{b:g}"] = dict(
                t=t, v=r["v"], gap=r["gap"], a=r["a"], lead_v=v_lead,
                lead_rear=x_front - VEHICLE_LENGTH_M)
        say("")
    return rows, series


# ==========================================================================
# 场景 C —— 真实跟驰（NGSIM）
# ==========================================================================
def _rmse(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(np.sqrt(np.mean(d * d)))


def exp_c(npz_path=None, max_pairs=None, t_step=0.1):
    say("=" * 78)
    say("场景 C  真实跟驰：用 NGSIM I-80 真实前车轨迹驱动")
    say("=" * 78)
    npz_path = Path(npz_path) if npz_path else RESULTS / "ngsim_pairs.npz"
    z = np.load(npz_path)
    lengths = z["lengths"]
    n_all = len(lengths)
    idx_list = range(n_all) if max_pairs is None else range(min(max_pairs, n_all))

    records = []
    for i in idx_list:
        m = int(lengths[i])
        if m < 20:
            continue
        t = z["t"][i, :m].copy()
        lead_rear = z["lead_rear_y"][i, :m]
        lead_v = z["lead_v"][i, :m]
        own_front = z["own_front_y"][i, :m]
        own_v = z["own_v"][i, :m]
        gap_true = z["gap_net"][i, :m]
        # 真实轨迹可能有轻微非等间隔，重采样到等间隔
        t = t - t[0]
        tg = np.arange(0.0, t[-1] + 1e-9, t_step)
        lead_rear = np.interp(tg, t, lead_rear)
        lead_v = np.interp(tg, t, lead_v)
        own_front = np.interp(tg, t, own_front)
        own_v = np.interp(tg, t, own_v)
        gap_true = np.interp(tg, t, gap_true)
        t = tg

        # 匀速基线
        base_front = own_front[0] + own_v[0] * t
        rec_base = dict(pair_id=int(z["pair_ids"][i]), model="baseline",
                        rmse_pos=_rmse(base_front, own_front),
                        rmse_gap=float("nan"), rmse_v=float("nan"),
                        gap_min=float(np.min(gap_true)), failed=False,
                        n=len(t), v_mean=float(np.mean(own_v)))

        for mdl in build_models():
            r = simulate_follower(mdl, t, lead_rear, lead_v,
                                  own_front[0], own_v[0], gap_true[0],
                                  dt=INT_DT, v_min=0.0)
            # 注意：lead_rear 是三个模型的共同输入，故
            #   gap_sim - gap_true = (lead_rear - x) - (lead_rear - own_front)
            #                      = own_front - x
            # 于是 rmse_gap 与 rmse_pos 在数学上恒等，不是两个独立指标。
            # 保留两列只是为了让「间距误差」这一说法在输出里可见。
            gap_sim = r["gap"]
            records.append(dict(
                pair_id=int(z["pair_ids"][i]), model=mdl.key,
                rmse_pos=_rmse(r["x"], own_front),
                rmse_gap=_rmse(gap_sim, gap_true),
                rmse_v=_rmse(r["v"], own_v),
                gap_min=float(np.min(gap_sim)),
                failed=bool(r["failed"]), fail_reason=r["fail_reason"],
                n=len(t), v_mean=float(np.mean(own_v))))
        records.append(rec_base)

    # 汇总
    say("")
    say("  模型        对数   RMSE位置 中位   均值   P90    RMSE间距 中位    RMSE速度 中位   失败率")
    by_model = {}
    for key in ("baseline", "ovm", "fvdm", "idm"):
        sub = [r for r in records if r["model"] == key]
        if not sub:
            continue
        pos = np.array([r["rmse_pos"] for r in sub])
        gp = np.array([r["rmse_gap"] for r in sub if np.isfinite(r["rmse_gap"])])
        gv = np.array([r["rmse_v"] for r in sub if np.isfinite(r["rmse_v"])])
        nf = sum(1 for r in sub if r["failed"])
        by_model[key] = dict(
            n=len(sub), rmse_pos_median=float(np.median(pos)),
            rmse_pos_mean=float(np.mean(pos)),
            rmse_pos_p90=float(np.percentile(pos, 90)),
            rmse_gap_median=(float(np.median(gp)) if gp.size else None),
            rmse_v_median=(float(np.median(gv)) if gv.size else None),
            n_failed=int(nf), fail_rate=float(nf / len(sub)))
        say(f"  {key:10s}  {len(sub):4d}  {np.median(pos):14.3f}"
            f"  {np.mean(pos):6.3f}  {np.percentile(pos, 90):5.3f}"
            f"  {np.median(gp) if gp.size else float('nan'):14.3f}"
            f"  {np.median(gv) if gv.size else float('nan'):14.3f}"
            f"  {nf / len(sub) * 100:6.1f}%")
    say("")
    return records, by_model


# ==========================================================================
def main():
    a_rows, a_series = exp_a()
    b_rows, b_series = exp_b()
    c_records, c_summary = exp_c()

    def dump_npz(path, series):
        np.savez_compressed(path, **{
            f"{k}__{f}": v[f] for k, v in series.items() for f in v})

    dump_npz(RESULTS / "exp_a_series.npz", a_series)
    dump_npz(RESULTS / "exp_b_series.npz", b_series)

    (RESULTS / "exp_a.json").write_text(
        json.dumps(a_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "exp_b.json").write_text(
        json.dumps(b_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "exp_c.json").write_text(
        json.dumps({"summary": c_summary, "pairs": c_records},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    import csv
    with open(RESULTS / "exp_c_pairs.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(c_records[0].keys()))
        w.writeheader()
        w.writerows(c_records)

    summary = dict(exp_a=a_rows, exp_b=b_rows, exp_c=c_summary)
    (RESULTS / "experiments_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "experiments_report.txt").write_text(
        "\n".join(LOG), encoding="utf-8")
    print("已写出 results/exp_a.json / exp_b.json / exp_c.json / experiments_summary.json")


if __name__ == "__main__":
    main()
