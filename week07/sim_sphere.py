# -*- coding: utf-8 -*-
"""
三维球面对趾场景：100 架无人机从球面出发，直线穿过球心飞向对趾点。

与"贴球面飞"不同，这里是**三维自由空间**：
    起点  均匀分布在半径 R 的球面上
    终点  关于球心对称的点  goal = -p0
    轨迹  两点之间的直线段（穿过球体内部），不施加任何边界约束

场景参数（按课堂要求）
----------------------
    N = 100，R = 10 m，安全半径 0.5 m（间隔 >= 1 m）
    vmax = 2 m/s，Δt = 0.1 s，τ = 5 s，初速 0
    距目标 <= 1 m 即停留，限时 120 s，同步更新

对比模型：orca / direct（不避让）/ repulsion（排斥场）
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))

from orca3d import (  # noqa: E402
    uniform_sphere, nn_stats, solve_velocity, region_colors,
)


def cbf_qp_velocity(pos_i, vel_pref, neighbors, safe_distance, vmax,
                     alpha=2.0):
    """三维控制屏障函数 QP 的快速投影解。

    对 h_ij=||p_i-p_j||²-D² 使用
    d h/dt + alpha*h >= 0，得到关于 v_i 的线性约束；
    在速度球内求最接近期望速度的可行点。只规划速度，不修改位置。
    """
    v = np.asarray(vel_pref, dtype=float).copy()
    nv = np.linalg.norm(v)
    if nv > vmax:
        v *= vmax / max(nv, 1e-12)
    constraints = []
    D2 = float(safe_distance) ** 2
    for pos_j, vel_j, _rj, _static in neighbors:
        r = np.asarray(pos_i) - np.asarray(pos_j)
        h = float(r @ r - D2)
        # r·(v_i-v_j) >= -alpha*h/2
        constraints.append((r, float(r @ vel_j - 0.5 * alpha * h)))
    for _ in range(8):
        worst = 0.0
        for n, b in constraints:
            gap = b - float(n @ v)
            if gap > 0.0:
                nn = float(n @ n)
                if nn > 1e-12:
                    v += (gap / nn) * n
                    worst = max(worst, gap)
        nv = np.linalg.norm(v)
        if nv > vmax:
            v *= vmax / max(nv, 1e-12)
        if worst <= 1e-7:
            break
    return v

# ---- 场景参数 ----
N_DRONE = 100
RADIUS = 10.0
V_MAX = 2.0
DT = 0.1
TAU = 5.0
T_END = 120.0
SAFE_RADIUS = 0.5
MIN_SEP = 2 * SAFE_RADIUS
ARRIVE_TOL = 1.0
NEIGHBOR_DIST = 12.0       # 三维空间里只考虑 12 m 内的邻居
MAX_NEIGHBORS = 24
#: ORCA 约束余量：理论 1.0 m，离散化/投影/不可行都会吃掉几毫米
ORCA_MARGIN = 0.05


def _repulsion(p, v, arr):
    """简单排斥场：在期望速度上叠加邻居的指数排斥。"""
    diff = p[:, None, :] - p[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    w = 1.6 * np.exp(-(dist - MIN_SEP) / 1.2)
    w[dist > 3.0] = 0.0
    w[:, arr] = 0.0
    safe = np.where(np.isfinite(dist), dist, 1.0)
    unit = diff / safe[..., None]
    return (w[..., None] * unit).sum(axis=1)


# ==========================================================================
def run(model="orca", n=N_DRONE, radius=RADIUS, vmax=V_MAX, dt=DT, tau=TAU,
        t_end=T_END, neighbor_dist=NEIGHBOR_DIST, max_neighbors=MAX_NEIGHBORS,
        record_every=1, verbose=True, margin=ORCA_MARGIN):
    # 起点：球面均匀分布；终点：关于球心对称的点
    p = uniform_sphere(n, radius)
    p0 = p.copy()
    goal = -p0.copy()
    v = np.zeros_like(p)
    uniform = nn_stats(p0, radius)

    arrived = np.zeros(n, dtype=bool)
    t_arrive = np.full(n, np.nan)
    path_len = np.zeros(n)

    # 对称性破缺（两条）：
    #   ① 侧向偏置——每架机选一个固定的绕行侧，避免全挤在球心那一条线上；
    #   ② 巡航速度差异——错开到达球心的时刻，避免同时争抢同一空间。
    # ORCA 在完全对称的对趾构型下会互相"礼让"到绕着球心打转（实测 20 架时
    # 全部到不了、最小间距 0.16 m），这两条是让场景可解的必要条件。
    rng = np.random.default_rng(7)
    goal_dir0 = goal - p0
    goal_dir0 /= np.maximum(np.linalg.norm(goal_dir0, axis=1, keepdims=True), 1e-9)
    bias = rng.normal(size=(n, 3))
    bias -= (bias * goal_dir0).sum(axis=1, keepdims=True) * goal_dir0
    bias /= np.maximum(np.linalg.norm(bias, axis=1, keepdims=True), 1e-9)
    BIAS_GAIN = 0.18
    speed_factor = 0.80 + 0.20 * rng.random(n)      # 0.80 ~ 1.00

    steps = int(round(t_end / dt))
    traj, arrived_hist = [], []
    collision_count_hist = []
    collision_event_hist = []
    collision_pairs_seen = set()
    collision_event_count = 0
    first_violation = None

    def pref_velocity():
        """期望速度：朝目标 + 侧向偏置，接近目标时按剩余距离限速。"""
        d = goal - p
        dist = np.linalg.norm(d, axis=1, keepdims=True)
        u = d / np.maximum(dist, 1e-9)
        out = vmax * speed_factor[:, None] * (u + BIAS_GAIN * bias)
        sp = np.linalg.norm(out, axis=1, keepdims=True)
        out = out / np.maximum(sp, 1e-9) * (vmax * speed_factor[:, None])
        return out * np.clip(dist / 2.0, 0.0, 1.0)

    for k in range(steps):
        pref = pref_velocity()
        pref[arrived] = 0.0

        if model == "direct":
            new_v = pref.copy()

        elif model == "repulsion":
            new_v = pref + _repulsion(p, v, arrived)
            sp = np.linalg.norm(new_v, axis=1)
            over = sp > vmax
            new_v[over] = new_v[over] / sp[over, None] * vmax

        elif model == "orca":
            new_v = np.zeros_like(p)
            diff = p[:, None, :] - p[None, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            for i in range(n):
                if arrived[i]:
                    continue
                order = np.argsort(dist[i])
                cand = [j for j in order[:max_neighbors]
                        if j != i and dist[i, j] < neighbor_dist]
                neigh = [(p[j], v[j], SAFE_RADIUS, bool(arrived[j]))
                         for j in cand]
                new_v[i] = solve_velocity(p[i], v[i], pref[i], neigh,
                                          SAFE_RADIUS, vmax, tau, dt, margin)
        elif model == "cbf":
            new_v = np.zeros_like(p)
            diff = p[:, None, :] - p[None, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            for i in range(n):
                if arrived[i]:
                    continue
                order = np.argsort(dist[i])
                cand = [j for j in order[:max_neighbors]
                        if j != i and dist[i, j] < neighbor_dist]
                neigh = [(p[j], v[j], SAFE_RADIUS, bool(arrived[j])) for j in cand]
                new_v[i] = cbf_qp_velocity(p[i], pref[i], neigh,
                                           MIN_SEP + 0.65, vmax, alpha=6.0)
        else:
            raise ValueError(model)

        new_v[arrived] = 0.0
        v = new_v

        path_len += np.linalg.norm(v, axis=1) * dt
        p = p + v * dt                       # 三维自由空间，无边界约束

        d_goal = np.linalg.norm(p - goal, axis=1)
        newly = (~arrived) & (d_goal <= ARRIVE_TOL)
        if newly.any():
            arrived[newly] = True
            t_arrive[newly] = (k + 1) * dt
            v[arrived] = 0.0

        dd = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        np.fill_diagonal(dd, np.inf)
        risk_mask = np.triu(dd < MIN_SEP, 1)
        risk_pairs = set(zip(*np.where(risk_mask)))
        new_risk_pairs = risk_pairs - collision_pairs_seen
        collision_event_count += len(new_risk_pairs)
        collision_pairs_seen.update(risk_pairs)
        if first_violation is None and risk_pairs:
            first_violation = (k + 1) * dt
        collision_count_hist.append(int(len(risk_pairs)))
        collision_event_hist.append(int(collision_event_count))

        if k % record_every == 0:
            traj.append(p.copy())
            arrived_hist.append(arrived.copy())

        if arrived.all():
            break

        if verbose and (k + 1) % 200 == 0:
            print(f"    t={dt*(k+1):5.1f}s  到达 {arrived.sum():3d}/{n}  "
                  f"最小间距 {dd.min():.3f} m", flush=True)

    traj.append(p.copy())
    arrived_hist.append(arrived.copy())
    traj = np.stack(traj)

    # ---- 指标 ----
    T = len(traj)
    min_sep_all = np.inf
    pair_min = np.full((n, n), np.inf)
    for k in range(T):
        dd = np.linalg.norm(traj[k][:, None, :] - traj[k][None, :, :], axis=2)
        np.fill_diagonal(dd, np.inf)
        min_sep_all = min(min_sep_all, dd.min())
        pair_min = np.minimum(pair_min, dd)

    iu = np.triu_indices(n, 1)
    violating = int((pair_min[iu] < MIN_SEP).sum())
    arrived_final = arrived.sum()
    t_all = float(np.nanmax(t_arrive)) if arrived_final == n else None

    metrics = dict(
        model=model, motion_space="3d_free",
        n=int(n), radius=float(radius), vmax=float(vmax), dt=float(dt),
        tau=float(tau), t_end=float(t_end), margin=float(margin),
        arrived=int(arrived_final),
        arrival_rate=float(arrived_final / n),
        min_separation=float(min_sep_all),
        violating_pairs=violating,
        collision_pair_count=violating,
        collision_event_count=int(collision_event_count),
        collision_samples=int(np.sum(collision_count_hist)),
        collision_frames=int(np.sum(np.asarray(collision_count_hist) > 0)),
        peak_collision_pairs=int(np.max(collision_count_hist) if collision_count_hist else 0),
        violating_pair_ratio=float(violating / len(iu[0])),
        t_all_arrive=t_all,
        mean_path=float(path_len.mean()),
        std_path=float(path_len.std()),
        theoretical_path=2.0 * radius,
        first_violation_time=first_violation,
        t_arrive=t_arrive,
        path_len=path_len,
        collision_count_hist=np.asarray(collision_count_hist, dtype=int),
        collision_event_hist=np.asarray(collision_event_hist, dtype=int),
        uniformity=uniform,
    )
    return dict(traj=traj, t=np.arange(T) * dt * record_every,
                p0=p0, goal=goal, metrics=metrics,
                arrived_hist=np.stack(arrived_hist))


# ==========================================================================
MODELS = ["cbf", "orca", "direct", "repulsion"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--n", type=int, default=N_DRONE)
    ap.add_argument("--radius", type=float, default=RADIUS)
    ap.add_argument("--t-end", type=float, default=T_END)
    ap.add_argument("--tag", default="")
    ap.add_argument("--margin", type=float, default=ORCA_MARGIN)
    ap.add_argument("--tau", type=float, default=TAU)
    args = ap.parse_args()

    rows = []
    for m in args.models:
        print(f"跑 {m} ({args.n} 架, R={args.radius:g}, T={args.t_end:g}s, "
              f"tau={args.tau:g}) ...", flush=True)
        out = run(m, n=args.n, radius=args.radius, t_end=args.t_end,
                  margin=args.margin, tau=args.tau)
        mt = out["metrics"]
        np.savez_compressed(
            RESULTS / f"traj_{m}{args.tag}.npz",
            traj=out["traj"], t=out["t"], p0=out["p0"], goal=out["goal"],
            arrived=out["arrived_hist"],
            colors=region_colors(out["p0"]),
            t_arrive=mt["t_arrive"], path_len=mt["path_len"],
            collision_count=mt["collision_count_hist"],
            collision_events=mt["collision_event_hist"])
        mt_save = {k: v for k, v in mt.items() if not isinstance(v, np.ndarray)}
        (RESULTS / f"metrics_{m}{args.tag}.json").write_text(
            json.dumps(mt_save, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(mt)
        print(f"    到达 {mt['arrived']}/{mt['n']}  "
              f"最小间距 {mt['min_separation']:.3f} m  "
              f"违规对 {mt['violating_pairs']}  "
              f"全部到达 {mt['t_all_arrive']}  "
              f"平均路程 {mt['mean_path']:.2f} m", flush=True)

    path = RESULTS / f"metrics_table{args.tag}.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "arrived", "n", "arrival_rate",
                    "min_separation_m", "violating_pairs",
                    "violating_pair_ratio", "t_all_arrive_s",
                    "mean_path_m", "std_path_m", "first_violation_s",
                    "collision_event_count", "collision_samples",
                    "collision_frames", "peak_collision_pairs"])
        for r in rows:
            w.writerow([r["model"], r["arrived"], r["n"],
                        f"{r['arrival_rate']:.3f}",
                        f"{r['min_separation']:.4f}",
                        r["violating_pairs"],
                        f"{r['violating_pair_ratio']:.4f}",
                        "" if r["t_all_arrive"] is None else f"{r['t_all_arrive']:.1f}",
                        f"{r['mean_path']:.2f}", f"{r['std_path']:.2f}",
                        "" if r["first_violation_time"] is None
                        else f"{r['first_violation_time']:.1f}",
                        r["collision_event_count"], r["collision_samples"],
                        r["collision_frames"], r["peak_collision_pairs"]])
    print("已写出", path)


if __name__ == "__main__":
    main()
