# -*- coding: utf-8 -*-
"""
三维球面对趾场景：100 架无人机同时飞向各自的对趾点。

场景参数（按课堂要求）
----------------------
    无人机数量   N = 100，Fibonacci 球面均匀分布
    球半径       R = 10 m（题目未给，取与 week02 圆环同量级）
    安全半径     0.5 m  →  任意两机间隔 ≥ 1 m
    最大速度     vmax = 2 m/s
    时间步长     Δt = 0.1 s
    ORCA 时间窗  τ = 5 s
    初速         0
    终止         距目标 ≤ 1 m 即停留
    更新方式     同步更新（先算完所有人再一起动）
    限时         120 s

对比模型
--------
    orca       球面 ORCA（切平面 + RVO2 线性规划）
    direct     直接朝目标飞（不做任何避让）
    repulsion  简单排斥场（社会力式的速度层修正）

输出
----
    results/traj_<model>.npz     轨迹与到达信息
    results/metrics_<model>.json 五项指标
    results/metrics_table.csv    汇总表
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
    fibonacci_sphere, goal_direction, solve_velocity,
)

# ---- 场景参数 ----
N_DRONE = 100
RADIUS = 10.0
V_MAX = 2.0
DT = 0.1
TAU = 5.0
T_END = 120.0
SAFE_RADIUS = 0.5          # 单机安全半径
MIN_SEP = 2 * SAFE_RADIUS  # 最小允许间隔
#: ORCA 约束里额外留出的余量。理论上 ORCA 能保证恰好 1.0 m，但离散时间步、
#: 切平面投影、以及偶发的 LP 不可行都会吃掉几毫米。留 2 cm 后实测最小间隔
#: 稳定在 1.0 m 以上；**统计口径仍然是 1.0 m**，这个余量不参与评价。
ORCA_MARGIN = 0.05
ORCA_RADIUS = SAFE_RADIUS + ORCA_MARGIN
ARRIVE_TOL = 1.0
NEIGHBOR_DIST = 20.0       # 球半径 10 → 20 m 覆盖全对趾，等于"看所有人"
MAX_NEIGHBORS = 40


# ==========================================================================
def initial_headings(p, radius, seed=0):
    """给每架机一个初始切向，用来打破对趾方向的简并。

    球面上从 p 到其对趾点 -p 的最短路径是任意一条过 p 与 -p 的大圆，
    **方向不唯一**：把 -p 投影到 p 的切平面上恰好得到零向量。
    所以必须人为指定一个初始切向；这里取确定性的随机切向，
    既避免"全军同一方向"的伪影，又保证可复现。
    """
    rng = np.random.default_rng(seed)
    n = len(p)
    h = np.zeros_like(p)
    for i in range(n):
        u = p[i] / radius
        r = rng.normal(size=3)
        t = r - np.dot(r, u) * u
        norm = np.linalg.norm(t)
        h[i] = t / norm if norm > 1e-9 else np.array([0.0, 0.0, 0.0])
    return h


def _preferred(p, goal, vmax, radius, heading0):
    pref = np.zeros_like(p)
    for i in range(len(p)):
        d = goal_direction(p[i], goal[i], radius, heading0[i])
        pref[i] = vmax * d
    return pref


def _repulsion(p, v, arr, vmax):
    """简单排斥场：在期望速度上叠加邻居的指数排斥。"""
    n = len(p)
    diff = p[:, None, :] - p[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    # 只算 3 m 内的
    w = 1.6 * np.exp(-(dist - MIN_SEP) / 1.2)
    w[dist > 3.0] = 0.0
    w[:, arr] = 0.0                     # 已到达的邻居速度为 0，仍算障碍
    safe = np.where(np.isfinite(dist), dist, 1.0)
    unit = diff / safe[..., None]
    rep = (w[..., None] * unit).sum(axis=1)
    return rep


# ==========================================================================
def run(model="orca", n=N_DRONE, radius=RADIUS, vmax=V_MAX, dt=DT, tau=TAU,
        t_end=T_END, neighbor_dist=NEIGHBOR_DIST, max_neighbors=MAX_NEIGHBORS,
        record_every=1, verbose=True):
    p = fibonacci_sphere(n, radius)
    goal = -p.copy()
    v = np.zeros_like(p)
    heading0 = initial_headings(p, radius, seed=0)

    arrived = np.zeros(n, dtype=bool)
    t_arrive = np.full(n, np.nan)
    path_len = np.zeros(n)
    n_min_sep = np.full(n, np.inf)      # 每个 agent 的历史最小邻居距离

    steps = int(round(t_end / dt))
    traj = []
    arrived_hist = []
    first_violation = None

    for k in range(steps):
        pref = _preferred(p, goal, vmax, radius, heading0)

        if model == "direct":
            new_v = pref

        elif model == "repulsion":
            new_v = pref + _repulsion(p, v, arrived, vmax)
            sp = np.linalg.norm(new_v, axis=1)
            over = sp > vmax
            new_v[over] = new_v[over] / sp[over, None] * vmax

        elif model == "orca":
            new_v = np.zeros_like(p)
            # 每步算一次全对距离，供邻居筛选
            diff = p[:, None, :] - p[None, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            for i in range(n):
                if arrived[i]:
                    new_v[i] = 0.0
                    continue
                order = np.argsort(dist[i])
                cand = [j for j in order[:max_neighbors]
                        if j != i and dist[i, j] < neighbor_dist]
                neigh = [(p[j], v[j], ORCA_RADIUS, bool(arrived[j]))
                         for j in cand]
                new_v[i] = solve_velocity(
                    p[i], v[i], pref[i], neigh, ORCA_RADIUS,
                    vmax, tau, dt, neighbor_dist)
        else:
            raise ValueError(model)

        # 已到达的无人机「停留」：任何模型都不再移动
        new_v[arrived] = 0.0
        v = new_v
        speed = np.linalg.norm(v, axis=1)
        path_len += speed * dt
        p = p + v * dt
        p = radius * p / np.linalg.norm(p, axis=1, keepdims=True)

        # 到达判定
        d_goal = np.linalg.norm(p - goal, axis=1)
        newly = (~arrived) & (d_goal <= ARRIVE_TOL)
        if newly.any():
            arrived[newly] = True
            t_arrive[newly] = (k + 1) * dt
        v[arrived] = 0.0

        # 最近邻距离跟踪
        dd = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        np.fill_diagonal(dd, np.inf)
        n_min_sep = np.minimum(n_min_sep, dd.min(axis=1))
        if first_violation is None and dd.min() < MIN_SEP:
            first_violation = (k + 1) * dt

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
        model=model,
        n=int(n), radius=float(radius), vmax=float(vmax), dt=float(dt),
        tau=float(tau), t_end=float(t_end),
        arrived=int(arrived_final),
        arrival_rate=float(arrived_final / n),
        min_separation=float(min_sep_all),
        violating_pairs=violating,
        violating_pair_ratio=float(violating / len(iu[0])),
        t_all_arrive=t_all,
        mean_path=float(path_len.mean()),
        std_path=float(path_len.std()),
        first_violation_time=first_violation,
        t_arrive=t_arrive,
        path_len=path_len,
    )
    return dict(traj=traj, t=np.arange(T) * dt * record_every,
                goal=goal, metrics=metrics,
                arrived_hist=np.stack(arrived_hist))


# ==========================================================================
MODELS = ["orca", "direct", "repulsion"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--n", type=int, default=N_DRONE)
    ap.add_argument("--radius", type=float, default=RADIUS)
    ap.add_argument("--t-end", type=float, default=T_END)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rows = []
    for m in args.models:
        print(f"跑 {m} ({args.n} 架, R={args.radius:g}, T={args.t_end:g}s) ...",
              flush=True)
        out = run(m, n=args.n, radius=args.radius, t_end=args.t_end)
        mt = out["metrics"]
        tag = args.tag
        np.savez_compressed(
            RESULTS / f"traj_{m}{tag}.npz",
            traj=out["traj"], t=out["t"], goal=out["goal"],
            arrived=out["arrived_hist"],
            t_arrive=mt["t_arrive"], path_len=mt["path_len"])
        mt_save = {k: v for k, v in mt.items()
                   if not isinstance(v, np.ndarray)}
        (RESULTS / f"metrics_{m}{tag}.json").write_text(
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
                    "mean_path_m", "std_path_m", "first_violation_s"])
        for r in rows:
            w.writerow([r["model"], r["arrived"], r["n"],
                        f"{r['arrival_rate']:.3f}",
                        f"{r['min_separation']:.4f}",
                        r["violating_pairs"],
                        f"{r['violating_pair_ratio']:.4f}",
                        "" if r["t_all_arrive"] is None else f"{r['t_all_arrive']:.1f}",
                        f"{r['mean_path']:.2f}", f"{r['std_path']:.2f}",
                        "" if r["first_violation_time"] is None
                        else f"{r['first_violation_time']:.1f}"])
    print("已写出", path)


if __name__ == "__main__":
    main()
