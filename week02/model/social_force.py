#!/usr/bin/env python3
"""week02 · 步骤② 行为模型 —— 圆环对趾场景的社会力模型。

本文件只实现课程要求的三项力，其余力项（墙壁、成组、视野各向异性、
随机扰动）先不加，留作后续扩展：

1. **终点的吸引力**（driving force / goal attraction）
   行人以特征时间 τ 把自己的速度向期望速度 v0·e_i 松弛，e_i 指向对趾点。
   这是把人"拉"向目标的那一项，也是唯一让行人前进的力。

2. **行人间的排斥力**（social / psychological repulsion）
   指数衰减的软排斥，A·exp((r_i + r_j − d_ij)/B)·n_ij。
   它在身体还没接触时就起作用，负责"提前避让"。

3. **接触力**（contact force）
   身体相互挤压时才出现的法向弹性力与切向摩擦：
   k·g(r_ij − d_ij)·n_ij + κ·g(r_ij − d_ij)·Δv_ji^t·t_ij，
   其中 g(x) = max(0, x)。它决定"挤在一起时会发生什么"。

控制方程
--------
    m_i dv_i/dt = m_i (v0_i e_i − v_i) / τ_i
                  + Σ_{j≠i} [ A exp((r_ij − d_ij)/B) + k g(r_ij − d_ij) ] n_ij
                  + Σ_{j≠i} κ g(r_ij − d_ij) (Δv_ji · t_ij) t_ij

参考
----
- Helbing & Molnár, *Social force model for pedestrian dynamics*,
  Phys. Rev. E 51(5):4282, 1995.  （排斥力 + 终点吸引）
- Helbing, Farkas & Vicsek, *Simulating dynamical features of escape panic*,
  Nature 407:487, 2000.  （接触力与摩擦项）

积分
----
接触力很硬（k = 1.2e5 kg/s²），用显式欧拉时步长必须远小于身体挤压的
振荡周期。身体的固有频率约 sqrt(k/m) ≈ 39 rad/s（周期 0.16 s），因此
内部步长默认取 0.002 s，再把结果按 25 fps 重采样，与实测数据对齐。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

# ------------------------------------------------------------------
# 参数
# ------------------------------------------------------------------

@dataclass
class SFMParams:
    """社会力模型参数。默认值取自 Helbing 的原始工作。"""

    # --- 终点吸引力 ---
    tau: float = 0.5          # 速度松弛时间 [s]
    v0: float = 2.0           # 期望速度 [m/s]，由实测自由流速标定

    # --- 行人间排斥力 ---
    A: float = 2000.0         # 排斥力强度 [N]
    B: float = 0.08           # 排斥力作用尺度 [m]
    rep_cutoff: float = 3.0   # 超过该距离不再计算排斥力（省算力）

    # --- 接触力 ---
    k: float = 1.2e5          # 法向弹性系数 [kg/s²]
    kappa: float = 2.4e5      # 切向摩擦系数 [kg/(m·s)]

    # --- 身体 ---
    mass: float = 80.0        # 体重 [kg]
    radius: float = 0.25      # 身体半径 [m]，两人相接触时中心距 0.5 m

    # --- 积分 ---
    dt: float = 0.002         # 内部步长 [s]

    # --- 失效分析用的开关（默认关闭）---
    # 只让来自前半平面的排斥力起作用，模拟"人看不到身后"。
    # 这不是课程要求的三项力之一，仅用于验证拥堵的成因，见 README。
    front_only: bool = False

    def scaled(self, **kw) -> "SFMParams":
        return replace(self, **kw)


# ------------------------------------------------------------------
# 参数预设
# ------------------------------------------------------------------

#: 按 ``tune.py`` 的筛选结果标定的一组参数。
#:
#: 选取规则：在"64 人全部到达"的配置里，取与实测偏差最小的一组。
#: 结果是排斥力被放大到文献值的 8 倍（A = 16000 N，作用在人身上约
#: 200 m/s²），作用尺度 B 保持文献值。代价见 README 的参数一节：
#: 拥堵确实被压掉了，但人流的横向散布反而比实测窄了三成。
TUNED = dict(v0=2.8, A=16000.0, B=0.08)

#: 期望速度的上界依据：实测行进段速率的 p90 为 2.85 m/s、p75 为 2.83 m/s，
#: 自由流（2 m 内无邻居）中位 2.02 m/s。文献值与实测值之间需要一次标定，
#: 见 README。

PRESETS: dict[str, callable] = {
    "literature": SFMParams,                       # Helbing 原始值
    "tuned": lambda: SFMParams(**TUNED),           # 本次标定
}


# ------------------------------------------------------------------
# 场景：圆环对趾
# ------------------------------------------------------------------

@dataclass
class Ring:
    """圆环对趾场景：行人在半径 R 的圆周上，目标为正对的对趾点。"""

    radius: float = 10.0

    def uniform(self, n: int, jitter: float = 0.0,
                seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """把 n 个行人均布在圆周上，返回 (位置, 目标点)。"""
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        if jitter > 0:
            rng = np.random.default_rng(seed)
            th = th + rng.normal(0, jitter, n)
        r = self.radius * np.ones(n)
        pos = np.column_stack([r * np.cos(th), r * np.sin(th)])
        return pos, -pos                      # 目标 = 对趾点


# ------------------------------------------------------------------
# 力
# ------------------------------------------------------------------

def _pairwise(pos: np.ndarray, vel: np.ndarray, p: SFMParams):
    """计算所有对子量，返回 (d, n_ij, t_ij, overlap, delta_v_t)。

    索引约定：``[i, j]`` 表示"作用在 i 上、由 j 产生"的量。
    """
    dx = pos[:, None, :] - pos[None, :, :]          # i − j
    d = np.linalg.norm(dx, axis=2)
    np.fill_diagonal(d, np.inf)                     # 排除自身
    safe = np.where(np.isfinite(d), d, 1.0)
    n = dx / safe[..., None]                        # n_ij：由 j 指向 i

    r_ij = 2.0 * p.radius
    overlap = np.maximum(0.0, r_ij - d)             # g(r_ij − d_ij)

    # 切向单位向量：n 逆时针转 90°
    t = np.stack([-n[..., 1], n[..., 0]], axis=-1)
    # Δv_ji · t_ij，其中 Δv_ji = v_j − v_i
    dv_t = ((vel[None, :, :] - vel[:, None, :]) * t).sum(-1)
    return d, n, t, overlap, dv_t


def accelerations(pos: np.ndarray, vel: np.ndarray, goal: np.ndarray,
                  p: SFMParams, v0: np.ndarray | None = None) -> np.ndarray:
    """返回 (A, 2) 的加速度。三项力合在一起。"""
    A = pos.shape[0]
    v0_arr = np.full(A, p.v0) if v0 is None else np.asarray(v0, dtype=float)

    # --- 1. 终点的吸引力：向以 v0 指向目标的期望速度松弛 ---
    e = goal - pos
    e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    drive = p.mass * (v0_arr[:, None] * e - vel) / p.tau

    # --- 2 & 3. 对子力：排斥力 + 接触力 ---
    d, n, t, overlap, dv_t = _pairwise(pos, vel, p)

    rep = p.A * np.exp((2.0 * p.radius - d) / p.B)
    rep = np.where(d < p.rep_cutoff, rep, 0.0)

    if p.front_only:
        # 只留下"来自前方"的排斥力：邻居方向 n_ij（由 j 指向 i）与自身
        # 行进方向同侧。用于验证拥堵成因，见 README 的失效分析。
        hv = np.linalg.norm(vel, axis=1, keepdims=True)
        eig = np.where(hv > 1e-6, vel / np.maximum(hv, 1e-9), e)
        rep = np.where((n * eig[:, None, :]).sum(-1) > 0, rep, 0.0)

    f_norm = (rep + p.k * overlap)[..., None] * n
    f_tang = (p.kappa * overlap * dv_t)[..., None] * t

    pair = (f_norm + f_tang).sum(axis=1)            # 对 j 求和

    return (drive + pair) / p.mass


# ------------------------------------------------------------------
# 积分
# ------------------------------------------------------------------

def simulate(pos0: np.ndarray, goal: np.ndarray, p: SFMParams,
             duration: float, v0: np.ndarray | None = None,
             sample_every: int = 1) -> np.ndarray:
    """显式欧拉积分，返回 (n_sample, A, 2) 的位置序列。

    ``duration`` 与 ``sample_every`` 共同决定采样步长：
    采样间隔 = dt × sample_every。
    """
    pos = np.asarray(pos0, dtype=float).copy()
    vel = np.zeros_like(pos)
    n_step = int(round(duration / p.dt))

    out = [pos.copy()]
    for s in range(n_step):
        acc = accelerations(pos, vel, goal, p, v0)
        vel = vel + acc * p.dt
        pos = pos + vel * p.dt
        if sample_every > 0 and (s + 1) % sample_every == 0:
            out.append(pos.copy())
    return np.stack(out, axis=0)


def to_trial(traj: np.ndarray, fps: float = 25.0, agents: list[int] | None = None):
    """把仿真轨迹包装成 ``circle_data.CircleTrial``，复用实测那套诊断量。"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from circle_data import CircleTrial

    n, A, _ = traj.shape
    ids = list(range(A)) if agents is None else list(agents)
    tracks = {ids[i]: traj[:, i, :] for i in range(A)}
    return CircleTrial(tracks=tracks, heading={i: 0.0 for i in ids},
                       frames=np.arange(n), fps=fps)


def min_pair_distance(traj: np.ndarray) -> np.ndarray:
    """(T,) 每个采样时刻全场最小的两人间距，单位米。"""
    out = np.empty(traj.shape[0])
    for k in range(traj.shape[0]):
        d = np.linalg.norm(traj[k][:, None, :] - traj[k][None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        out[k] = d.min()
    return out


def centrality(traj: np.ndarray) -> np.ndarray:
    """(T,) 每个时刻"离圆心 3 m 以内的人数占比"。"""
    return (np.linalg.norm(traj, axis=2) < 3.0).mean(axis=1)


def arrival_curve(traj: np.ndarray, goal: np.ndarray,
                  tol: float = 0.6) -> np.ndarray:
    """(T,) 每个时刻累计到达人数（到达后不会再离开目标点 tol 之内）。"""
    dd = np.linalg.norm(traj - goal[None], axis=2) < tol
    # 用"之后一直满足"来判定到达，避免路过被误计
    out = np.zeros(traj.shape[0], dtype=int)
    for i in range(traj.shape[1]):
        far = np.where(~dd[:, i])[0]
        out[(far.max() + 1) if far.size else 0:] += 1
    return np.maximum.accumulate(out)


def contact_stats(traj: np.ndarray, radius: float = 0.25) -> dict:
    """统计身体接触（两人中心距小于 2r）的发生情况。"""
    r_ij = 2.0 * radius
    n_pair = np.zeros(traj.shape[0], dtype=int)
    depth = []
    for k in range(traj.shape[0]):
        d = np.linalg.norm(traj[k][:, None, :] - traj[k][None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        n_pair[k] = int((d < r_ij).sum() // 2)
        if n_pair[k]:
            depth.append(r_ij - d[d < r_ij])
    depth = np.concatenate(depth) if depth else np.array([0.0])
    return {
        "pair_samples": int(n_pair.sum()),
        "touching_frames": float((n_pair > 0).mean()),
        "max_pairs": int(n_pair.max()),
        "max_overlap": float(depth.max()),
        "median_overlap": float(np.median(depth)),
    }


def measured_contacts(tr, radius: float = 0.25) -> dict:
    """实测轨迹中身体接触的发生情况（同一套判据，便于直接对比）。"""
    return contact_stats(tr.positions(), radius)


# ------------------------------------------------------------------
# 自检
# ------------------------------------------------------------------

def _demo() -> int:
    p = SFMParams()
    ring = Ring(10.0)
    pos0, goal = ring.uniform(64)
    traj = simulate(pos0, goal, p, duration=17.0, sample_every=20)
    print(f"参数: A={p.A:g} B={p.B:g} k={p.k:g} kappa={p.kappa:g} "
          f"tau={p.tau:g} v0={p.v0:g} m={p.mass:g}")
    print(f"轨迹形状 {traj.shape}  末位置半径 "
          f"{np.linalg.norm(traj[-1], axis=1).min():.2f} ~ "
          f"{np.linalg.norm(traj[-1], axis=1).max():.2f} m")
    dist = np.linalg.norm(traj[-1] - goal, axis=1)
    print(f"末位置离目标 {dist.min():.2f} ~ {dist.max():.2f} m   "
          f"中位 {np.median(dist):.2f} m")

    # 最小间距随时间
    mind = np.array([np.min(np.linalg.norm(
        t[:, None, :] - t[None, :, :], axis=2)
        + np.eye(len(t)) * 1e3) for t in traj])
    print(f"最小两人间距 全程最小 {mind.min():.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
