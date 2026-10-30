# -*- coding: utf-8 -*-
"""
三维 ORCA（Optimal Reciprocal Collision Avoidance）与球面均匀采样。

场景
----
100 架无人机均匀分布在半径 R 的球面上（起点），各自飞向**关于球心对称的
对趾点**（终点）。无人机在**三维自由空间**里直线穿球心飞行，不贴球面。

ORCA 的三维版本
---------------
2D 的 ORCA 半平面在 3D 里变成**半空间**：

    ORCA_{A|B} = { v | n · (v - (v_A + share·u)) >= 0 }

其中 u 是从当前相对速度指向速度障碍边界最近点的向量，n = u/|u|。

关键技巧：p 与 u 张成一个平面，而速度锥绕 p 轴对称。于是三维问题可以
**精确降维到该平面内的二维问题**，套用 RVO2 的二维公式，再把结果抬回三维：

    a = p/|p|                  (径向)
    e = normalize(u - (u·a)a)  (平面内的切向)
    p2 = (|p|, 0), u2 = (u·a, u·e)
    -> 解得 du2
    -> du3 = du2[0]·a + du2[1]·e

这不是近似：p、u 共面，锥面在该平面内就是两条母线，最近点必然落在该平面内。

速度求解
--------
3D 下可行域是「若干半空间 ∩ 速度球」，是凸集。求离期望速度最近的点用
**有效集法**（等式约束投影 + 逐步加入最违反的约束）：

    v = v_pref - N_a^T (N_a N_a^T)^{-1} (N_a v_pref - b_a)

2D 版本用的是 RVO2 的线性规划；3D 没有现成 LP，这里改用 QP 的有效集法，
约束数不大时得到的就是精确投影。
"""

from __future__ import annotations

import numpy as np

EPS = 1e-9


# ==========================================================================
# 1) 球面均匀采样
# ==========================================================================
def fibonacci_sphere(n, radius=1.0):
    """Fibonacci 球面分布（确定性）。只作为松弛初值。"""
    i = np.arange(n, dtype=float)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - 2.0 * i / max(n - 1, 1)
    r = np.sqrt(np.maximum(1.0 - y * y, 0.0))
    theta = phi * i
    pts = np.column_stack([np.cos(theta) * r, y, np.sin(theta) * r])
    return pts * radius


def relax_uniform(p, radius, iters=800, step_m=0.04):
    """球面排斥松弛（Thomson 型），把点分布压均匀。"""
    p = np.asarray(p, dtype=float).copy()
    n = len(p)
    if n < 3:
        return p
    for _ in range(iters):
        diff = p[:, None, :] - p[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        w = 1.0 / np.maximum(d, 1e-3) ** 2
        w[d > 4.0 * radius] = 0.0
        np.fill_diagonal(w, 0.0)
        unit = diff / np.where(np.isfinite(d), d, 1.0)[..., None]
        disp = (w[..., None] * unit).sum(axis=1)

        nrm = p / radius
        disp = disp - (disp * nrm).sum(axis=1, keepdims=True) * nrm
        mag = np.linalg.norm(disp, axis=1)
        mx = mag.max()
        if mx < 1e-12:
            break
        p = p + disp * min(step_m / mx, 1.0)
        p = radius * p / np.linalg.norm(p, axis=1, keepdims=True)
    return p


def uniform_sphere(n, radius=1.0, iters=800):
    return relax_uniform(fibonacci_sphere(n, radius), radius, iters=iters)


def nn_stats(p, radius):
    """最近邻距离统计，量化均匀程度。"""
    diff = p[:, None, :] - p[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    area = 4.0 * np.pi * radius * radius
    d_ideal = np.sqrt(area / (len(p) * np.sqrt(3.0) / 2.0))
    return dict(
        nn_mean=float(nn.mean()), nn_std=float(nn.std()),
        nn_cv=float(nn.std() / nn.mean()),
        nn_min=float(nn.min()), nn_max=float(nn.max()),
        nn_max_over_min=float(nn.max() / nn.min()),
        d_ideal_hex=float(d_ideal),
        nn_mean_over_ideal=float(nn.mean() / d_ideal),
        radius_error=float(np.abs(np.linalg.norm(p, axis=1) - radius).max()),
    )


# ==========================================================================
# 2) ORCA 半空间（三维）
# ==========================================================================
def _perp_basis(a):
    """给定单位向量 a，返回与它正交的单位向量 e。"""
    ref = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e = np.cross(a, ref)
    n = np.linalg.norm(e)
    if n < 1e-9:
        e = np.cross(a, np.array([0.0, 0.0, 1.0]))
        n = np.linalg.norm(e)
    return e / max(n, 1e-12)


def orca_halfspace(rel_pos, rel_vel, combined_radius, tau, dt, v_self,
                   share=0.5):
    """构造一条三维 ORCA 半空间 {v : n·(v - point) >= 0}。

    rel_pos : p_B - p_A
    rel_vel : v_A - v_B
    share   : 本机承担的避让比例（静止邻居取 1.0）
    """
    p = np.asarray(rel_pos, dtype=float)
    u = np.asarray(rel_vel, dtype=float)
    R = float(combined_radius)
    dist = np.linalg.norm(p)
    if dist < EPS:
        return None

    if dist <= R:
        # 已经进入安全半径时仍只使用 ORCA 速度规划。
        # 不做位置修正、反弹或其它实体碰撞响应；下一步位置只由规划速度积分得到。
        # 使用 tau 时间窗而不是 1/dt 的硬推开速度，便于如实统计风险/碰撞结果。
        w = u - p / max(float(tau), EPS)
        wl = np.linalg.norm(w)
        if wl < EPS:
            return None
        du = (R / max(float(tau), EPS) - wl) * (w / wl)
    else:
        a = p / dist                       # 径向单位向量
        s = float(u @ a)
        perp = u - s * a
        rho = float(np.linalg.norm(perp))
        e = perp / rho if rho > 1e-9 else _perp_basis(a)

        p2 = np.array([dist, 0.0])
        u2 = np.array([s, rho])

        # 降维到 (a, e) 平面后完全沿用 RVO2 的二维分支
        w2 = u2 - p2 / tau
        w_len_sq = float(w2 @ w2)
        dot1 = float(w2 @ p2)

        if dot1 < 0.0 and dot1 * dot1 > R * R * w_len_sq:
            # 最近点在「截断球冠」上
            w_len = np.sqrt(w_len_sq)
            if w_len < EPS:
                return None
            n2_dir = w2 / w_len
            du2 = (R / tau - w_len) * n2_dir
        else:
            # 最近点在锥面（两条母线）上
            leg = np.sqrt(max(dist * dist - R * R, 0.0))
            if dist * w2[1] > 0.0:          # det(p2, w2) = dist * w2[1]
                d2 = np.array([leg / dist, R / dist])
            else:
                d2 = np.array([-leg / dist, R / dist])
            u2_dot = float(u2 @ d2)
            du2 = u2_dot * d2 - u2
            # 锥面外法向（沿角度增大的方向）
            n2_dir = np.array([-d2[1], d2[0]])

        if float(np.linalg.norm(du2)) < 1e-9:
            # 当前相对速度恰好落在速度障碍边界上。此时 du≈0，
            # 若把约束丢掉，下一帧约束消失、速度直接跳回期望值，
            # 就会出现"安全帧 / 危险帧"交替的振荡（实测）。
            # 用切平面外法向保留约束：禁止继续向锥内转。
            n3 = n2_dir[0] * a + n2_dir[1] * e
            n3_len = np.linalg.norm(n3)
            if n3_len < EPS:
                return None
            return n3 / n3_len, np.asarray(v_self, dtype=float)

        du = du2[0] * a + du2[1] * e

    n_len = np.linalg.norm(du)
    if n_len < EPS:
        return None                        # 约束在当前速度下不活跃
    n = du / n_len
    point = np.asarray(v_self, dtype=float) + share * du
    return n, point


# ==========================================================================
# 3) 速度求解：半空间 ∩ 速度球 上的最近点（有效集法）
# ==========================================================================
def project_velocity(v_pref, normals, points, vmax, max_iter=60):
    """求 argmin |v - v_pref| s.t. n_i·(v - p_i) >= 0, |v| <= vmax。

    可行域是「半空间交 ∩ 速度球」。之前用纯有效集法（等式约束投影）在
    约束多、速度球同时起作用时会来回震荡、不收敛（实测 20 架时全部原地打转）。
    这里改成**交替投影**（POCS）：

        1. 反复把 v 投影到最违反的那个半空间边界上；
        2. 再把 v 拉回速度球内；
        3. 两步交替，直到所有约束满足或达到迭代上限。

    两个集合都是凸的，交替投影必收敛到交集内的一点；
    代价是不保证是"离期望速度最近"的那一点，但可行性优先。
    """
    v = np.asarray(v_pref, dtype=float).copy()
    nv = np.linalg.norm(v)
    if nv > vmax:
        v = v * vmax / nv

    if not normals:
        return v

    N = np.asarray(normals, dtype=float)
    b = np.array([float(np.dot(n, p)) for n, p in zip(normals, points)])

    best_v = v.copy()
    best_viol = float(np.max(b - N @ v))

    for _ in range(max_iter):
        # (1) 投影半空间：每次处理最违反的一条，内层多扫几次
        for _ in range(10):
            gaps = b - N @ v               # >0 表示违反
            i = int(np.argmax(gaps))
            if gaps[i] <= 1e-9:
                break
            v = v + gaps[i] * N[i]

        # (2) 拉回速度球
        nv = np.linalg.norm(v)
        if nv > vmax:
            v = v * vmax / nv

        viol = float(np.max(b - N @ v))
        if viol < best_viol:
            best_viol = viol
            best_v = v.copy()

        # (3) 收敛判定（球内且无违反）
        if viol <= 1e-9:
            break

    # 可行域为空时（密集拥堵），返回“违反最小”的那个解。
    # 若直接返回最后一次迭代，可能出现速度突然反向、整群原地打转。
    return best_v


def could_collide(rel_pos, rel_vel, combined_radius, tau, margin=0.10):
    """[0, tau] 内是否存在距离 < R 的时刻？

    rel_pos = p_B - p_A，rel_vel = v_A - v_B，故 B 相对 A 的位置是 p - t·u。
    最近时刻 t* = (p·u)/|u|²，夹到 [0, tau] 后判距离。

    没有这一层时，12 m 内的邻居都会生成半空间，100 架机时约束互相冲突、
    可行域变空，实测所有无人机原地打转（0/100 到达）。
    """
    p = np.asarray(rel_pos, dtype=float)
    u = np.asarray(rel_vel, dtype=float)
    r_lim = combined_radius + margin
    pp = float(p @ p)
    if pp < r_lim * r_lim:
        return True
    ww = float(u @ u)
    if ww < 1e-12:
        return False
    t = float(p @ u) / ww
    t = min(max(t, 0.0), tau)
    closest = p - t * u
    return float(closest @ closest) < r_lim * r_lim


def solve_velocity(pos_i, vel_i, pref_vel, neighbors, radius_self,
                   vmax, tau, dt, margin=0.0, prefilter=True):
    """解一次三维 ORCA。

    neighbors : [(pos_j (3,), vel_j (3,), radius_j, is_static), ...]

    prefilter=False 时对邻居范围内**所有**主体生成半空间。这是 RVO2 的做法，
    也更稳：按「当前速度下 τ 内是否会撞」筛邻居，会在"刚躲开"的那一步把约束
    丢掉，下一帧又恢复直冲，导致速度在安全/危险两态间振荡（实测过）。
    """
    normals, points = [], []
    for nb in neighbors:
        pos_j, vel_j, r_j = nb[0], nb[1], nb[2]
        is_static = bool(nb[3]) if len(nb) > 3 else False
        rel = pos_j - pos_i
        dv = vel_i - vel_j
        R = radius_self + r_j + margin

        if prefilter:
            # 近处的邻居无论如何都保留约束，避免"躲开就丢约束"的振荡
            if np.linalg.norm(rel) > 3.0 * R and not could_collide(rel, dv, R, tau):
                continue

        hs = orca_halfspace(rel, dv, R, tau, dt, vel_i,
                            share=1.0 if is_static else 0.5)
        if hs is None:
            continue
        normals.append(hs[0])
        points.append(hs[1])

    return project_velocity(pref_vel, normals, points, vmax)


# ==========================================================================
# 4) 颜色：固定不变、按空间区域渐变
# ==========================================================================
def region_colors(p0):
    """按初始方位角给每架机一个固定颜色（色相沿球面平滑渐变）。

    色相 = 方位角 0..2π 映射到 0..1（首尾相接）；亮度再叠一点纬度信息，
    让上下半球也有层次。颜色在整段仿真与动画里保持不变。
    """
    import colorsys
    th = np.arctan2(p0[:, 1], p0[:, 0])          # -π..π
    h = (th + np.pi) / (2.0 * np.pi)             # 0..1
    lat = np.arcsin(np.clip(p0[:, 2] / np.linalg.norm(p0, axis=1), -1, 1))
    v = 0.72 + 0.28 * (0.5 + 0.5 * np.sin(lat))
    return np.array([colorsys.hsv_to_rgb(hh, 0.85, vv) for hh, vv in zip(h, v)])
