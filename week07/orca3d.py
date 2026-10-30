# -*- coding: utf-8 -*-
"""
球面约束下的 ORCA（Optimal Reciprocal Collision Avoidance）。

问题
----
100 架无人机被约束在一个半径 R 的球面上，各自飞向自己的**对趾点**。
ORCA 原本是平面上的速度障碍法，这里要把它搬到**球面切空间**上：

    1. 在无人机位置 p 处取切平面，基底 (e1, e2)，法向 n = p/|p|；
    2. 把邻居的相对位置、相对速度都投影到该切平面 → 一个标准 2D ORCA 问题；
    3. 解出的新速度再映射回三维（v = v1 e1 + v2 e2）；
    4. 沿速度积分后把位置重新归一化回球面（相当于沿测地线走）。

投影误差的量级：球面上相距 d 的两点，径向高度差约 d²/(2R)。
本场景 d ≤ 1 m、R = 10 m 时约 5 cm，远小于 1 m 的安全间隔，可以忽略。

ORCA 本体
---------
半平面构造与线性规划严格按 van den Berg et al. (2011) / RVO2 的实现：

    ORCA_{A|B} = { v | (v - (v_A + 0.5 u)) · n >= 0 }

其中 (n, u) 由相对位置、相对速度、联合半径与时间窗 tau 决定。
随后在「速度圆盘 |v| <= vmax」与所有半平面的交集上，求离期望速度最近的点。

参考
----
- J. van den Berg, S. J. Guy, M. Lin, D. Manocha,
  *Reciprocal n-Body Collision Avoidance*, ISRR 2011.
- RVO2 参考实现（Apache-2.0）中的 LinearProgram1/2/3。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-9


# ==========================================================================
# 基础几何
# ==========================================================================
def det(a, b):
    return a[0] * b[1] - a[1] * b[0]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def abs_sq(a):
    return a[0] * a[0] + a[1] * a[1]


def normalize2(a):
    n = np.hypot(a[0], a[1])
    if n < EPS:
        return np.array([0.0, 0.0])
    return a / n


def tangent_basis(p):
    """球面点 p 处的正交切基 (e1, e2) 与单位法向 n。"""
    n = p / np.linalg.norm(p)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2, n


@dataclass
class Line:
    point: np.ndarray       # 半平面边界上一点
    direction: np.ndarray   # 边界方向（单位向量）


# ==========================================================================
# 线性规划（RVO2 的三个函数）
# ==========================================================================
def linear_program1(lines, line_no, radius, opt_velocity, direction_opt):
    line = lines[line_no]
    dot_product = dot(line.point, line.direction)
    discriminant = dot_product * dot_product + radius * radius - abs_sq(line.point)
    if discriminant < 0.0:
        return False, None

    sqrt_disc = np.sqrt(discriminant)
    t_left = -dot_product - sqrt_disc
    t_right = -dot_product + sqrt_disc

    for i in range(line_no):
        denominator = det(line.direction, lines[i].direction)
        numerator = det(lines[i].direction, line.point - lines[i].point)

        if abs(denominator) <= EPS:
            if numerator < 0.0:
                return False, None
            continue

        t = numerator / denominator
        if denominator >= 0.0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)
        if t_left > t_right:
            return False, None

    if direction_opt:
        if dot(opt_velocity, line.direction) > 0.0:
            result = line.point + t_right * line.direction
        else:
            result = line.point + t_left * line.direction
    else:
        t = dot(line.direction, opt_velocity - line.point)
        if t < t_left:
            result = line.point + t_left * line.direction
        elif t > t_right:
            result = line.point + t_right * line.direction
        else:
            result = line.point + t * line.direction
    return True, result


def linear_program2(lines, radius, opt_velocity, direction_opt):
    """返回 (失败的行号, 结果)。失败行号 == len(lines) 表示全部满足。"""
    if direction_opt:
        result = opt_velocity * radius
    elif abs_sq(opt_velocity) > radius * radius:
        result = normalize2(opt_velocity) * radius
    else:
        result = opt_velocity.copy()

    for i in range(len(lines)):
        if det(lines[i].direction, lines[i].point - result) > 0.0:
            temp = result.copy()
            ok, new = linear_program1(lines, i, radius, opt_velocity, direction_opt)
            if not ok:
                return i, temp
            result = new
    return len(lines), result


def linear_program3(lines, num_obst_lines, begin_line, radius, result):
    """LP2 失败时的回退：逐条把不满足的约束投影进去再解。

    ⚠️ 必须从传入的 result 出发（RVO2 的做法）。若从零向量重新开始，
    会把前面已经满足的约束丢掉，解出来的速度可能仍然违反约束。
    """
    result = np.asarray(result, dtype=float).copy()
    distance = 0.0
    for i in range(begin_line, len(lines)):
        if det(lines[i].direction, lines[i].point - result) > distance:
            proj_lines = [Line(lines[k].point, lines[k].direction)
                          for k in range(num_obst_lines)]
            for j in range(num_obst_lines, i):
                determinant = det(lines[i].direction, lines[j].direction)
                if abs(determinant) <= EPS:
                    if dot(lines[i].direction, lines[j].direction) > 0.0:
                        continue
                    point = 0.5 * (lines[i].point + lines[j].point)
                else:
                    point = (lines[i].point
                             + (det(lines[j].direction,
                                    lines[i].point - lines[j].point)
                                / determinant) * lines[i].direction)
                direction = normalize2(lines[j].direction - lines[i].direction)
                proj_lines.append(Line(point, direction))

            temp = result.copy()
            fail, new = linear_program2(
                proj_lines, radius,
                np.array([-lines[i].direction[1], lines[i].direction[0]]), True)
            result = temp if fail < len(proj_lines) else new
            distance = det(lines[i].direction, lines[i].point - result)
    return result


# ==========================================================================
# ORCA 半平面
# ==========================================================================
def orca_line(rel_pos, rel_vel, combined_radius, tau, dt, v_self,
              share=0.5):
    """给定相对位置/相对速度，构造一条 ORCA 半平面。

    rel_pos : p_B - p_A（切平面 2D）
    rel_vel : v_A - v_B（切平面 2D）
    share   : 本机承担的避让比例。互惠时 0.5；对方是**静止障碍**
              （已到达停住的无人机）时必须取 1.0，否则双方都假设对方会让，
              实际没人让，就会出现穿透。
    """
    dist_sq = abs_sq(rel_pos)
    r_sq = combined_radius * combined_radius
    inv_tau = 1.0 / tau

    if dist_sq > r_sq:
        # 当前未碰撞：看相对速度是否落在截断锥/圆柱内
        w = rel_vel - inv_tau * rel_pos
        w_len_sq = abs_sq(w)
        dot1 = dot(w, rel_pos)

        if dot1 < 0.0 and dot1 * dot1 > r_sq * w_len_sq:
            # 投影到截断圆
            w_len = np.sqrt(w_len_sq)
            if w_len < EPS:
                return None
            unit_w = w / w_len
            direction = np.array([unit_w[1], -unit_w[0]])
            u = (combined_radius * inv_tau - w_len) * unit_w
        else:
            # 投影到两条切线腿
            leg = np.sqrt(max(dist_sq - r_sq, 0.0))
            if det(rel_pos, w) > 0.0:
                direction = np.array([
                    rel_pos[0] * leg - rel_pos[1] * combined_radius,
                    rel_pos[0] * combined_radius + rel_pos[1] * leg,
                ]) / dist_sq
            else:
                direction = -np.array([
                    rel_pos[0] * leg + rel_pos[1] * combined_radius,
                    -rel_pos[0] * combined_radius + rel_pos[1] * leg,
                ]) / dist_sq
            dot2 = dot(rel_vel, direction)
            u = dot2 * direction - rel_vel
    else:
        # 已经重叠：用 1/dt 的截断圆把速度直接推开
        inv_dt = 1.0 / dt
        w = rel_vel - inv_dt * rel_pos
        w_len = np.linalg.norm(w)
        if w_len < EPS:
            return None
        unit_w = w / w_len
        direction = np.array([unit_w[1], -unit_w[0]])
        u = (combined_radius * inv_dt - w_len) * unit_w

    point = v_self + share * u
    return Line(point, direction)


# ==========================================================================
# 单步求解
# ==========================================================================
def could_collide(r, w, combined_radius, tau, margin=0.10):
    """在 [0, tau] 内是否存在 |r + w t| < R 的时刻？

    r = p_j - p_i, w = v_i - v_j。用来**预筛邻居**：
    只有真正可能撞上的邻居才需要生成 ORCA 半平面，否则线性规划的规模
    会退化到 O(N^2)（100 架机时每步就是上万条约束）。
    """
    r_sq = abs_sq(r)
    r_lim = combined_radius + margin
    if r_sq < r_lim * r_lim:
        return True
    ww = dot(w, w)
    if ww < 1e-12:
        return r_sq < r_lim * r_lim
    t = -dot(r, w) / ww
    t = min(max(t, 0.0), tau)
    closest = r + t * w
    return abs_sq(closest) < r_lim * r_lim


def solve_velocity(pos_i, vel_i, pref_vel, neighbors, radius_self,
                   vmax, tau, dt, neighbor_dist=20.0):
    """在球面切平面上解一次 ORCA。

    neighbors : [(pos_j (3,), vel_j (3,), radius_j), ...]
    返回新的三维速度。
    """
    e1, e2, _n = tangent_basis(pos_i)
    vel2 = np.array([vel_i @ e1, vel_i @ e2])

    lines = []
    for neighbor in neighbors:
        pos_j, vel_j, r_j = neighbor[0], neighbor[1], neighbor[2]
        is_static = bool(neighbor[3]) if len(neighbor) > 3 else False
        rel = pos_j - pos_i
        # 只保留切平面分量（径向分量对近距离影响是二阶小量）
        rel2 = np.array([rel @ e1, rel @ e2])
        if np.linalg.norm(rel2) > neighbor_dist:
            continue
        dv = vel_i - vel_j
        dv2 = np.array([dv @ e1, dv @ e2])

        # 预筛：τ 内不可能相撞的邻居不生成约束
        if not could_collide(rel2, -dv2, radius_self + r_j, tau):
            continue

        line = orca_line(rel2, dv2, radius_self + r_j, tau, dt, vel2,
                         share=1.0 if is_static else 0.5)
        if line is not None:
            lines.append(line)

    pref2 = np.array([pref_vel @ e1, pref_vel @ e2])

    if not lines:
        new2 = pref2
    else:
        fail, new2 = linear_program2(lines, vmax, pref2, False)
        if fail < len(lines):
            new2 = linear_program3(lines, 0, fail, vmax, new2)

    # 限速
    n = np.linalg.norm(new2)
    if n > vmax:
        new2 = new2 / n * vmax

    return new2[0] * e1 + new2[1] * e2


# ==========================================================================
# 球面工具
# ==========================================================================
def fibonacci_sphere(n, radius=1.0):
    """Fibonacci 球面均匀分布（确定性、无随机种子）。"""
    i = np.arange(n, dtype=float)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - 2.0 * i / max(n - 1, 1)
    r = np.sqrt(np.maximum(1.0 - y * y, 0.0))
    theta = phi * i
    pts = np.column_stack([np.cos(theta) * r, y, np.sin(theta) * r])
    return pts * radius


def relax_uniform(p, radius, iters=800, step_m=0.04):
    """球面上的排斥松弛，把点分布压得更均匀（Thomson 型）。

    Fibonacci 只是**准**均匀：有螺旋纹，最近邻距离的离散度约 15%。
    这里对每对点施加 ~1/d^2 的排斥、投影到切平面、再归一化回球面，
    反复迭代直到最近邻距离的变异系数收敛。

    这一步直接决定"冲突有多集中"——分布越均匀，对撞带越分散。
    """
    p = np.asarray(p, dtype=float).copy()
    n = len(p)
    if n < 3:
        return p

    for it in range(iters):
        diff = p[:, None, :] - p[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        # 1/d^2 排斥；截断远处，避免全对互相拖拽
        w = 1.0 / np.maximum(d, 1e-3) ** 2
        w[d > 4.0 * radius] = 0.0
        np.fill_diagonal(w, 0.0)
        unit = diff / np.where(np.isfinite(d), d, 1.0)[..., None]
        disp = (w[..., None] * unit).sum(axis=1)

        # 投影到切平面
        nrm = p / radius
        disp = disp - (disp * nrm).sum(axis=1, keepdims=True) * nrm
        mag = np.linalg.norm(disp, axis=1)
        mx = mag.max()
        if mx < 1e-12:
            break
        # 每步最大位移固定为 step_m，收敛后自动变小
        scale = np.minimum(step_m / mx, 1.0)
        p = p + disp * scale
        p = radius * p / np.linalg.norm(p, axis=1, keepdims=True)

    return p


def uniform_sphere(n, radius=1.0, iters=800):
    """先 Fibonacci 播种，再排斥松弛 → 均匀分布。"""
    return relax_uniform(fibonacci_sphere(n, radius), radius, iters=iters)


def nn_stats(p, radius):
    """最近邻距离统计，用来量化"有多均匀"。"""
    diff = p[:, None, :] - p[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    area = 4.0 * np.pi * radius * radius
    d_ideal = np.sqrt(area / (len(p) * np.sqrt(3.0) / 2.0))   # 六边形密排
    return dict(
        nn_mean=float(nn.mean()), nn_std=float(nn.std()),
        nn_cv=float(nn.std() / nn.mean()),
        nn_min=float(nn.min()), nn_max=float(nn.max()),
        nn_max_over_min=float(nn.max() / nn.min()),
        d_ideal_hex=float(d_ideal),
        nn_mean_over_ideal=float(nn.mean() / d_ideal),
        radius_error=float(np.abs(np.linalg.norm(p, axis=1) - radius).max()),
    )


def geodesic_distance(p, q, radius):
    """球面上两点的大圆距离。"""
    c = np.clip(np.dot(p, q) / (radius * radius), -1.0, 1.0)
    return radius * np.arccos(c)


def goal_direction(p, goal, radius, fallback=None):
    """从 p 指向 goal 的大圆切向单位向量。

    对趾情形下 goal 在 p 的切平面上投影为零（方向不唯一），
    此时返回 fallback（由调用方给定的初始切向）。
    """
    n = p / radius
    t = goal - np.dot(goal, n) * n
    norm = np.linalg.norm(t)
    if norm < 1e-7:
        if fallback is None:
            raise ValueError("对趾点方向简并，需要给 fallback 切向")
        return np.asarray(fallback, dtype=float)
    return t / norm
