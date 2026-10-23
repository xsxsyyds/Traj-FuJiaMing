# -*- coding: utf-8 -*-
"""统一社会力与显式让行规则：固定参数的教学实验。

两类主体共用 a = (v_des-v)/tau + f_social/m + f_wall/m。
社会排斥为 A*w*exp((r_i+r_j-d)/B)，附接触弹性与切向摩擦；
w 由前向各向异性与以米为单位的横向高斯衰减组成。
车辆用三个内接圆盘近似计算作用力；这些圆盘不覆盖矩形角部，
因此最终碰撞检查必须使用 evaluate.py 中的圆—矩形和矩形 SAT。

直廊：车辆只沿固定纵轴加减速；行人有横向偏好与墙力。
开放区域：直接朝对趾目标运动，并按相对运动的最近接近预测向右避让。
车辆禁止横向平移，横向力转换为转向，最小转弯半径 5.2 m、
最大角速度 0.6 rad/s；这是简化运动学约束，不是完整轮胎动力学。
前方投影重叠时按纵向净空缩放前进速度，横向运动不被停止规则抹掉。

所有参数为未标定的教学设定；本实现不保证无碰撞或无死锁。
模型路线参考 Anvari et al. (2015)，并非该论文的逐式复现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# ==========================================================================
# 0) 参数
# ==========================================================================

#: 主体类型参数。单位：m, s, kg。
PARAMS: dict[str, dict] = {
    "ped": dict(
        mass=70.0,
        v0=1.30,          # 期望速度 [m/s]
        tau=0.50,         # 速度松弛时间 [s]
        a_max=3.0,        # 加速度上限 [m/s^2]
        b_max=5.0,        # 减速度上限 [m/s^2]
        radius=0.25,      # 圆半径 [m]
        length=None, width=None,
        lam=0.30,         # 视野各向异性系数（0 = 只看前方，1 = 全向）
        t_safe=1.0,       # 让行判据的时间余量 [s]
        look=14.0,        # 前向观察距离 [m]
        d_slow=2.0,       # 开始减速的净空 [m]
        d_stop=0.15,      # 停住的净空 [m]
        margin=0.10,      # 横向重叠判据的余量 [m]
    ),
    "car": dict(
        mass=1500.0,
        v0=7.00,
        tau=2.50,
        a_max=2.0,
        b_max=6.0,
        radius=None,
        length=4.50,
        width=1.80,
        lam=0.05,         # 车几乎只对前方有反应
        t_safe=2.0,
        look=45.0,
        d_slow=16.0,
        d_stop=3.5,
        margin=0.10,
    ),
}

#: 对子作用强度 A [N]（行 ← 列）。
#:
#: 这里刻意做得**比 Helbing 原始值温和**：本模型里"纵向减速/停车"交给
#: 显式的让行规则（见 yield_speed_scale），社会力只负责**横向避让**。
#: 否则车会在还有 9 m 时就被径向排斥力硬刹住，出现"明明能过却停下"的假死锁。
A_MAT = {
    ("ped", "ped"): 2.0e3,
    ("ped", "car"): 2.0e3,
    ("car", "ped"): 6.0e3,
    ("car", "car"): 6.0e4,
}

#: 对子作用尺度 B [m]。
B_MAT = {
    ("ped", "ped"): 0.35,
    ("ped", "car"): 1.80,
    ("car", "ped"): 2.00,
    ("car", "car"): 3.00,
}

#: 接触力参数（Helbing 2000）。
K_CONTACT = 1.2e5      # 法向弹性 [kg/s^2]
KAPPA_FRIC = 2.4e5     # 切向摩擦 [kg/(m·s)]

#: 边界排斥
A_WALL = 70.0
B_WALL = 0.10
K_WALL = 1.2e5

DT = 0.01              # 积分步长 [s]；有限步长检查不等于连续时间安全证明
SAMPLE_DT = 0.05       # 输出采样步长 [s]


# ==========================================================================
# 1) 碰撞几何
# ==========================================================================
def rect_corners(center, heading, length, width):
    """矩形四角（世界坐标），顺序为 左前/右前/右后/左后。"""
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]])
    hx, hy = length / 2.0, width / 2.0
    local = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]])
    return local @ R.T + np.asarray(center)


def circle_rect_distance(center_c, radius, rect_center, heading, length, width):
    """圆心到矩形的最近距离再减去半径 <= 0 即碰撞。返回 (距离, 最近点)。"""
    c, s = np.cos(heading), np.sin(heading)
    d = np.asarray(center_c) - np.asarray(rect_center)
    # 转到矩形局部坐标系
    lx = c * d[0] + s * d[1]
    ly = -s * d[0] + c * d[1]
    hx, hy = length / 2.0, width / 2.0
    qx, qy = np.clip(lx, -hx, hx), np.clip(ly, -hy, hy)
    dist = np.hypot(lx - qx, ly - qy)
    # 最近点转回世界坐标
    wx = rect_center[0] + c * qx - s * qy
    wy = rect_center[1] + s * qx + c * qy
    return float(dist - radius), np.array([wx, wy])


def _project(points, axis):
    v = points @ axis
    return v.min(), v.max()


def rect_rect_overlap(c1, h1, l1, w1, c2, h2, l2, w2):
    """OBB vs OBB，分离轴定理。返回 (是否相交, 最小穿透深度)。"""
    p1 = rect_corners(c1, h1, l1, w1)
    p2 = rect_corners(c2, h2, l2, w2)
    axes = []
    for h in (h1, h2):
        axes.append(np.array([np.cos(h), np.sin(h)]))
        axes.append(np.array([-np.sin(h), np.cos(h)]))
    depth = np.inf
    for ax in axes:
        a0, a1 = _project(p1, ax)
        b0, b1 = _project(p2, ax)
        if a1 < b0 or b1 < a0:
            return False, 0.0
        depth = min(depth, min(a1 - b0, b1 - a0))
    return True, float(depth)


def pair_collision(a: "Agent", b: "Agent"):
    """两种主体的精确碰撞判据。返回 (是否碰撞, 穿透深度)。"""
    if a.kind == "ped" and b.kind == "ped":
        d = float(np.linalg.norm(a.pos - b.pos))
        return d < a.radius + b.radius, max(0.0, a.radius + b.radius - d)
    if a.kind == "ped" and b.kind == "car":
        gap, _ = circle_rect_distance(a.pos, a.radius, b.pos, b.heading,
                                      b.length, b.width)
        return gap < 0.0, max(0.0, -gap)
    if a.kind == "car" and b.kind == "ped":
        gap, _ = circle_rect_distance(b.pos, b.radius, a.pos, a.heading,
                                      a.length, a.width)
        return gap < 0.0, max(0.0, -gap)
    ok, dep = rect_rect_overlap(a.pos, a.heading, a.length, a.width,
                                b.pos, b.heading, b.length, b.width)
    return ok, dep


# ==========================================================================
# 2) 道路
# ==========================================================================
class Road:
    """道路抽象：提供横向边界、期望速度方向与边界力。

    横向坐标 lat 以道路中心线为 0，范围 [-half_width, +half_width]。
    """

    kind = "base"

    def __init__(self, width: float):
        self.width = float(width)
        self.half_width = float(width) / 2.0

    # --- 子类实现 ---
    def tangent(self, pos) -> np.ndarray:
        raise NotImplementedError

    def normal(self, pos) -> np.ndarray:
        raise NotImplementedError

    def lat_of(self, pos) -> float:
        raise NotImplementedError

    def to_world(self, s, lat):
        raise NotImplementedError

    def wall_gap(self, pos, radius):
        """返回 (最小净空, 指向路内的单位向量)。"""
        raise NotImplementedError

    # --- 通用 ---
    def desired_velocity(self, pos, sigma, v0, lat_pref, k_lat=0.9):
        """沿切线前进 + 朝偏好横向位置回正。"""
        t = self.tangent(pos) * sigma
        n = self.normal(pos)
        lat = self.lat_of(pos)
        v_lat = k_lat * (lat_pref - lat)
        v = v0 * t + v_lat * n
        # 限制横向速度，避免回正过猛
        vmax_lat = max(0.6, 0.35 * v0)
        if abs(v_lat) > vmax_lat:
            v = v0 * t + np.sign(v_lat) * vmax_lat * n
        return v


class Corridor(Road):
    """直廊：x ∈ [0, L]，y ∈ [0, W]（lat = y - W/2）。"""

    kind = "corridor"

    def __init__(self, length: float, width: float):
        super().__init__(width)
        self.length = float(length)

    def tangent(self, pos):
        return np.array([1.0, 0.0])

    def normal(self, pos):
        return np.array([0.0, 1.0])

    def lat_of(self, pos):
        return float(pos[1] - self.half_width)

    def to_world(self, s, lat):
        return np.array([s, lat + self.half_width])

    def wall_gap(self, pos, radius):
        y = float(pos[1])
        d_bottom = y - radius                     # 到下墙
        d_top = self.width - y - radius           # 到上墙
        if d_bottom < d_top:
            return d_bottom, np.array([0.0, 1.0])
        return d_top, np.array([0.0, -1.0])


class RingRoad(Road):
    """环道：中心线半径 R，路宽 W；lat = r - R。"""

    kind = "ring"

    def __init__(self, radius: float, width: float):
        super().__init__(width)
        self.radius = float(radius)

    def _theta(self, pos):
        return float(np.arctan2(pos[1], pos[0]))

    def tangent(self, pos):
        th = self._theta(pos)
        return np.array([-np.sin(th), np.cos(th)])

    def normal(self, pos):
        th = self._theta(pos)
        return np.array([np.cos(th), np.sin(th)])

    def lat_of(self, pos):
        return float(np.linalg.norm(pos) - self.radius)

    def to_world(self, s, lat):
        th = s / self.radius
        r = self.radius + lat
        return np.array([r * np.cos(th), r * np.sin(th)])

    def wall_gap(self, pos, radius):
        r = float(np.linalg.norm(pos))
        d_in = (r - radius) - (self.radius - self.half_width)
        d_out = (self.radius + self.half_width) - (r + radius)
        n = self.normal(pos)
        if d_in < d_out:
            return d_in, n            # 推到外侧
        return d_out, -n              # 推到内侧


class OpenArea(Road):
    """Unbounded plane; the initial circle is a layout, not a ring road or wall."""
    kind = "open"

    def __init__(self):
        super().__init__(np.inf)

    def wall_gap(self, pos, radius):
        return np.inf, np.zeros(2)


def forward(a, road):
    if road.kind == "open":
        d = a.goal - a.pos
        return d / max(np.linalg.norm(d), 1e-12)
    return road.tangent(a.pos) * a.sigma


# ==========================================================================
# 3) 主体
# ==========================================================================
@dataclass
class Agent:
    aid: int
    kind: str                      # "ped" | "car"
    pos: np.ndarray                # (2,) 世界坐标
    vel: np.ndarray                # (2,)
    heading: float = 0.0           # 朝向（rad）
    sigma: float = 1.0             # 沿道路的前进方向 +1 / -1
    lat_pref: float = 0.0          # 偏好横向位置
    arrived: bool = False
    t_arrive: float = np.nan
    s0: float = 0.0
    goal: np.ndarray = field(default_factory=lambda: np.zeros(2))

    # ---- 参数快捷方式 ----
    @property
    def p(self):
        return PARAMS[self.kind]

    @property
    def mass(self):
        return self.p["mass"]

    @property
    def radius(self):
        return self.p["radius"] if self.kind == "ped" else self.p["width"] / 2.0

    @property
    def length(self):
        return self.p["length"]

    @property
    def width(self):
        return self.p["width"]

    # ---- 用于"力"的圆盘分解 ----
    def discs(self):
        """返回 [(圆心, 半径), ...]。行人是 1 个圆，车是沿车轴 3 个圆。"""
        if self.kind == "ped":
            return [(self.pos, self.p["radius"])]
        L, W = self.p["length"], self.p["width"]
        r = W / 2.0
        off = (L / 2.0) - r            # 覆盖纵向长度，但不覆盖矩形角部
        c, s = np.cos(self.heading), np.sin(self.heading)
        u = np.array([c, s])           # 车轴单位向量
        return [(self.pos - off * u, r), (self.pos, r), (self.pos + off * u, r)]

    # ---- 碰撞体积的"外接半宽"，用于让行判据 ----
    def half_extent_lat(self):
        if self.kind == "ped":
            return self.p["radius"]
        # 车相对道路法向的投影半宽（车轴沿切线，故取车宽的一半）
        return self.p["width"] / 2.0

    def half_extent_long(self):
        """沿行进方向的半长：行人取半径，车取车长的一半。"""
        if self.kind == "ped":
            return self.p["radius"]
        return self.p["length"] / 2.0


# ==========================================================================
# 4) 力
# ==========================================================================
def _disc_arrays(agents):
    """把活跃主体的圆盘展平成数组。"""
    centers, radii, owner, kinds = [], [], [], []
    for i, a in enumerate(agents):
        if a.arrived:
            continue
        for c, r in a.discs():
            centers.append(c)
            radii.append(r)
            owner.append(i)
            kinds.append(a.kind)
    return (np.array(centers, float), np.array(radii, float),
            np.array(owner, int), kinds)


def social_forces(agents, lam_override=None):
    """返回 (N,2) 的加速度贡献。"""
    n = len(agents)
    acc = np.zeros((n, 2))
    if n == 0:
        return acc

    centers, radii, owner, kinds = _disc_arrays(agents)
    if len(centers) == 0:
        return acc

    # 两两圆盘
    d_vec = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(d_vec, axis=2)
    np.fill_diagonal(d, np.inf)
    safe = np.where(np.isfinite(d), d, 1.0)
    n_ij = d_vec / safe[..., None]

    same_owner = owner[:, None] == owner[None, :]
    r_sum = radii[:, None] + radii[None, :]
    overlap = np.maximum(0.0, r_sum - d)

    # 按主体类型取 A、B
    kind_arr = np.array(kinds)
    A = np.empty_like(d)
    B = np.empty_like(d)
    for (ki, kj), a_val in A_MAT.items():
        A[(kind_arr[:, None] == ki) & (kind_arr[None, :] == kj)] = a_val
    for (ki, kj), b_val in B_MAT.items():
        B[(kind_arr[:, None] == ki) & (kind_arr[None, :] == kj)] = b_val

    # 视野各向异性：按"接收者"主体的朝向算
    head = np.array([np.arctan2(a.vel[1], a.vel[0])
                     if np.linalg.norm(a.vel) > 1e-6 else a.heading
                     for a in agents])
    lam = np.array([PARAMS[a.kind]["lam"] for a in agents])
    # 从 i 指向 j 的方向（注意 n_ij 是 i-j，即由 j 指向 i）
    to_j = -n_ij
    fx = np.cos(head)[owner][:, None]
    fy = np.sin(head)[owner][:, None]
    cos_phi = to_j[..., 0] * fx + to_j[..., 1] * fy
    w = lam[owner][:, None] + (1 - lam[owner][:, None]) * (1 + cos_phi) / 2

    # ---- 横向走廊衰减 ----
    # 只有"横向上确实挡在我前面"的主体才产生显著作用力。
    # 没有这一项时，径向排斥会把 1.9 m 开外、本可以并排通过的车也算进来，
    # 结果是行人被硬生生推停、双方僵死（实测过）。
    # 形式取高斯 exp(-(d_lat/L_lat)^2)，与各向异性社会力的做法一致。
    delta = -d_vec
    d_lon = delta[..., 0] * fx + delta[..., 1] * fy
    d_lat = np.abs(delta[..., 0] * fy - delta[..., 1] * fx)
    L_lat = np.array([0.9 if a.kind == "ped" else 1.2 for a in agents])[owner][:, None]
    w_lat = np.exp(-(d_lat / L_lat) ** 2)
    w_lat = np.where(d_lon > 0, w_lat, 0.15 * w_lat)   # 身后作用显著减弱
    w = w * w_lat

    rep_mag = A * w * np.exp((r_sum - d) / B)
    cont_mag = K_CONTACT * overlap
    mag = rep_mag + cont_mag
    mag = np.where(same_owner | ~np.isfinite(d), 0.0, mag)

    # 切向摩擦项
    t_vec = np.stack([-n_ij[..., 1], n_ij[..., 0]], axis=-1)
    # 速度按"圆盘所属主体"展开到圆盘粒度，否则 (N,N,2) 对不上 (D,D,2)
    v_disc = agents_vel(agents)[owner]
    dv = v_disc[None, :, :] - v_disc[:, None, :]
    dv_t = (dv * t_vec).sum(-1)
    fric = (KAPPA_FRIC * overlap * dv_t)[..., None] * t_vec
    fric = np.where(same_owner[..., None] | ~np.isfinite(d)[..., None],
                    0.0, fric)

    f = mag[..., None] * n_ij + fric
    np.add.at(acc, owner, f.sum(axis=1))
    # 转成加速度
    for i, a in enumerate(agents):
        acc[i] /= a.mass
    return acc


def agents_vel(agents):
    return np.array([a.vel for a in agents], float)


def wall_forces(agents, road: Road):
    acc = np.zeros((len(agents), 2))
    if road.kind == "open":
        return acc
    for i, a in enumerate(agents):
        if a.arrived:
            continue
        for c, r in a.discs():
            gap, n_in = road.wall_gap(c, r)
            mag = A_WALL * np.exp(-gap / B_WALL) + K_WALL * max(0.0, -gap)
            acc[i] += mag * n_in / (a.mass * len(a.discs()))
    return acc


# ==========================================================================
# 5) 让行规则
# ==========================================================================
def yield_speed_scale(agents, road: Road):
    """前向净空不足时缩放期望速度。返回 (N,) 的缩放因子 ∈ [0,1]。"""
    n = len(agents)
    scale = np.ones(n)
    for i, a in enumerate(agents):
        if a.arrived:
            continue
        prm = a.p
        t_i = forward(a, road)
        n_i = np.array([-t_i[1], t_i[0]])
        best = np.inf
        for j, b in enumerate(agents):
            if i == j or b.arrived:
                continue
            d = b.pos - a.pos
            ahead = float(d @ t_i)                 # 纵向（前为正）
            if ahead <= 0.0 or ahead > prm["look"]:
                continue
            lat_sep = abs(float(d @ n_i))
            if lat_sep > (a.half_extent_lat() + b.half_extent_lat()
                          + prm["margin"]):
                continue                            # 横向错开，不算阻挡
            gap = ahead - a.half_extent_long() - b.half_extent_long()
            if gap < best:
                best = gap
        if np.isfinite(best):
            lo, hi = prm["d_stop"], prm["d_slow"]
            scale[i] = float(np.clip((best - lo) / max(hi - lo, 1e-6), 0.0, 1.0))
    return scale


# ==========================================================================
# 6) 仿真
# ==========================================================================
def step(agents, road: Road, dt=DT, keep_right=True, use_yield=True):
    n = len(agents)
    acc = np.zeros((n, 2))

    # 让行缩放每步只算一次（原来放在循环里会退化成 O(N^2)）
    scales = yield_speed_scale(agents, road) if use_yield else np.ones(n)

    # (1) 驱动项
    for i, a in enumerate(agents):
        if a.arrived:
            continue
        prm = a.p
        tangent = forward(a, road)
        if road.kind == "open":
            v_des = prm["v0"] * tangent
            # Anticipatory right-side avoidance, using relative position/velocity.
            # Same formula for both types; only horizon and lateral speed differ.
            if keep_right:
                risk = 0.0
                for b in agents:
                    if b is a or b.arrived:
                        continue
                    delta = b.pos - a.pos
                    if delta @ tangent <= 0:
                        continue
                    rel = a.vel - b.vel
                    tca = float(delta @ rel) / max(float(rel @ rel), 1e-9)
                    horizon = 6.0
                    if 0 < tca < horizon:
                        miss = np.linalg.norm(delta - tca * rel)
                        safe = a.half_extent_long() + b.half_extent_long() + 1.0
                        risk = max(risk, float(np.clip(1-miss/safe,0,1)) * (1-tca/horizon))
                right = np.array([tangent[1], -tangent[0]])
                v_des += (1.2 if a.kind == "ped" else 3.0) * risk * right
        else:
            lat_pref = a.lat_pref if keep_right else 0.0
            v_des = road.desired_velocity(a.pos, a.sigma, prm["v0"], lat_pref, k_lat=2.0)
        # Waiting only removes forward motion; lateral avoidance must remain possible.
        v_des -= (1.0 - scales[i]) * float(v_des @ tangent) * tangent
        acc[i] += (v_des - a.vel) / prm["tau"]

    # (2) 主体间作用力
    if n > 1:
        acc += social_forces(agents)

    # (3) 边界
    acc += wall_forces(agents, road)

    # (4) 积分（限幅）
    for i, a in enumerate(agents):
        if a.arrived:
            continue
        prm = a.p
        ax, ay = acc[i]
        # 分方向限幅：沿运动方向用 b_max 限制减速、a_max 限制加速
        speed = np.linalg.norm(a.vel)
        if speed > 1e-6:
            e = a.vel / speed
            raw_along = float(acc[i] @ e)
            along = np.clip(raw_along, -prm["b_max"], prm["a_max"])
            perp = acc[i] - raw_along * e
            acc[i] = along * e + perp
        else:
            mag = np.linalg.norm(acc[i])
            if mag > prm["a_max"]:
                acc[i] = acc[i] / mag * prm["a_max"]

        if a.kind == "car":
            # No lateral sliding: translate longitudinal acceleration and lateral
            # force into speed and curvature-limited heading changes.
            e = np.array([np.cos(a.heading), np.sin(a.heading)])
            side = np.array([-e[1], e[0]])
            speed = max(0.0, float(a.vel @ e))
            speed = float(np.clip(speed + np.clip(acc[i] @ e, -prm['b_max'], prm['a_max']) * dt, 0, prm['v0']))
            if road.kind == 'open':
                yaw_limit = min(0.6, speed / 5.2)
                yaw_rate = np.clip((acc[i] @ side) / max(speed, 0.5), -yaw_limit, yaw_limit)
                a.heading += float(yaw_rate) * dt
            a.vel = speed * np.array([np.cos(a.heading), np.sin(a.heading)])
        else:
            a.vel = a.vel + acc[i] * dt
            speed = np.linalg.norm(a.vel)
            if speed > 1.5 * prm['v0']:
                a.vel *= 1.5 * prm['v0'] / speed
            if road.kind == 'corridor':
                # No involuntary walking backward; waiting and sidestepping remain.
                a.vel[0] = a.sigma * max(0.0, a.sigma * a.vel[0])
            if np.linalg.norm(a.vel) > 1e-6:
                a.heading = float(np.arctan2(a.vel[1],a.vel[0]))
        a.pos = a.pos + a.vel * dt
    return acc


def simulate(agents, road: Road, duration: float, dt=DT,
             sample_dt=SAMPLE_DT, keep_right=True, use_yield=True,
             arrival_tol=1.2):
    n_sample = int(round(duration / sample_dt))
    sub = max(1, int(round(sample_dt / dt)))
    if not np.isclose(sub*dt, sample_dt):
        raise ValueError('sample_dt must be an integer multiple of dt')

    hist_pos = np.zeros((n_sample + 1, len(agents), 2))
    hist_vel = np.zeros_like(hist_pos)
    hist_head = np.zeros((n_sample + 1, len(agents)))

    def snapshot(k, t):
        for i, a in enumerate(agents):
            hist_pos[k, i] = a.pos
            hist_vel[k, i] = a.vel
            hist_head[k, i] = a.heading

    snapshot(0, 0.0)
    for k in range(1, n_sample + 1):
        for _ in range(sub):
            step(agents, road, dt, keep_right, use_yield)
        t = k * sample_dt
        # 到达判定
        for a in agents:
            reached = (a.sigma*a.pos[0] > (road.length if a.sigma > 0 else 0) + a.half_extent_long()) if road.kind == 'corridor' else np.linalg.norm(a.pos-a.goal) < arrival_tol
            if not a.arrived and reached:
                a.arrived = True
                a.t_arrive = t
                a.vel = a.vel * 0.0
        snapshot(k, t)
        if all(a.arrived for a in agents):
            break
    return dict(t=np.arange(k + 1) * sample_dt,
                pos=hist_pos[:k+1], vel=hist_vel[:k+1], heading=hist_head[:k+1])


# ==========================================================================
# 7) 指标
# ==========================================================================
def summary(agents, hist, road, label=""):
    """Use exact geometry and exclude agents after arrival."""
    from evaluate import evaluate
    return evaluate(agents, hist, road, label)
