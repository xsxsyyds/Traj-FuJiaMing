# -*- coding: utf-8 -*-
"""
三种经典跟驰模型的统一复现框架。

包含：
    OVM   最优速度模型   Bando et al. (1995), Phys. Rev. E 51, 1035
    FVDM  全速度差模型   Jiang, Wu & Zhu (2001), Phys. Rev. E 64, 017101
    IDM   智能驾驶员模型 Treiber, Hennecke & Helbing (2000), Phys. Rev. E 62, 1805

----------------------------------------------------------------------
统一接口约定（三个模型共用，避免符号、单位混淆）
----------------------------------------------------------------------
输入量一律用**净间距** gap_net，单位 m，定义为
    前车车尾 到 本车车头 之间的空档距离（bumper-to-bumper）
这样三个模型可以用同一个函数签名调用：

    accel(v_self, v_lead, gap_net) -> 加速度 [m/s^2]

    v_self   本车速度   [m/s]
    v_lead   前车速度   [m/s]
    gap_net  净间距     [m]

注意符号陷阱：OVM/FVDM 原文里的 s 是**车头间距（质心到质心）**，
与净间距差一个车长 l_c，即 s = gap_net + l_c。IDM 原文里的 s 本身就是净间距。
本模块把这一差异吸收在内部，对外只暴露净间距。

积分器用 Treiber 推荐的弹道格式（对匀加速精确）：
    x <- x + v dt + 0.5 a dt^2
    v <- v + a dt
"""

from __future__ import annotations

import numpy as np

# 三个模型原文采用的车长（Bando/Jiang 的 l_c 与 Treiber 的 l 恰好都取 5 m）
VEHICLE_LENGTH_M = 5.0

# 数值保护：仅用于避免浮点溢出，不改变物理
_ACCEL_ABS_MAX = 20.0     # m/s^2，超出即视为数值失效
_GAP_FLOOR = 0.05         # m，用于计算加速度时的分母下限


# ==========================================================================
# 0) 最优速度函数 V(s)
# ==========================================================================
class OptimalVelocityFunction:
    r"""OVM 与 FVDM 共用的最优速度函数。

    原文形式（Bando 1995 提出，Helbing & Tilch 1998 对标定）：

        V(s) = V1 + V2 * tanh[ C1 * (s - l_c) - C2 ]

    其中 s 是车头间距（质心到质心），l_c 是车长。代入 s = gap_net + l_c 后，
    参数化形式等价于「只依赖净间距」：

        V(gap_net) = V1 + V2 * tanh( C1 * gap_net - C2 )

    标定值（Helbing & Tilch 1998；Jiang 2001 第 5 式沿用）：
        V1 = 6.75 m/s, V2 = 7.91 m/s, C1 = 0.13 1/m, C2 = 1.57, l_c = 5 m
    """

    def __init__(self, V1=6.75, V2=7.91, C1=0.13, C2=1.57, lc=VEHICLE_LENGTH_M,
                 clamp_min=0.0):
        self.V1 = float(V1)
        self.V2 = float(V2)
        self.C1 = float(C1)
        self.C2 = float(C2)
        self.lc = float(lc)
        self.clamp_min = clamp_min

    def __call__(self, gap_net):
        raw = self.V1 + self.V2 * np.tanh(self.C1 * np.asarray(gap_net, dtype=float) - self.C2)
        if self.clamp_min is None:
            return raw
        return np.maximum(raw, self.clamp_min)

    # ---- 供校验用的解析性质 ----
    @property
    def v_max(self):
        """净间距 → ∞ 时的最优速度，即模型的速度上限。"""
        return self.V1 + self.V2

    @property
    def gap_at_zero_speed(self):
        """V = 0 对应的净间距（原文中 7.4 m 车头间距对应的静止状态）。"""
        x = np.arctanh(-self.V1 / self.V2)
        return (x + self.C2) / self.C1

    def inverse(self, v):
        """由目标速度反解净间距：V(gap) = v  =>  gap。"""
        v = np.asarray(v, dtype=float)
        ratio = np.clip((v - self.V1) / self.V2, -1.0 + 1e-12, 1.0 - 1e-12)
        x = np.arctanh(ratio)
        return (x + self.C2) / self.C1

    def as_dict(self):
        return dict(V1=self.V1, V2=self.V2, C1=self.C1, C2=self.C2,
                    lc=self.lc, v_max=self.v_max)


# ==========================================================================
# 1) 模型基类
# ==========================================================================
class CarFollowingModel:
    key = "base"
    label_zh = "基类"
    label_en = "base"
    paper = ""
    color = "#333333"

    def accel(self, v_self, v_lead, gap_net):
        raise NotImplementedError

    def equilibrium_gap(self, v):
        """稳态净间距：净间距与速度都不再变化时对应的 gap。"""
        raise NotImplementedError

    def v_ceiling(self):
        """该模型能达到的自由流速度上限 [m/s]。"""
        return np.inf

    def params(self):
        return {}

    # ---- 便捷批量入口 ----
    def step_states(self, v, v_lead, gap):
        """向量化：对本车速度/前车速度/净间距数组求加速度。"""
        return self.accel(np.asarray(v, dtype=float),
                          np.asarray(v_lead, dtype=float),
                          np.asarray(gap, dtype=float))


# ==========================================================================
# 2) OVM —— Bando et al. (1995)
# ==========================================================================
class OVM(CarFollowingModel):
    r"""最优速度模型（Optimal Velocity Model）。

    动力学方程（Bando 1995 第 2 式）：

        dv_n/dt = kappa * [ V(s_n) - v_n ]

    s_n 为车头间距，kappa 为驾驶员敏感度 [1/s]。

    本实现采用 Helbing & Tilch (1998) 的标定参数 kappa = 0.85 1/s，
    与 Jiang (2001) 表 I 中 OVM 一栏一致，便于横向对比。

    结构性特点：只对「速度与最优速度之差」响应，**完全没有相对速度项**，
    因此前车急减速时无法提前感知，只能等间距被压缩后才反应。
    """

    key = "ovm"
    label_zh = "OVM 最优速度模型"
    label_en = "OVM"
    paper = "Bando et al. 1995, Phys. Rev. E 51, 1035"
    color = "#1F4E79"

    def __init__(self, kappa=0.85, ovf=None, **ovf_kw):
        self.kappa = float(kappa)
        self.ovf = ovf if ovf is not None else OptimalVelocityFunction(**ovf_kw)

    def accel(self, v_self, v_lead, gap_net):
        v_self = np.asarray(v_self, dtype=float)
        v_opt = self.ovf(np.maximum(gap_net, _GAP_FLOOR))
        return self.kappa * (v_opt - v_self)

    def desired_speed(self, gap_net):
        return self.ovf(gap_net)

    def equilibrium_gap(self, v):
        return self.ovf.inverse(v)

    def v_ceiling(self):
        return self.ovf.v_max

    def params(self):
        return dict(kappa=self.kappa, **self.ovf.as_dict())


# ==========================================================================
# 3) FVDM —— Jiang, Wu & Zhu (2001)
# ==========================================================================
class FVDM(CarFollowingModel):
    r"""全速度差模型（Full Velocity Difference Model）。

    动力学方程（Jiang 2001 第 7 式）：

        dv_{n+1}/dt = kappa * [ V(s) - v_{n+1} ] + lam * (v_n - v_{n+1})

    即在 OVM 上增加「全」速度差项：不论前后车谁快谁慢都计入。
    （此前的 GFM 只用 Heaviside 函数保留 v_n < v_{n+1} 的那一半。）

    原文参数：kappa = 0.41 1/s（同 GFM），lam 取阶跃函数
        lam = 0.5 1/s  (s <= s_c = 100 m)，  0 (s > s_c)
    原文说明在仿真中车头间距不超过 100 m，故等价于常数 lam = 0.5 1/s。
    """

    key = "fvdm"
    label_zh = "FVDM 全速度差模型"
    label_en = "FVDM"
    paper = "Jiang, Wu & Zhu 2001, Phys. Rev. E 64, 017101"
    color = "#C00000"

    def __init__(self, kappa=0.41, lam=0.5, s_c=100.0, lam_far=0.0,
                 ovf=None, **ovf_kw):
        self.kappa = float(kappa)
        self.lam = float(lam)
        self.s_c = float(s_c)
        self.lam_far = float(lam_far)
        self.ovf = ovf if ovf is not None else OptimalVelocityFunction(**ovf_kw)

    def accel(self, v_self, v_lead, gap_net):
        v_self = np.asarray(v_self, dtype=float)
        v_lead = np.asarray(v_lead, dtype=float)
        gap_net = np.asarray(gap_net, dtype=float)
        # 原文按车头间距判断，此处换算成净间距
        lam = np.where(gap_net + self.ovf.lc <= self.s_c, self.lam, self.lam_far)
        v_opt = self.ovf(np.maximum(gap_net, _GAP_FLOOR))
        return self.kappa * (v_opt - v_self) + lam * (v_lead - v_self)

    def desired_speed(self, gap_net):
        return self.ovf(gap_net)

    def equilibrium_gap(self, v):
        """稳态时 dv/dt = 0 且 v_lead = v_self，故退化为 V(gap) = v。"""
        return self.ovf.inverse(v)

    def v_ceiling(self):
        return self.ovf.v_max

    def params(self):
        return dict(kappa=self.kappa, lam=self.lam, s_c=self.s_c,
                    lam_far=self.lam_far, **self.ovf.as_dict())


# ==========================================================================
# 4) IDM —— Treiber, Hennecke & Helbing (2000)
# ==========================================================================
class IDM(CarFollowingModel):
    r"""智能驾驶员模型（Intelligent Driver Model）。

    加速度（Treiber 2000 第 6 式）：

        dv_a/dt = a * [ 1 - (v_a / v0)^delta - ( s*(v_a, dv_a) / s_a )^2 ]

    期望净间距（第 7 式，取 s1 = 0）：

        s* = s0 + s1 * sqrt(v/v0) + T v + v * dv / (2 sqrt(a b))

    其中 dv = v_self - v_lead 为「接近率」，dv > 0 表示正在逼近前车。
    原文 Table I 参数：
        v0 = 120 km/h = 33.33 m/s，T = 1.6 s，a = 0.73 m/s^2，
        b = 1.67 m/s^2，delta = 4，s0 = 2 m，s1 = 0，l = 5 m

    稳态净间距（第 8 式）：
        s_e(v) = (s0 + v T) / sqrt(1 - (v/v0)^delta)

    结构性特点：制动项同时依赖净间距与接近率，因此天生「不撞车」；
    且加速与制动使用不同参数（a != b），加减速不对称，符合实测。
    """

    key = "idm"
    label_zh = "IDM 智能驾驶员模型"
    label_en = "IDM"
    paper = "Treiber, Hennecke & Helbing 2000, Phys. Rev. E 62, 1805"
    color = "#2A9D8F"

    def __init__(self, v0=33.3333, T=1.6, a=0.73, b=1.67, delta=4,
                 s0=2.0, s1=0.0, length=VEHICLE_LENGTH_M):
        self.v0 = float(v0)
        self.T = float(T)
        self.a = float(a)
        self.b = float(b)
        self.delta = float(delta)
        self.s0 = float(s0)
        self.s1 = float(s1)
        self.length = float(length)

    def accel(self, v_self, v_lead, gap_net):
        v = np.maximum(np.asarray(v_self, dtype=float), 0.0)
        v_lead = np.asarray(v_lead, dtype=float)
        s = np.maximum(np.asarray(gap_net, dtype=float), _GAP_FLOOR)

        s_star = (self.s0
                  + self.s1 * np.sqrt(v / self.v0)
                  + self.T * v
                  + v * (v - v_lead) / (2.0 * np.sqrt(self.a * self.b)))

        return self.a * (1.0 - (v / self.v0) ** self.delta - (s_star / s) ** 2)

    def desired_gap(self, v_self, v_lead=0.0):
        v = np.maximum(np.asarray(v_self, dtype=float), 0.0)
        return (self.s0
                + self.s1 * np.sqrt(v / self.v0)
                + self.T * v
                + v * (v - np.asarray(v_lead, dtype=float)) / (2.0 * np.sqrt(self.a * self.b)))

    def equilibrium_gap(self, v):
        v = np.asarray(v, dtype=float)
        denom = np.sqrt(np.maximum(1.0 - (v / self.v0) ** self.delta, 1e-12))
        return (self.s0 + v * self.T) / denom

    def v_ceiling(self):
        return self.v0

    def params(self):
        return dict(v0=self.v0, T=self.T, a=self.a, b=self.b, delta=self.delta,
                    s0=self.s0, s1=self.s1, length=self.length)


# ==========================================================================
# 5) 工厂
# ==========================================================================
def build_models(which=None):
    """返回三个模型的实例列表（默认全部，按论文发表顺序）。"""
    models = [OVM(), FVDM(), IDM()]
    if which:
        models = [m for m in models if m.key in set(which)]
    return models


def model_by_key(key):
    for m in build_models():
        if m.key == key:
            return m
    raise KeyError(key)


# ==========================================================================
# 6) 数值保护与失败判定
# ==========================================================================
def equilibrium_speed(model, gap_net):
    """给定净间距，解出稳态速度（equilibrium_gap 对 v 单调递增，用二分）。

    OVM/FVDM 有解析解 V(gap)；IDM 需数值反解。
    """
    gap_net = float(np.ravel(gap_net)[0])
    v_hi = model.v_ceiling()
    if not np.isfinite(v_hi):
        v_hi = 60.0
    lo, hi = 0.0, float(v_hi)
    if float(np.ravel(model.equilibrium_gap(lo))[0]) > gap_net:
        return 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(np.ravel(model.equilibrium_gap(mid))[0]) < gap_net:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def guard_accel(a):
    """把加速度裁到数值安全区间，返回 (裁剪后的值, 是否触发保护)。"""
    a = np.asarray(a, dtype=float)
    hit = np.abs(a) > _ACCEL_ABS_MAX
    a = np.clip(np.nan_to_num(a, nan=0.0, posinf=_ACCEL_ABS_MAX,
                              neginf=-_ACCEL_ABS_MAX),
                -_ACCEL_ABS_MAX, _ACCEL_ABS_MAX)
    return a, hit


# ==========================================================================
# 7) 仿真器
# ==========================================================================
def simulate_follower(model, t_grid, lead_x, lead_v, x0, v0, gap0,
                      dt=0.02, stop_on_failure=True, v_min=None):
    """给定前车轨迹，前向积分跟驰者的运动。

    参数
    ----
    model      : CarFollowingModel
    t_grid     : (n,) 输出时刻 [s]，须等间隔
    lead_x     : (n,) 前车**后保险杠**位置 [m]（即前车车头位置 - 车长）
    lead_v     : (n,) 前车速度 [m/s]
    x0, v0     : 跟驰者初值（车头位置 [m]、速度 [m/s]）
    gap0       : t_grid[0] 时刻的净间距 [m]，用于一致性校验
    dt         : 内部积分步长 [s]
    stop_on_failure : 首次判定失败后是否冻结状态
    v_min      : 速度下界；给 0.0 表示「不允许倒车」（真实驾驶员不会倒退跟驰）。
                 None 表示不作限制，此时 OVM 型模型在间距被压到静止平衡间距
                 以下时会解出负速度，属模型固有缺陷，需单独报告。

    返回
    ----
    dict(x, v, a, gap, failed, t_fail, fail_reason, a_peak_dec, n_clamped)
    """
    t_grid = np.asarray(t_grid, dtype=float)
    lead_x = np.asarray(lead_x, dtype=float)
    lead_v = np.asarray(lead_v, dtype=float)
    if lead_x.shape != t_grid.shape:
        raise ValueError("lead_x 与 t_grid 长度不一致")

    # 线性插值函数（前车轨迹按 0.1 s 采样，内部步长更细）
    def lead_state(t):
        t = np.clip(t, t_grid[0], t_grid[-1])
        x = np.interp(t, t_grid, lead_x)
        v = np.interp(t, t_grid, lead_v)
        return x, v

    n = len(t_grid)
    x = np.empty(n)
    v = np.empty(n)
    acc = np.empty(n)
    gap = np.empty(n)

    x[0] = float(x0)
    v[0] = float(v0)
    gap[0] = float(gap0)

    failed = False
    t_fail = np.nan
    fail_reason = ""
    a_peak_dec = 0.0
    a_peak_acc = 0.0
    n_clamped = 0
    v_at_fail = np.nan
    gap_at_fail = np.nan

    for i in range(n - 1):
        if failed and stop_on_failure:
            x[i + 1] = x[i]
            v[i + 1] = v[i]
            acc[i + 1] = 0.0
            gap[i + 1] = gap[i]
            continue

        t = t_grid[i]
        t_next = t_grid[i + 1]
        n_sub = max(1, int(round((t_next - t) / dt)))
        h = (t_next - t) / n_sub

        xi, vi = x[i], v[i]
        for _ in range(n_sub):
            lx, lv = lead_state(t)
            g = (lx - xi)          # 前车后保险杠 - 本车车头 = 净间距
            a_raw = float(np.ravel(model.accel(vi, lv, g))[0])
            a_safe, hit = guard_accel(a_raw)
            a_safe = float(a_safe)
            # 弹道积分（对匀加速精确）
            xi = xi + vi * h + 0.5 * a_safe * h * h
            vi = vi + a_safe * h
            if v_min is not None and vi < v_min:
                vi = float(v_min)
                n_clamped += 1
            t += h

        x[i + 1] = xi
        v[i + 1] = vi
        lx, lv = lead_state(t_next)
        g_next = lx - xi
        gap[i + 1] = g_next
        acc[i + 1] = (v[i + 1] - v[i]) / max(t_next - t_grid[i], 1e-12)

        a_peak_dec = min(a_peak_dec, float(acc[i + 1]))
        a_peak_acc = max(a_peak_acc, float(acc[i + 1]))

        # ---- 失败判定 ----
        if g_next <= 0.0:
            # 记下「发生接触时的相对接近速度」——这才是事故严重程度，
            # 而净间距本身只会停在 0 附近（检测到即刻冻结）
            failed, t_fail, fail_reason = True, t_next, "collision"
            v_at_fail, gap_at_fail = float(vi - lv), float(g_next)
        elif v_min is None and vi < -1e-6:
            # 未启用不倒车约束时才把「解出负速度」记为失效（模型固有缺陷）
            failed, t_fail, fail_reason = True, t_next, "reverse"
            v_at_fail, gap_at_fail = float(vi), float(g_next)
        elif not np.isfinite(vi) or abs(vi) > 1e3:
            failed, t_fail, fail_reason = True, t_next, "divergence"
            v_at_fail, gap_at_fail = float(vi), float(g_next)

    return dict(x=x, v=v, a=acc, gap=gap, failed=failed, t_fail=t_fail,
                fail_reason=fail_reason, a_peak_dec=a_peak_dec,
                a_peak_acc=a_peak_acc, n_clamped=n_clamped,
                v_at_fail=v_at_fail, gap_at_fail=gap_at_fail)


def solve_constant_leader(model, v_lead, gap0, v_follow0, t_end, dt=0.02,
                          t_step=0.02):
    """前车恒定速度场景：解析给出前车轨迹，积分跟驰者。

    前车后保险杠位置取 lx(t) = x0_lead_rear + v_lead * t，
    跟驰者车头初位置取 x0 = lx(0) - gap0。
    """
    n = int(round(t_end / t_step)) + 1
    t = np.arange(n) * t_step
    lead_rear = v_lead * t                       # 前车后杠位置（相对坐标）
    lead_v = np.full(n, float(v_lead))
    x0 = lead_rear[0] - gap0
    out = simulate_follower(model, t, lead_rear, lead_v, x0, v_follow0,
                            gap0, dt=dt)
    out["t"] = t
    out["lead_x_rear"] = lead_rear
    out["lead_x_front"] = lead_rear + VEHICLE_LENGTH_M
    out["lead_v"] = lead_v
    return out


def simulate_ring(model, n_cars, ring_length, t_end, dt=0.01, x_init=None,
                  v_init=None):
    """环形道路仿真（周期边界），用于线性稳定性校验。

    默认初值取 Bando 1995 的扰动方式：car 1 前移 0.1 m，其余均匀分布。
    """
    n_steps = int(round(t_end / dt))
    headway = ring_length / n_cars

    if x_init is None:
        x = np.arange(n_cars, dtype=float) * headway
        x[0] += 0.1
    else:
        x = np.array(x_init, dtype=float)

    if v_init is None:
        # 从稳态出发：净间距 = 车头间距 - 车长
        v_eq = equilibrium_speed(model, headway - VEHICLE_LENGTH_M)
        # 速度存到数组里，保持与自定义初值同类型
        v = np.full(n_cars, float(v_eq))
    else:
        v = np.array(v_init, dtype=float)

    hist_x = np.empty((n_steps + 1, n_cars))
    hist_v = np.empty((n_steps + 1, n_cars))
    hist_x[0] = x
    hist_v[0] = v

    for k in range(n_steps):
        x_sorted = np.sort(x)
        # 对每个车找它的直接前车（环上）
        idx = np.argsort(x)
        # gap 用排序后的相邻差：前车是排序中后一位
        gaps_sorted = np.empty(n_cars)
        gaps_sorted[:-1] = np.diff(x_sorted)
        gaps_sorted[-1] = (x_sorted[0] + ring_length) - x_sorted[-1]
        lead_v_sorted = np.empty(n_cars)
        lead_v_sorted[:-1] = v[idx][1:]
        lead_v_sorted[-1] = v[idx][0]

        gap = np.empty(n_cars)
        lead_v = np.empty(n_cars)
        gap[idx] = gaps_sorted - VEHICLE_LENGTH_M      # 净间距
        lead_v[idx] = lead_v_sorted

        a, _ = guard_accel(model.accel(v, lead_v, gap))
        a = np.asarray(a, dtype=float)

        x = x + v * dt + 0.5 * a * dt * dt
        v = v + a * dt
        x = np.mod(x, ring_length)

        hist_x[k + 1] = x
        hist_v[k + 1] = v

    return dict(t=np.arange(n_steps + 1) * dt, x=hist_x, v=hist_v)
