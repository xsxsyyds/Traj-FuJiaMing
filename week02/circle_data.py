#!/usr/bin/env python3
"""week02 · 圆环对趾实验数据的统一读取与运动学量。

week02 的分析脚本与仿真脚本都从这里取数据，避免各自重复解析。

数据格式
--------
``data/circle-10m-64-1.txt``，5 列纯文本，空格分隔：

    agent  frame  x  y  unknown

- ``agent``    行人编号 1..64
- ``frame``    帧号 37..461，**64 人共用同一套连续帧**（每人恰好 425 帧）
- ``x, y``     位置，单位 cm（圆周半径约 1000 cm = 10 m）
- ``unknown``  每人为定值（160/170/180），与行走方向不对应，不使用

⚠️ 第 2 列是**帧号**而不是行人编号，极易读反。只有固定第 1 列、让第 2 列
递增，轨迹才会从圆周一端平滑穿过圆心走到对侧。

时间基
------
帧率 25 fps，即帧间隔 ``DT = 0.04 s``，每条记录总长 17.0 s。
位置由质心轨迹给出，速度用位置的中心差分估计；相邻帧位移约 5 cm，
差分噪声明显，因此速度默认先做一次 5 帧滑动平均（窗口 0.2 s）。
``smooth=1`` 可关闭（原始速度的长尾见 README 的数据说明）。

坐标系约定
----------
- 世界系：原点在圆心，目标点 = 起点关于圆心的对趾点，``goal = -start``。
- 对齐系：每条轨迹绕圆心旋转 ``-θ₀``，使起点落在 ``(+r, 0)``、目标落在
  ``(−r, 0)``。于是所有轨迹共用同一条出发点与同一条目标线，横向偏移
  ``y`` 可以直接横向比较。
- 对齐系中的**行进方向是 −x**，故 ``y > 0`` 为行进方向的右手侧、
  ``y < 0`` 为左手侧。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

# ---- 时间基 --------------------------------------------------------
FPS = 25.0
DT = 1.0 / FPS

# ---- 样式（全 week02 共用） ----------------------------------------
BASE = "#1F4E79"
ACCENT = "#C00000"
GRID = "#D9D9D9"
TEXT = "#333333"
CYCLIC = "twilight_shifted"      # 角度是周期量，用循环色图

#: 字体候选：表头用中文，因此必须有一个含 CJK 字形的字体。
#: Windows 上首选微软雅黑（拉丁与 CJK 都全，且不缺 U+2212 减号）。
CJK_FONTS = ("Microsoft YaHei", "Noto Sans SC", "SimHei", "SimSun",
             "DejaVu Sans")


def pick_font() -> str:
    """挑一个既含中文字形、又不缺减号的可用字体。"""
    from matplotlib import font_manager as fm

    have = {f.name for f in fm.fontManager.ttflist}
    for name in CJK_FONTS:
        if name in have:
            return name
    return "DejaVu Sans"


def apply_style() -> None:
    """设置 matplotlib 全局样式。绘图前调用一次。

    约定（2026-09-18）：**图题（表头）用中文，横纵坐标、图例、图内标注用
    英文**。因此字体必须同时具备 CJK 与拉丁字形；`axes.unicode_minus`
    关掉，避免中文字体缺 U+2212 时把负号渲染成方框。

    ⚠️ 微软雅黑**缺 U+2070（上标 0）**，所以 `v⁰` 这类写法会渲染成方框，
    图上一律写成 `v0`。其余常用符号（Δ τ κ — … 「」（） ： ¹²³ ° ± ≥）
    经检查都齐全。
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": pick_font(),
        "font.size": 9,
        "axes.unicode_minus": False,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#666666",
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "savefig.bbox": "tight",
    })


def save(fig, outdir: Path, stem: str, dpi: int = 300,
         root: Path | None = None) -> None:
    """只输出高清 PNG（week02 一律不出 PDF）。"""
    p = Path(outdir) / f"{stem}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, facecolor="white")
    try:
        shown = p.relative_to(root) if root else p
    except ValueError:
        shown = p
    print(f"[写出] {shown}")


def style_axes(ax, xlabel: str = "x [m]", ylabel: str = "y [m]",
               equal: bool = True) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if equal:
        ax.set_aspect("equal", adjustable="datalim")


def draw_ring(ax, R: float = 10.0, lw: float = 1.0) -> None:
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(R * np.cos(t), R * np.sin(t), ls=(0, (5, 4)),
            color="#9AA5B1", lw=lw, zorder=1)


# ------------------------------------------------------------------
# 工具
# ------------------------------------------------------------------

def _moving_average(a: np.ndarray, k: int) -> np.ndarray:
    """沿第 0 轴（时间）做长度 k 的滑动平均，边界用端点延拓。"""
    if k <= 1:
        return a.astype(float)
    pad = k // 2
    out = np.empty_like(a, dtype=float)
    kernel = np.ones(k) / k
    for d in range(a.shape[-1]):
        for i in range(a.shape[1]):
            x = a[:, i, d]
            out[:, i, d] = np.convolve(np.pad(x, pad, mode="edge"),
                                       kernel, mode="valid")[:x.size]
    return out


# ------------------------------------------------------------------
# 数据集
# ------------------------------------------------------------------

@dataclass
class CircleTrial:
    """一次圆环对趾实验的全部轨迹。位置单位米，时间单位秒。

    内部把所有轨迹整理成共用时基的张量 ``X[t, i]``（时间 × 行人 × 坐标），
    这是做速度、方向与密度分析的统一入口。
    """

    tracks: dict[int, np.ndarray]     # agent -> (N, 2) [x, y]，米
    heading: dict[int, float]         # agent -> 原始第 5 列取值
    frames: np.ndarray                # (N,) 帧号
    fps: float = FPS

    # ---- 基本形状 -------------------------------------------------
    @property
    def agents(self) -> list[int]:
        return sorted(self.tracks)

    @property
    def n_agent(self) -> int:
        return len(self.tracks)

    @property
    def n_frame(self) -> int:
        return int(self.frames.size)

    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.n_frame) * self.dt

    @property
    def duration(self) -> float:
        return (self.n_frame - 1) * self.dt

    def positions(self) -> np.ndarray:
        """(N, A, 2) 位置张量。"""
        return np.stack([self.tracks[a] for a in self.agents], axis=1)

    # ---- 几何（逐人） ---------------------------------------------
    def theta0(self, a: int) -> float:
        p = self.tracks[a][0]
        return float(np.arctan2(p[1], p[0]))

    def radius(self, a: int) -> np.ndarray:
        return np.linalg.norm(self.tracks[a], axis=1)

    def start(self, a: int) -> np.ndarray:
        return self.tracks[a][0]

    def goal(self, a: int) -> np.ndarray:
        """目标点 = 起点的对趾点。"""
        return -self.tracks[a][0]

    def goals(self) -> np.ndarray:
        """(A, 2) 所有行人的目标点。"""
        return -self.positions()[0]

    def r_start_end(self, a: int) -> tuple[float, float]:
        r = self.radius(a)
        return float(r[0]), float(r[-1])

    def r_min(self, a: int) -> float:
        return float(self.radius(a).min())

    def r_max(self, a: int) -> float:
        return float(self.radius(a).max())

    def path_length(self, a: int) -> float:
        return float(np.linalg.norm(np.diff(self.tracks[a], axis=0),
                                    axis=1).sum())

    def net_length(self, a: int) -> float:
        p = self.tracks[a]
        return float(np.linalg.norm(p[-1] - p[0]))

    def steps(self, a: int) -> np.ndarray:
        """相邻帧位移（米）。等时间采样下正比于速度。"""
        return np.linalg.norm(np.diff(self.tracks[a], axis=0), axis=1)

    # ---- 运动学（整场） -------------------------------------------
    def velocities(self, smooth: int = 5) -> np.ndarray:
        """(N, A, 2) 速度，单位 m/s。默认先平滑位置再中心差分。"""
        p = self.positions()
        if smooth and smooth > 1:
            p = _moving_average(p, smooth)
        return np.gradient(p, self.dt, axis=0)

    def speeds(self, smooth: int = 5) -> np.ndarray:
        """(N, A) 速率，单位 m/s。"""
        return np.linalg.norm(self.velocities(smooth), axis=2)

    def headings(self, smooth: int = 5) -> np.ndarray:
        """(N, A) 朝向角，单位弧度，未解缠。"""
        v = self.velocities(smooth)
        return np.arctan2(v[:, :, 1], v[:, :, 0])

    def turning_rate(self, smooth: int = 5) -> np.ndarray:
        """(N, A) 转向率，单位 deg/s。朝向先解缠再求导。"""
        h = np.unwrap(self.headings(smooth), axis=0)
        return np.degrees(np.gradient(h, self.dt, axis=0))

    # ---- 对齐系 ----------------------------------------------------
    def aligned(self, smooth: int = 0) -> np.ndarray:
        """(N, A, 2) 绕圆心旋转 −θ₀ 后的轨迹。

        只做纯旋转（保半径），所有起点落到 ``(+r, 0)``、目标落到
        ``(−r, 0)`` 附近——起点本身分布在半径 9.7~10.5 m 的薄环上，
        这个残差本身也是信息。
        """
        X = self.positions()
        if smooth and smooth > 1:
            X = _moving_average(X, smooth)
        out = np.empty_like(X)
        for i, a in enumerate(self.agents):
            th = self.theta0(a)
            c, s = np.cos(th), np.sin(th)
            out[:, i, 0] = X[:, i, 0] * c + X[:, i, 1] * s
            out[:, i, 1] = -X[:, i, 0] * s + X[:, i, 1] * c
        return out

    def lateral(self, smooth: int = 0) -> np.ndarray:
        """(N, A) 对齐系中的横向偏移，单位米。``y > 0`` 为行进方向右手侧。"""
        return self.aligned(smooth)[:, :, 1]

    def midjourney_index(self) -> np.ndarray:
        """(A,) 每人走完自己累计路程一半时对应的帧下标。

        用"各自的半程"而不按时间百分比取点，是因为不同配置下走完全程
        所需的时间不同；按半程取样才可比。
        """
        X = self.positions()
        seg = np.linalg.norm(np.diff(X, axis=0), axis=2)          # (N−1, A)
        cum = np.concatenate([np.zeros((1, self.n_agent)),
                              np.cumsum(seg, axis=0)], axis=0)
        total = cum[-1]
        out = np.empty(self.n_agent, dtype=int)
        for i in range(self.n_agent):
            out[i] = (int(np.searchsorted(cum[:, i], 0.5 * total[i]))
                      if total[i] > 0 else self.n_frame // 2)
        return np.clip(out, 0, self.n_frame - 1)

    def lateral_iqr(self) -> float:
        """每人行至各自半程时横向偏移的跨人 IQR [m]，衡量"人流散开多宽"。

        与轨迹总时长无关，因此实测（17 s）与仿真（25 s）可以直接比较。
        """
        lat = self.lateral()
        j = self.midjourney_index()
        v = lat[j, np.arange(self.n_agent)]
        return float(np.percentile(v, 75) - np.percentile(v, 25))

    def bearing_deviation(self, smooth: int = 5) -> np.ndarray:
        """(N, A) 与"起点→目标直线"的夹角，单位度，带符号。

        正 = 偏向行进方向右侧（对齐系的 +y），负 = 偏左。
        速率过低的样本方向无意义，返回 ``nan``。
        """
        v = self.velocities(smooth)
        sp = np.linalg.norm(v, axis=2)
        X = self.positions()
        u = -X[0] / np.linalg.norm(X[0], axis=1, keepdims=True)
        vd = v / np.maximum(sp, 1e-9)[:, :, None]
        cos = np.clip((vd * u[None]).sum(2), -1, 1)
        sin = vd[:, :, 0] * (-u[None, :, 1]) + vd[:, :, 1] * u[None, :, 0]
        dev = np.degrees(np.arctan2(sin, cos))
        dev[sp < 0.3] = np.nan
        return dev

    # ---- 到达 ------------------------------------------------------
    def arrival_index(self, tol: float = 0.6) -> np.ndarray:
        """(A,) 每人"再也不离开目标点半径 tol 之内"的那一帧下标。

        用"最后一次距离大于 tol 的帧"加一，因此中途路过目标又走开的人
        不会被误判为已到达。
        """
        X = self.positions()
        d = np.linalg.norm(X - self.goals()[None], axis=2)
        out = np.empty(self.n_agent, dtype=int)
        for i in range(self.n_agent):
            far = np.where(d[:, i] >= tol)[0]
            out[i] = int(far.max() + 1) if far.size else 0
        return out

    def arrival_time(self, tol: float = 0.6) -> np.ndarray:
        """(A,) 到达时刻，单位秒。"""
        return self.arrival_index(tol) * self.dt

    def travel_mask(self, tol: float = 0.6, head: int = 5,
                    tail: int = 12) -> np.ndarray:
        """(N, A) 布尔掩码：只保留"行进段"。

        掐掉起步的 ``head`` 帧与到达前的 ``tail`` 帧，避免把起步加速段
        与到达后站立段混进行进统计。
        """
        arr = self.arrival_index(tol)
        m = np.zeros((self.n_frame, self.n_agent), dtype=bool)
        for i in range(self.n_agent):
            hi = max(head + 1, arr[i] - tail)
            m[head:hi, i] = True
        return m

    # ---- 相互作用 --------------------------------------------------
    def neighbours(self, radius: float = 2.0, smooth: int = 1) -> np.ndarray:
        """(N, A) 每人在给定半径内的邻居数（不含自己）。"""
        X = self.positions()
        if smooth and smooth > 1:
            X = _moving_average(X, smooth)
        out = np.zeros((self.n_frame, self.n_agent))
        for t in range(self.n_frame):
            tree = cKDTree(X[t])
            out[t] = tree.query_ball_point(X[t], radius,
                                           return_length=True) - 1
        return out

    def nearest_distance(self) -> np.ndarray:
        """(N,) 每帧全场最近的两人间距，单位米。"""
        X = self.positions()
        out = np.empty(self.n_frame)
        for t in range(self.n_frame):
            tree = cKDTree(X[t])
            dd, _ = tree.query(X[t], k=2)
            out[t] = dd[:, 1].min()
        return out


def load(path: Path | str) -> CircleTrial:
    """读取圆环对趾数据文件。"""
    path = Path(path)
    raw = np.loadtxt(path)
    if raw.ndim != 2 or raw.shape[1] < 5:
        raise ValueError(f"{path} 期望 5 列，实际 {raw.shape}")

    agent, sample, x, y, hd = (raw[:, 0], raw[:, 1], raw[:, 2],
                               raw[:, 3], raw[:, 4])
    tracks: dict[int, np.ndarray] = {}
    head: dict[int, float] = {}
    for a in np.unique(agent).astype(int):
        m = agent == a
        order = np.argsort(sample[m])
        tracks[a] = np.column_stack([x[m][order], y[m][order]]) / 100.0
        head[a] = float(np.unique(hd[m])[0])

    frames = np.sort(np.unique(sample))
    return CircleTrial(tracks=tracks, heading=head, frames=frames)


def default_data_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "circle-10m-64-1.txt"


# ------------------------------------------------------------------
# 自检
# ------------------------------------------------------------------

def main() -> int:
    tr = load(default_data_path())
    print(f"行人数 {tr.n_agent}   帧数 {tr.n_frame}   时长 {tr.duration:.1f} s"
          f"   ({tr.fps:g} fps)")
    sp = tr.speeds()
    print(f"速率 中位 {np.median(sp):.2f} m/s   最大 {sp.max():.2f} m/s")
    arr = tr.arrival_time()
    print(f"到达 中位 {np.median(arr):.1f} s   最晚 {arr.max():.1f} s")
    n = tr.neighbours(2.0)
    print(f"2 m 内邻居 中位 {np.median(n):.1f}   最大 {int(n.max())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
