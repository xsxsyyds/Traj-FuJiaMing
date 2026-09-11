"""行人轨迹数据统一读取模块。

两套数据集在此处被抹平为同一接口：

======================  ============================  ==============
数据集                  原始文件                       坐标单位
======================  ============================  ==============
ETH / EWAP              eth/<seq>/obsmat.txt           米（世界坐标）
UCY Crowd               ucy/<seq>/<seq>.txt            米（地面平面）
======================  ============================  ==============

关于 UCY 的单位
---------------
UCY 官方发布的 `.vsp` 文件是**图像像素坐标**，带透视压缩，不能直接用于
速度、间距、密度等物理量的计算。本模块读取的是 `datasets/build_ucy_metric.py`
生成的米制版本，行人编号与原始 `.vsp` 一一对应。

用法
----
    from common.traj_io import load_scene, SCENES

    sc = load_scene("seq_eth")
    for pid, tr in sc.tracks.items():
        t, x, y = tr[:, 0], tr[:, 1], tr[:, 2]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# 所有序列的原始视频与标注均为 25 fps
FPS = 25.0

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets"

# 场景名 -> (数据集, 相对数据文件路径)
SCENES: dict[str, tuple[str, str]] = {
    "seq_eth":    ("ETH", "eth/seq_eth/obsmat.txt"),
    "seq_hotel":  ("ETH", "eth/seq_hotel/obsmat.txt"),
    "zara01":     ("UCY", "ucy/zara01/zara01.txt"),
    "zara02":     ("UCY", "ucy/zara02/zara02.txt"),
    "students03": ("UCY", "ucy/students03/students03.txt"),
}

ETH_SCENES = [k for k, v in SCENES.items() if v[0] == "ETH"]
UCY_SCENES = [k for k, v in SCENES.items() if v[0] == "UCY"]


@dataclass
class Scene:
    """一个场景的全部轨迹。

    Attributes
    ----------
    name : 场景名，如 ``seq_eth``
    dataset : ``ETH`` 或 ``UCY``
    source : 数据文件路径
    tracks : 行人 ID -> ``(N, 3)`` 数组，三列依次为 时间[s]、x[m]、y[m]
    """

    name: str
    dataset: str
    source: Path
    tracks: dict[int, np.ndarray] = field(default_factory=dict)

    # ---- 基本统计 -------------------------------------------------
    @property
    def n_ped(self) -> int:
        return len(self.tracks)

    @property
    def n_point(self) -> int:
        return int(sum(len(v) for v in self.tracks.values()))

    @property
    def t_range(self) -> tuple[float, float]:
        t0 = min(v[0, 0] for v in self.tracks.values())
        t1 = max(v[-1, 0] for v in self.tracks.values())
        return float(t0), float(t1)

    @property
    def duration(self) -> float:
        t0, t1 = self.t_range
        return t1 - t0

    # ---- 几何 -----------------------------------------------------
    def bounds(self) -> tuple[float, float, float, float]:
        """返回 ``(xmin, xmax, ymin, ymax)``，单位为米。"""
        pts = self.points()
        return (float(pts[:, 0].min()), float(pts[:, 0].max()),
                float(pts[:, 1].min()), float(pts[:, 1].max()))

    def points(self) -> np.ndarray:
        """所有轨迹点拼成的 ``(M, 2)`` 数组，用于统计与散点绘制。"""
        return np.vstack([v[:, 1:3] for v in self.tracks.values()])

    def track_length(self, pid: int) -> float:
        """单个行人的累计路程，单位米。"""
        p = self.tracks[pid][:, 1:3]
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())

    def summary(self) -> dict:
        x0, x1, y0, y1 = self.bounds()
        t0, t1 = self.t_range
        return {
            "场景": self.name,
            "数据集": self.dataset,
            "行人数": self.n_ped,
            "轨迹点数": self.n_point,
            "时长/s": round(t1 - t0, 1),
            "x范围/m": (round(x0, 1), round(x1, 1)),
            "y范围/m": (round(y0, 1), round(y1, 1)),
        }


# ------------------------------------------------------------------
# 读取实现
# ------------------------------------------------------------------

def _load_eth(path: Path) -> dict[int, np.ndarray]:
    """ETH ``obsmat.txt``：``frame id pos_x pos_z pos_y v_x v_z v_y``。

    ``pos_z`` / ``v_z`` 垂直于地面，官方说明未使用，此处忽略。
    """
    raw = np.loadtxt(path)
    tracks: dict[int, np.ndarray] = {}
    for pid in np.unique(raw[:, 1]).astype(int):
        d = raw[raw[:, 1] == pid]
        d = d[np.argsort(d[:, 0])]
        tracks[pid] = np.column_stack([d[:, 0] / FPS, d[:, 2], d[:, 4]])
    return tracks


def _load_ucy(path: Path) -> dict[int, np.ndarray]:
    """UCY 米制版：``frame_id ped_id x[m] y[m]``。"""
    raw = np.loadtxt(path)
    tracks: dict[int, np.ndarray] = {}
    for pid in np.unique(raw[:, 1]).astype(int):
        d = raw[raw[:, 1] == pid]
        d = d[np.argsort(d[:, 0])]
        tracks[pid] = np.column_stack([d[:, 0] / FPS, d[:, 2], d[:, 3]])
    return tracks


def load_scene(name: str) -> Scene:
    """按场景名载入轨迹。可用场景见 :data:`SCENES`。"""
    if name not in SCENES:
        raise KeyError(f"未知场景 {name!r}，可选：{list(SCENES)}")
    dataset, rel = SCENES[name]
    path = DATA / rel
    if not path.exists():
        hint = ("请先运行 python datasets/download_datasets.py"
                + (" 和 python datasets/build_ucy_metric.py"
                   if dataset == "UCY" else ""))
        raise FileNotFoundError(f"缺少数据文件 {path}\n{hint}")

    loader = _load_eth if dataset == "ETH" else _load_ucy
    return Scene(name=name, dataset=dataset, source=path,
                 tracks=loader(path))


def load_all(names: list[str] | None = None) -> list[Scene]:
    """批量载入，默认全部 5 个场景。"""
    return [load_scene(n) for n in (names or list(SCENES))]


if __name__ == "__main__":
    for sc in load_all():
        s = sc.summary()
        print(f"{s['场景']:12s} [{s['数据集']}] 行人 {s['行人数']:4d}  "
              f"点 {s['轨迹点数']:6d}  {s['时长/s']:6.1f}s  "
              f"x{s['x范围/m']}  y{s['y范围/m']}")
