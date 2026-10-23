# -*- coding: utf-8 -*-
"""
场景一：窄长廊，人车迎面通过。

布局
----
    x=0 ────────────────────────────────────────── x=L
        [6 名行人 →]                    [1 辆车 ←]

行人从左侧出发向右，车辆从右侧出发向左，双方都要到达对面。
无路口、无信号灯；人可侧移或等待，车可减速或停车。

运行
----
    python run_corridor.py            # 跑全部宽度
    python run_corridor.py --quick    # 只跑 3.5 m
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))

from unified_sfm import (  # noqa: E402
    Agent, Corridor, PARAMS, DT, simulate, summary,
)

L = 60.0            # 长廊长度 [m]
N_PED = 6
T_END = 150.0
PED_SPACING = 2.4   # 行人纵向间距 [m]（太密会被让行规则判成互相阻挡）


def build(width: float, keep_right: bool = True, seed: int = 0):
    rng = np.random.default_rng(seed)
    road = Corridor(L, width)
    agents = []

    # 行人：左侧起步，靠各自右侧（+x 行进时右侧为 -lat）
    for i in range(N_PED):
        s = 1.2 + PED_SPACING * i + rng.uniform(-0.15, 0.15)
        lat = -0.65 + rng.uniform(-0.08, 0.08)
        pos = road.to_world(s, lat)
        agents.append(Agent(
            aid=i, kind="ped", pos=pos, vel=np.array([0.4, 0.0]),
            heading=0.0, sigma=+1.0, lat_pref=-min(width / 2 - 0.4, 1.45),
            goal=np.array([L + 2.0, pos[1]])))

    # 车辆：右侧起步，靠自身右侧（-x 行进时右侧为 +lat）
    lat_car = 0.0
    pos = road.to_world(L - 3.0, lat_car)
    agents.append(Agent(
        aid=99, kind="car", pos=pos, vel=np.array([-1.0, 0.0]),
        heading=np.pi, sigma=-1.0, lat_pref=lat_car,
        goal=np.array([-2.0, pos[1]])))

    for a in agents:
        t = road.tangent(a.pos) * a.sigma
        a.heading = float(np.arctan2(t[1], t[0]))
    return agents, road


def run_case(width, keep_right=True, label="", t_end=T_END):
    agents, road = build(width, keep_right)
    hist = simulate(agents, road, duration=t_end, sample_dt=DT,
                    keep_right=keep_right, use_yield=True)
    rep = summary(agents, hist, road, label=label or f"W={width:g}")
    rep["width"] = width
    rep["keep_right"] = keep_right
    rep["corridor_length"] = L
    return rep, hist, agents, road


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cases = [
        (5.0, True, "宽 5.0 m"),
        (3.5, True, "窄 3.5 m（任务设定）"),
        (2.2, True, "极窄 2.2 m（空间不足）"),
        (3.0, False, "3.0 m + 不靠右（规则失效）"),
        (3.0, True, "3.0 m + 靠右（修正尝试）"),
    ]
    if args.quick:
        cases = [cases[1]]

    out = []
    for w, kr, label in cases:
        print(f"跑 {label} ...", flush=True)
        rep, hist, agents, road = run_case(w, kr, label)
        out.append(rep)
        print(f"   到达 行人 {rep['arrived_ped']}/{rep['n_ped']} "
              f"车 {rep['arrived_car']}/{rep['n_car']}  "
              f"用时 {rep['t_last_arrival']}  "
              f"碰撞帧 {rep['collision_frames']}  "
              f"越界 {rep['boundary_max_intrusion']:.3f} m", flush=True)
        np.savez_compressed(
            RESULTS / f"corridor_{w:g}_{'right' if kr else 'center'}.npz",
            t=hist["t"], pos=hist["pos"], vel=hist["vel"],
            heading=hist["heading"],
            kinds=np.array([a.kind for a in agents]),
            arrived=np.array([a.arrived for a in agents]),
            t_arrive=np.array([a.t_arrive for a in agents]))

    (RESULTS / "corridor_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "corridor_report.txt").write_text(
        "\n".join(
            [f"{r['label']:28s} 到达 人{r['arrived_ped']}/{r['n_ped']} "
             f"车{r['arrived_car']}  用时 {r['t_last_arrival']}  "
             f"碰撞帧 {r['collision_frames']}  最小净空 {r['min_clearance']:.3f}  "
             f"越界 {r['boundary_max_intrusion']:.3f}"
             for r in out]), encoding="utf-8")
    print("已写出 results/corridor_summary.json")


if __name__ == "__main__":
    main()
