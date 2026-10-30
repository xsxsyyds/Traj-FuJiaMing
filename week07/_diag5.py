# -*- coding: utf-8 -*-
"""看 5 架机的径向距离与到目标距离随时间的变化。"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_sphere as S

out = S.run("orca", n=20, t_end=60.0, verbose=False)
traj = out["traj"]; goal = out["goal"]; t = out["t"]; p0 = out["p0"]
print("t      |p| 半径 min/median/max     到目标距离 min/median/max   平均速度")
v = np.diff(traj, axis=0) / 0.1
for k in range(0, len(t) - 1, 50):
    r = np.linalg.norm(traj[k], axis=1)
    d = np.linalg.norm(traj[k] - goal, axis=1)
    sp = np.linalg.norm(v[k], axis=1)
    print(f"{t[k]:5.1f}   {r.min():5.2f} {np.median(r):6.2f} {r.max():6.2f}"
          f"     {d.min():5.2f} {np.median(d):6.2f} {d.max():6.2f}"
          f"      {sp.mean():4.2f}")
print("\n到达:", out["metrics"]["arrived"], "/", out["metrics"]["n"])
print("最小间隔:", round(out["metrics"]["min_separation"], 3))
