# -*- coding: utf-8 -*-
"""逐步检查：每步解完速度后，真实的最小接近距离是否 >= 约束半径。"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orca3d import solve_velocity

VMAX, TAU, DT = 2.0, 5.0, 0.1
MARGIN = 0.05
R_CONSTRAINT = 1.0 + MARGIN          # 0.5 + 0.5 + margin


def min_dist_over_tau(p_rel, u_rel, tau):
    ww = float(u_rel @ u_rel)
    if ww < 1e-12:
        return float(np.linalg.norm(p_rel))
    t = float(p_rel @ u_rel) / ww
    t = min(max(t, 0.0), tau)
    return float(np.linalg.norm(p_rel - t * u_rel))


pa = np.array([10.0, 0.0, 0.0])
pb = np.array([-10.0, 0.0, 0.05])
va = np.zeros(3); vb = np.zeros(3)
ga, gb = -pa.copy(), -pb.copy()

print(f"约束半径 {R_CONSTRAINT:.3f} m")
print(" t     d_now   min over tau   ok?")
worst = 1e9
for k in range(200):
    da, db = ga - pa, gb - pb
    pa_pref = VMAX * da / max(np.linalg.norm(da), 1e-9)
    pb_pref = VMAX * db / max(np.linalg.norm(db), 1e-9)
    na = solve_velocity(pa, va, pa_pref, [(pb, vb, 0.5, False)],
                        0.5, VMAX, TAU, DT, MARGIN, prefilter=False)
    nb = solve_velocity(pb, vb, pb_pref, [(pa, va, 0.5, False)],
                        0.5, VMAX, TAU, DT, MARGIN, prefilter=False)
    va, vb = na, nb
    p_rel = pb - pa
    u_rel = va - vb
    m = min_dist_over_tau(p_rel, u_rel, TAU)
    worst = min(worst, m)
    if k % 10 == 0 or m < R_CONSTRAINT:
        print(f"{k*DT:5.1f}  {np.linalg.norm(p_rel):7.3f}  {m:10.3f}   "
              f"{'OK' if m >= R_CONSTRAINT - 1e-6 else 'VIOLATION'}")
    pa = pa + va * DT
    pb = pb + vb * DT

print(f"\n全程最坏 min-over-tau = {worst:.4f} m")
print(f"实际最小间距 = {np.linalg.norm(pa-pb):.3f} m（终点处）")
