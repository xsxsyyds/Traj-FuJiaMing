# -*- coding: utf-8 -*-
"""两机对冲单元测试：验证三维 ORCA 半空间与速度求解。"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orca3d import orca_halfspace, solve_velocity, project_velocity

R = 1.0          # 两机联合半径（0.5+0.5）
TAU, DT, VMAX = 5.0, 0.1, 2.0

# --- 1) 半空间基本检查：正面对冲 ---
pA = np.array([10.0, 0.0, 0.0])
pB = np.array([-10.0, 0.0, 0.0])
vA = np.array([-2.0, 0.0, 0.0])
vB = np.array([2.0, 0.0, 0.0])
hs = orca_halfspace(pB - pA, vA - vB, R, TAU, DT, vA)
print("半空间 n =", np.round(hs[0], 4), " point =", np.round(hs[1], 4))
n, pt = hs
print("当前速度是否违反:", n @ (vA - pt) < 0, " 值=", round(float(n @ (vA - pt)), 4))

new_v = solve_velocity(pA, vA, np.array([-2.0, 0.0, 0.0]),
                       [(pB, vB, 0.5, False)], 0.5, VMAX, TAU, DT, 0.0)
print("解出的新速度:", np.round(new_v, 4), " |v|=", round(float(np.linalg.norm(new_v)), 3))
print("是否满足约束:", round(float(n @ (new_v - pt)), 4))

# --- 2) 两机闭环仿真 ---
def sim(offset=(0.0, 0.05, 0.0), steps=300):
    pa = np.array([10.0, 0.0, 0.0])
    pb = np.array([-10.0, 0.0, 0.0]) + np.array(offset)
    va = np.zeros(3); vb = np.zeros(3)
    goal_a = -pa.copy(); goal_b = -pb.copy()
    mins = 1e9
    for k in range(steps):
        # 期望速度
        da = goal_a - pa; db = goal_b - pb
        pa_pref = VMAX * da / max(np.linalg.norm(da), 1e-9)
        pb_pref = VMAX * db / max(np.linalg.norm(db), 1e-9)
        na = solve_velocity(pa, va, pa_pref, [(pb, vb, 0.5, False)],
                            0.5, VMAX, TAU, DT, 0.05)
        nb = solve_velocity(pb, vb, pb_pref, [(pa, va, 0.5, False)],
                            0.5, VMAX, TAU, DT, 0.05)
        va, vb = na, nb
        pa = pa + va * DT
        pb = pb + vb * DT
        d = np.linalg.norm(pa - pb)
        mins = min(mins, d)
        if k % 30 == 0:
            print(f"  t={k*DT:4.1f} d={d:6.3f} |va|={np.linalg.norm(va):4.2f} "
                  f"|vb|={np.linalg.norm(vb):4.2f}")
    return mins, pa, pb

print("\n两机仿真（B 偏移 5 cm）:")
m, fa, fb = sim()
print(f"最小间距 {m:.3f} m   A 终点 {np.round(fa,2)}  B 终点 {np.round(fb,2)}")

print("\n两机仿真（B 偏移 50 cm）:")
m2, fa2, fb2 = sim(offset=(0.0, 0.5, 0.0))
print(f"最小间距 {m2:.3f} m   A 终点 {np.round(fa2,2)}  B 终点 {np.round(fb2,2)}")
