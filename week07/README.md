# week07 —— 三维球面 100 机对趾冲突（ORCA）

100 架无人机均匀分布在半径 10 m 的球面上，同时飞向各自的对趾点。
上层算法用 **ORCA**，对比 Direct（不避让）与 Repulsion（排斥场）。

详细分析见 [`作业.md`](作业.md)。

## 文件

| 文件 | 作用 |
|---|---|
| `orca3d.py` | 球面 ORCA：切平面 + RVO2 半平面/线性规划 + TTC 邻居预筛 |
| `sim_sphere.py` | 场景、三个模型、五项指标统计 |
| `plot_results.py` | 三维轨迹图 / 指标对比 / 最小间隔曲线 |
| `make_video.py` | mp4 动画（左 ORCA，右 Direct） |

## 场景参数

| 项 | 值 |
|---|---|
| 数量 | 100（Fibonacci 球面均匀分布） |
| 球半径 | 10 m（题目未给，自选） |
| 安全半径 | 0.5 m → 最小间隔 1 m |
| vmax | 2 m/s |
| Δt | 0.1 s |
| τ | 5 s |
| 初速 | 0 |
| 终止 | 距目标 ≤ 1 m 停留 |
| 限时 | 120 s |

## 结果（100 架，R = 10 m）

| 模型 | 到达率 | 最小间隔 [m] | 侵犯 1 m 的机对数 | 全部到达 [s] | 平均路程 [m] |
|---|---|---|---|---|---|
| **ORCA** | **100/100** | **1.062** | **0 / 4950** | 21.3 | 30.89 ± 0.79 |
| Direct | 100/100 | 0.022 | 286 / 4950 | 15.3 | 30.60 ± 0.00 |
| Repulsion | 100/100 | 0.067 | 47 / 4950 | 21.9 | 32.08 ± 0.69 |

## 实现上的四个关键点

1. **切平面投影**：把球面问题化为每个 agent 局部切平面上的标准 2D ORCA；
   投影误差 d²/(2R) ≈ 5 cm（且方向保守）。
2. **TTC 预筛**：只对 τ 内可能相撞的邻居生成半平面，否则 100 机时线性规划退化为 O(N²)。
3. **静止邻居全责**：已到达停住的无人机不会让路，`share` 必须取 1.0；
   否则会出现"一架永远到不了"的死锁（实测 99/100 → 修复后 100/100）。
4. **5 cm 约束余量**：理论上限 1.0 m，离散化与 LP 不可行会吃掉几毫米；
   约束半径取 0.55 m，**评价口径仍是 1.0 m**。

## 图与视频

| 文件 | 内容 |
|---|---|
| `results/fig1_trajectories_3d.png` | 三个模型的三维轨迹 |
| `results/fig2_metrics.png` | 四项指标柱状对比 |
| `results/fig3_min_separation.png` | 最小间隔随时间变化 |
| `results/sphere_orca_vs_direct.mp4` | ORCA vs Direct 动画 |
| `results/metrics_table.csv` | 指标原始表 |

## 复现

```bash
python sim_sphere.py        # 三个模型
python plot_results.py      # 三张 PNG
python make_video.py        # mp4
```

可选参数：`--n 100 --radius 10 --t-end 120 --models orca direct repulsion`。

## 依赖

`numpy`、`matplotlib`、`imageio-ffmpeg`（提供 ffmpeg 可执行文件给 matplotlib 写 mp4）。
