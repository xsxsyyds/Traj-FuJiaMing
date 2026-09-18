# SYSU-Code

中山大学课程代码

## 行人轨迹分析

基于 ETH / UCY 数据集的轨迹分析与行为建模作业，按周提交。

### 目录结构

```
common/traj_io.py     轨迹数据统一读取模块（供各周脚本复用）
datasets/             ETH / UCY 数据集、说明与构建脚本
week01/               A1 画轨迹
week02/               圆环对趾：形成行为判断 → 建模 → 验证
week03/ … week08/     后续每周一个目录
```

每周目录内放三样：图或视频、一段结论、代码。

---

## 数据

| 数据集 | 序列 | 主数据文件 | 单位 |
|---|---|---|---|
| ETH (EWAP) | seq_eth、seq_hotel | `obsmat.txt` | 米 |
| UCY Crowd | zara01、zara02、students03 | `<场景>.txt` | 米 |
| 圆环对趾实验 | circle-10m-64 | `circle-10m-64-1.txt` | 厘米 |

来源、格式与引用详见 [datasets/README.md](datasets/README.md)。

> ⚠️ 两个容易踩的单位陷阱：
> - UCY 官方发布的 `.vsp` 是**图像像素坐标**（带透视压缩），不能直接算速度、间距、密度。仓库中另有由 `build_ucy_metric.py` 生成的米制版本 `<场景>.txt`。
> - 圆环实验的坐标单位是**厘米**，圆周半径约 1000 cm = 10 m；其第 2 列是**帧号**而非行人编号。

---

## 进度

| 周 | 任务 | 状态 |
|---|---|---|
| week01 | A1 画轨迹 | ✅ 完成 → [week01/](week01/) |
| week02 | 圆环对趾：形成行为判断 | ✅ 完成 |
| week02 | 圆环对趾：模型主张 + 实现验证 | 进行中 |
| — | A2 算个体量 | 待做 |
| — | A3 算对子量 | 待做 |
| — | A4 画基本图 | 待做 |
| — | A5 行为分析 | 待做 |

### 五个动作（轨迹分析主线）

| 编号 | 动作 | 算什么 | 产出 |
|---|---|---|---|
| A1 | 画轨迹 | 每个个体一条线 | 轨迹图 |
| A2 | 算个体量 | 速度 / 方向 / 加速度 | 速度分布图 |
| A3 | 算对子量 | 最近邻间距、TTC | 间距 / TTC 分布 |
| A4 | 画基本图 | 密度 / 流率 / 速度 | q-k-v 图 |
| A5 | 行为分析 | 成行、避让、震荡、瓶颈 | 现象图 |

---

## 复现

```bash
# 数据准备
python datasets/download_datasets.py     # 获取 ETH / UCY 原始数据（约 228 MB）
python datasets/build_ucy_metric.py      # 生成 UCY 米制版本
python datasets/check_ucy_units.py       # 校验 UCY 坐标单位（可选）

# week01
python week01/a1_trajectories.py         # A1 画轨迹图

# week02
python week02/analyze_circle.py          # 圆环对趾轨迹与几何统计
```
