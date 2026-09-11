# SYSU-Code

中山大学课程代码

## 行人轨迹数据分析

基于 ETH 与 UCY 行人轨迹数据集的交通建模分析作业，按周提交。

### 数据集

| 数据集 | 序列 | 主数据文件 | 单位 |
|---|---|---|---|
| ETH (EWAP) | seq_eth、seq_hotel | `obsmat.txt` | 米 |
| UCY Crowd | zara01、zara02、students03 | `<场景>.txt` | 米 |

放在 `datasets/`，来源、格式与引用详见 [datasets/README.md](datasets/README.md)。

> UCY 官方发布的 `.vsp` 是图像像素坐标（带透视压缩），不能直接算物理量。仓库中另有由 `build_ucy_metric.py` 生成的米制版本 `<场景>.txt`，**分析一律用米制版本**。

### 五个动作

| 编号 | 动作 | 算什么 | 产出 |
|---|---|---|---|
| A1 | 画轨迹 | 每个个体一条线 | 轨迹图 |
| A2 | 算个体量 | 速度 / 方向 / 加速度 | 速度分布图 |
| A3 | 算对子量 | 最近邻间距、TTC | 间距 / TTC 分布 |
| A4 | 画基本图 | 密度 / 流率 / 速度 | q-k-v 图 |
| A5 | 行为分析 | 成行、避让、震荡、瓶颈 | 现象图 |

### 目录约定

```
common/traj_io.py     轨迹数据统一读取模块（供各周脚本复用）
datasets/             数据集、说明与构建脚本
week01/ … week08/     每周一个目录，含脚本、图表与结论
```

每周目录内放三样：图或视频、一段结论、代码。

### 进度

| 任务 | 内容 | 状态 |
|---|---|---|
| A1 | 画轨迹 | ✅ 完成，见 [week01/](week01/) |
| A2 | 算个体量 | 待做 |
| A3 | 算对子量 | 待做 |
| A4 | 画基本图 | 待做 |
| A5 | 行为分析 | 待做 |

```bash
python datasets/download_datasets.py    # 获取原始数据（约 228 MB）
python datasets/build_ucy_metric.py     # 生成 UCY 米制版本
python week01/a1_trajectories.py        # A1 画轨迹图
```
