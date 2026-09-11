# SYSU-Code

中山大学课程代码

## 行人轨迹数据分析

基于 ETH 与 UCY 行人轨迹数据集的交通建模分析作业，按周提交。

### 数据集

| 数据集 | 序列 | 主数据文件 |
|---|---|---|
| ETH (EWAP) | seq_eth、seq_hotel | `obsmat.txt` |
| UCY Crowd | zara01、zara02、students03 | `*.vsp` |

放在 `datasets/`，来源、格式与引用详见 [datasets/README.md](datasets/README.md)。
重新获取数据：`python datasets/download_datasets.py`。

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
week01/ … week08/     每周一个目录
datasets/             数据集与说明
```

每周目录内放三样：图或视频、一段结论、代码。
