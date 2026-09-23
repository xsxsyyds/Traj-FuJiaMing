# week03 — Social LSTM 的邻居信息有用吗？

本目录存放第三周课堂实践的材料。任务不是在圆环轨迹上自己建模，而是在给定的
TrajNet++ 代码里**只改一个方向**，看邻居信息的处理方式如何影响预测误差。

## 目录

| 路径 | 内容 |
|---|---|
| `Week3-TrajNet++/` | 课程下发的课堂包（上游源码 + 课程数据 + 训练/评价入口），保持原样 |
| `Week3-TrajNet++/先读我-课堂说明.md` | 课堂包自带说明：安装、作业方向、数据说明、源码阅读要点 |
| `Week3-TrajNet++/classroom/SOURCE_AND_CHANGES.md` | 上游仓库、固定提交、对上游的唯一修改 |
| `Week3-TrajNet++/classroom/data_audit.json` | 原始文件哈希、帧映射、划分与全部假设 |
| `Week3-TrajNet++/classroom/验证记录.md` | 课堂包作者在本机的验证记录与基线数值 |
| `Week3-TrajNet++/trajnetbaselines/lstm/gridbased_pooling.py` | 邻居信息汇总的核心实现，本周主要改动对象 |
| `Week3-TrajNet++/trajnetbaselines/lstm/trainer.py` | 上游训练器，`--cell_side`/`--n`/`--type` 等参数入口 |
| `README.md` | 本文件：任务解读与运行方式 |
| `作业.md` | 本周提交正文（改动说明、误差表、对比图、结论） |

## 本周任务（课堂 PPT 解读）

题目：**Social LSTM 中的邻居信息有用吗？** 60 分钟课堂作业。

1. **观察邻居信息** —— 看模型怎样汇总邻居信息，想一想哪里可以改。
2. **任选一个调整** —— 去掉邻居信息、调整邻域范围或网格大小，也可以尝试自己的邻居筛选或汇总方法。
3. **比较修改前后** —— 分别训练，比较 ADE / FDE 和轨迹，解释结果。

共同条件（必须遵守）：

- 同一数据划分、8 点历史、12 点未来、同一批测试样本；
- 验证集用于调整，预测只用当前及过去信息，其他训练条件尽量一致；
- **只选一个方向，不要求结果一定变好**；课堂使用提供的教学代码。

提交到 `week03/`：对比图 + ADE/FDE + 修改代码 + 简述改动和发现，并注明 AI 辅助。

参考源码：EPFL VITA TrajNet++ Baselines 官方仓库，含 Social LSTM 实现。

## 可选的修改方向（课堂包给出）

| 方向 | 参数或文件 | 要回答的问题 |
|---|---|---|
| 无交互基线 | `--type vanilla` | 邻居信息是否帮助预测？此比较同时改变网络结构 |
| 邻域范围 | `--cell_side`、`--n` | 需要关注多远？ |
| 网格分辨率 | 同时调整上述两项，固定总边长 | 空间划分多细比较合适？ |
| 池化规则 | `trajnetbaselines/lstm/gridbased_pooling.py` | 同格多个邻居怎样汇总？ |
| 相同结构的信息消融 | 在池化模块中屏蔽邻居特征，重新训练 | 尽量排除网络参数量差异后，邻居是否仍有贡献？ |

## 数据与划分（课堂包给定，不自行更改）

- 原始文件 `classroom/raw/circle-10m-64-1.txt`：行人 ID、原始帧号、x、y、附加列（未使用）。
  64 人、425 个连续原始帧（37–461），轨迹同步完整。该文件与 `../week02/data/circle-10m-64-1.txt`
  逐字节相同（sha256 `05d52540…14d790`）。
- 已转换数据 `DATA_BLOCK/circle_classroom/`：先按采样后的时间轴划分 60%/20%/20%，
  再截取 8 个历史点与 12 个未来点，每隔 2 个原始帧采样一次，窗口步长 8 个采样点。
- 主要预测对象固定为 ID 1、17、33、49，其余 60 人保留为邻居。训练/验证/测试分别
  56/12/12 个场景。
- **单位与帧率未确认**：坐标乘 0.01 由厘米换算为米、原始 25 fps 都是待确认假设。
  确认之前，邻域与误差只能解释为换算后坐标单位。

## 运行

课堂包建议 Python 3.9/3.10 + PyTorch CPU，不要用 3.12/3.13：

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-classroom.txt
.\.venv\Scripts\python.exe run_classroom.py --output baseline
.\.venv\Scripts\python.exe evaluate_classroom.py OUTPUT_BLOCK/circle_classroom/lstm_social_baseline.pkl --split val
```

方案确定后再把 `--split val` 改为 `--split test` 报告测试集结果，不要反复根据测试结果调模型。
每个实验用不同的 `--output` 名称，避免覆盖已有模型。

## 需要留意的实现差异

- 上游训练器会用**真实邻居未来**做教师强制，并含远距离行人筛选；课堂评价脚本只传入
  历史，所有行人的未来都由模型递推，保留全部 64 人。各组统一采用这个评价协议。
- 上游网格用**索引赋值**填格，同格多邻居不是标准的求和池化。求和、平均、最大值等规则
  需要在代码中明确实现，并检查空格、自己不作为邻居、同格多人的处理及梯度传播。
- 上游 `--front` 在当前实现中只是把网格移到坐标轴一侧，相关朝向归一化代码被注释掉了，
  未经修改不能称为"行人朝向前方"。
- 训练日志里的 `loss` / `test_loss` 是损失数值，不能直接当作 ADE/FDE；日志里的
  `test_loss` 实际仍算在验证集上。

## 说明

- 源码来源：<https://github.com/vita-epfl/trajnetplusplusbaselines>，固定提交
  `99a6e9d8675face1aeeb17227b73dd3d1267f463`，MIT 许可证（见包内 `LICENSE`）。
  课堂包对上游的唯一改动是 `trajnetbaselines/__init__.py` 去掉自动导入经典模型。
- 课程数据的授权范围沿用原数据提供方约定，本包不赋予额外公开再分发权利。
- AI 辅助情况：课堂包的梳理、本文件的编写由 AI 辅助完成；实验改动、训练与结论判断
  由本人执行与核对。具体到每周提交的 AI 辅助声明写在 `作业.md` 中。
