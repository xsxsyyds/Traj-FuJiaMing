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
| `Week3-TrajNet++/plot_prediction.py` | 画单场景预测图：原始轨迹淡化、预测段加粗（见下文用法） |
| `results/` | 训练产物、评价指标与预测图（课堂包之外，不受包内 `.gitignore` 影响） |
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
  56/12/12 个场景（场景 id 连续编号：train 0–55、val 56–67、test 68–79）。
- **单位与帧率未确认**：坐标乘 0.01 由厘米换算为米、原始 25 fps 都是待确认假设。
  确认之前，邻域与误差只能解释为换算后坐标单位。

## 运行

课堂包建议 Python 3.9/3.10 + PyTorch CPU，不要用 3.12/3.13。本机使用 conda 环境
`social-lstm`（Python 3.9.23 + torch 1.12.1+cpu），完整配置指令见
`../week03_环境配置指令.md`：

```powershell
& "D:\Users\xsxsyyds\miniconda3\envs\social-lstm\python.exe" run_classroom.py --output baseline
& "D:\Users\xsxsyyds\miniconda3\envs\social-lstm\python.exe" evaluate_classroom.py OUTPUT_BLOCK/circle_classroom/lstm_social_baseline.pkl --split val
```

方案确定后再把 `--split val` 改为 `--split test` 报告测试集结果，不要反复根据测试结果调模型。
每个实验用不同的 `--output` 名称，避免覆盖已有模型。

## 画预测图（`plot_prediction.py`）

本目录新增 `plot_prediction.py`，把单个场景的**原始轨迹与预测段**画在一起：原始轨迹全程
降低不透明度与饱和度，预测段加粗并保持满饱和度，用于直观呈现某一次预测的好坏。

```powershell
# 列出可用场景 id（不需要模型，几秒出结果）
python plot_prediction.py --split val --list-scenes

# 出图：取景半径 2.5 个换算单位，背景淡画其余行人
python plot_prediction.py OUTPUT_BLOCK/circle_classroom/lstm_social_baseline.pkl `
    --split val --scene-id 56 --zoom 2.5 --all-peds --label baseline
```

主力参数：

| 参数 | 作用 |
|---|---|
| `--scene-id` | 场景 id。train 0–55、val 56–67、test 68–79（按文件行序连续编号） |
| `--all-preds` | **画全部 64 人的轨迹与预测，每人一种颜色**（按行人 ID 稳定取色） |
| `--zoom PAD` | 以轨迹为中心取景，四周留 PAD 个换算单位；`--all-preds` 时以全部行人为取景依据 |
| `--all-peds` | 淡灰画出其余行人的原始轨迹，交代场景背景（只画真实轨迹，不画预测） |
| `--gt-alpha` / `--gt-sat` | 原始轨迹的不透明度 / 饱和度系数（默认 0.45 / 0.45） |
| `--gt-lw` / `--pred-lw` | 原始轨迹 / 预测段的线宽（默认 1.3 / 3.4） |
| `--show-cv` | 额外画匀速外推基线（深蓝虚线） |
| `--panorama` | 左右对照：左 = 该行人完整行程（425 帧全景），右 = 本场景窗口与预测。**用来解释「一个场景只是整段行程的一小截」** |
| `--save-npz` | 同时存下 `gt/history/pred/cv` 数组，之后改配色重绘不必再加载模型 |

输出默认写入 `../results/pred_<label>_<split>_scene<id>.png`（dpi=300）。
刻意放在课堂包目录之外 —— 包自带 `.gitignore` 含有 `*.png`，放在包内会被忽略掉。

评价协议与 `evaluate_classroom.py` 逐行一致：只传 8 帧历史、不传真实未来、64 个邻居全部保留、
单次均值 rollout 预测 12 帧，因此图上标出的 ADE/FDE 可与评价脚本的输出对齐。

## 模型预测的是「全部行人」，不是只有一个人

这一点容易误会，代码里可以确认：

- `LSTM.forward()` 的返回值文档写着 `pred_scene : Tensor [pred_length, num_tracks, 2]`，
  说明是 *Forecast the entire sequence*，即一次前向就为场景里**每个行人**都输出了预测。
- `evaluate_classroom.py` 里的 `positions[-12:, 0]` **只是取第 0 列**，因为 TrajNet++ 的官方
  指标（ADE/FDE）只评 primary 行人。这不是"模型只预测了一个人"。
- 想看到全部 64 人的预测，用 `--all-preds`。

另外，`--all-preds` 的配色**按行人 ID 取色**（黄金角在色环上散布，并避开红色区），所以
同一个人在不同场景、左右两个面板里的颜色都是一致的；primary 固定用深红加粗。

## 数据是「64 人圆环对趾」，但一个场景只有 1.5 秒

这句话最容易造成误会，实测数据如下：

- 原始文件 `classroom/raw/circle-10m-64-1.txt`：**64 人、425 个连续帧（37–461）**。首帧 64 个
  起点拟合圆半径 **10.07 m**（各点到圆心距离中位 10.09 m、标准差仅 0.14 m）；四个主预测对象
  的终点距各自起点的对趾点只差 0.12–0.31 m，迂回系数 1.09–1.28（接近直穿圆心）。
  确认就是「64 人从半径 10 m 圆周出发、奔向对趾点」。
- **但一个「场景」只是 20 个采样点**。以 val 场景 56（primary = ID 1）为例：覆盖原始帧
  291–329，**时间跨度约 1.56 秒，占整段约 17 秒行程的 9%**；primary 在这窗口里只移动了
  2.03 m、绕圆心转过 3.3°。
- 所以单场景预测图里只看到一小截弧段是正常的，不是数据缺失。用 `--panorama` 可以把
  全景与窗口画在一起对照。

`--list-scenes` 可列出各场景的 id 与规模（每个场景恒为 64 人 / 20 帧）。

## results/ 里的图

| 文件 | 内容 |
|---|---|
| `data_circle_overview.png` | 数据诊断：左=64 人完整轨迹（425 帧），右=单个场景窗口 |
| `pred_baseline_val56_primary.png` | 只画 primary（ID 1）：原始轨迹淡化 + 预测段加粗 |
| `pred_baseline_val56_allpeds.png` | **全部 64 人，每人一色**，原始轨迹淡化 + 预测段加粗 |
| `pred_baseline_val56_panorama.png` | 全景对照（primary）：完整行程 vs 本场景窗口 |
| `pred_baseline_val56_allpeds_panorama.png` | 全景对照（全部行人着色） |
| `pred_baseline_test68_primary.png` | 测试集首场景的同类图 |
| `lstm_social_baseline.pkl.*` | 训练产物、训练日志、验证集指标与上游自带评价图 |

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
