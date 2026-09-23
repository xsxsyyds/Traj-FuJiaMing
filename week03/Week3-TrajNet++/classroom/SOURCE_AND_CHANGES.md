# 来源与修改

- 上游仓库：https://github.com/vita-epfl/trajnetplusplusbaselines
- 下载日期：2026-09-23
- 固定提交：99a6e9d8675face1aeeb17227b73dd3d1267f463
- 许可证：MIT，原 LICENSE 与 README.rst 原样保留。
- 数据：本课程目录 `课件/Week-3/实验数据/circle-10m-64-1.txt`，未从互联网另找相似数据替代。

## 对上游文件的唯一修改

`trajnetbaselines/__init__.py` 中移除自动执行 `from . import classical`，替换为说明注释。原因是只训练 LSTM 时，原入口仍会强制导入 ORCA 的 rvo2 与 socialforce 等非本课依赖。原 classical 模块源码全部保留，使用这些经典模型时仍需自行安装对应依赖。

模型、网格池化、损失与原训练器未改动。学生后续修改需另外记录。

## 新增课堂文件

- `prepare_classroom.py`：格式转换、先划分后截窗、数据审计。
- `run_classroom.py`：固定小模型、CPU 线程、种子和窗口，调用原训练器，记录总耗时。
- `evaluate_classroom.py`：仅历史输入的递推预测，与匀速基线共同评价主要目标行人，输出图和 JSON。
- `requirements-classroom.txt`：课堂依赖版本；未使用原 setup.py 的 torch 1.10.0 固定项。
- `DATA_BLOCK/circle_classroom/`：由原始数据转换，单位和帧率假设见说明。
- `classroom/` 与中文说明：来源、原始数据、数据审计、验证记录。

压缩包不包含 .git、虚拟环境、依赖缓存、测试生成的权重或个人电脑路径日志。请阅读安装说明，不能将它视为离线免安装软件。
